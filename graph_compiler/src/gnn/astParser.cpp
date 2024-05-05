#include "astParser.h"
#include "args.h"
#include "nodeUtils.h"
#include <cassert>

namespace GNN {

AstParser::AstParser(GraphGenerator *graphGenerator) : graphGenerator(graphGenerator) {
    pragmaParser = graphGenerator->pragmaParser.get();
    variableMapper = graphGenerator->variableMapper.get();
    derefTracker = graphGenerator->derefTracker.get();
}

void AstParser::makeVariable(SgInitializedName *varDec) {
    if (variableDeclarationsProcessed.count(varDec)) {
        return;
    }

    variableDeclarationsProcessed.insert(varDec);

    // add the variable to the graph
    // as local variable can consume resources
    Node *localVariable = variableMapper->createLocalVariableNode(varDec);
    new VariableDeclareEdge(localVariable);
}

void AstParser::catchBreakStatements() {
    std::queue<MergeStartEdge *> currentBreakMergeEdges = breakMergeEdges.top();
    breakMergeEdges.pop();
    while (!currentBreakMergeEdges.empty()) {
        MergeStartEdge *edge = currentBreakMergeEdges.front();
        currentBreakMergeEdges.pop();
        new MergeEndEdge(edge);
    }
}

// Handle every line of code in a basic block
void AstParser::handleBB(std::vector<SgStatement *> statements) {

    // we only reuse array dereferences inside the same BB
    derefTracker->makeNewDerefMap();

    // the code currently can handle a maximum of 1 return statement
    // per function
    int numberOfReturnStatements = 0;

    // for each line of code in a basic block
    for (SgStatement *statement : statements) {
        std::cerr << statement->unparseToString() << std::endl;
        if (SgPragmaDeclaration *pragmaDec = isSgPragmaDeclaration(statement)) {
            continue;
        }

        // if the line of code is 1 or more variable declarations
        if (SgVariableDeclaration *varDecStatement = isSgVariableDeclaration(statement)) {
            // for each variable declared in the line of code
            for (SgInitializedName *varDec : varDecStatement->get_variables()) {
                makeVariable(varDec);
                // if the variable is initialized
                if (SgExpression *expr = isSgExpression(varDec->get_initptr())) {
                    // if it is initialized with an expression
                    if (SgAssignInitializer *init = isSgAssignInitializer(expr)) {
                        Node *var = variableMapper->readVariable(varDec);
                        Node *rhs = readExpression(init->get_operand());
                        Node *writeNode = writeExpression(varDec, rhs);
                        new ControlFlowEdge(writeNode);
                    } else if (SgConstructorInitializer *init = isSgConstructorInitializer(expr)) {
                        std::cout << "Constructors are currently excluded" << std::endl;
                    } else if (auto *init = isSgAggregateInitializer(expr)) {
                        Node *variable = variableMapper->readVariable(varDec);
                        if(graphGenerator->checkArg(PROXY_PROGRAML)){
                            Node *bitcast = new BitcastNode();
                            new ControlFlowEdge(bitcast);
                            new DataFlowEdge(variable, bitcast);

                            // make empty func dec, the signature doesn't matter
                            SgType *returnType = SageBuilder::buildIntType();
                            SgFunctionParameterList *paramList = SageBuilder::buildFunctionParameterList();

                            // Create a function declaration
                            SgName functionName("memcopy");
                            SgFunctionDeclaration *funcDec =
                                SageBuilder::buildNondefiningFunctionDeclaration(functionName, returnType, paramList, NULL);

                            functionDecsNeeded.push(funcDec);
                            FunctionCallNode *funcCallNode = new FunctionCallNode();
                            FunctionCallEdge *funcCallEdge = new FunctionCallEdge(funcCallNode, funcDec);

                            new DataFlowEdge(bitcast, funcCallNode);

                            TypeStruct boolType = TypeStruct(DataType::INTEGER, 1);
                            Node *constant1 = new ConstantNode("Memcopy Bool", boolType);
                            new DataFlowEdge(constant1, funcCallNode);

                            TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                            Node *constant2 = new ConstantNode("Memcopy Constant Pointer", pointerType);
                            new DataFlowEdge(constant2, funcCallNode);

                            Node *constant3 = new ConstantNode("Memcopy Input ?", pointerType);
                            new DataFlowEdge(constant3, funcCallNode);
                        } else {
                            Node *var = variableMapper->readVariable(varDec);
                            Node *rhs = new ConstantNode("Array Initialization", var->getType());
                            Node *writeNode = writeExpression(varDec, rhs);
                            int numVals = init->get_initializers()->get_expressions().size();
                            writeNode->unrollFactor.full *= numVals;
                            new ControlFlowEdge(writeNode);
                        }

                    } else {
                        throw std::runtime_error("variable initialized in unknown way: " + expr->unparseToString());
                    }
                }
            }

            // if the line of code is an expression
        } else if (SgExprStatement *exprStatement = isSgExprStatement(statement)) {
            // read the expression
            readExpression(exprStatement->get_expression());

            // if its a return statement
        } else if (SgReturnStmt *returnStatement = isSgReturnStmt(statement)) {
            // record that a return statement has been found
            numberOfReturnStatements++;

            // process the expression
            // and if it returns a value, save it in the member variable
            // so whatever called handleStatements can find it
            functionReturn = readExpression(returnStatement->get_expression());

            // if its a for statement
        } else if (SgForStatement *forStatement = isSgForStatement(statement)) {
            pragmaParser->stackPragmas();
            // get the init statement
            SgStatementPtrList forInit = forStatement->get_for_init_stmt()->get_init_stmt();

            LocalScalarNode *iterator = getIteratorFromForInit(forInit);
            iterator->inIteratorBoundsRegion = true;

            // handle the init statement
            graphGenerator->newBB(true);
            handleBB(forInit);
            iterator->inIteratorBoundsRegion = false;

            new ProgramlBranchEdge();

            breakMergeEdges.push(std::queue<MergeStartEdge *>());

            // safely cast to the body to a bb
            SgBasicBlock *bb = isSgBasicBlock(forStatement->get_loop_body());
            if (!bb) {
                bb = SageBuilder::buildBasicBlock(forStatement->get_loop_body());
            }
            assert(bb);

            // get any pragmas in this bb and apply them
            pragmaParser->parsePragmas(bb);

            std::cerr << "enter loop cond" << std::endl;

            // unroll pragmas in a bb don't affect the condition
            // other pragmas (pipeline) do
            pragmaParser->enterLoopCondition();

            std::cerr << "loop cond entered" << std::endl;

            // we need the branch making control flow edges
            // and the comparison for adding pragma nodes to
            // and they're output by reference, so here's where
            // we store them
            Node *branchNode = nullptr;
            Node *comparisonNode = nullptr;

            // store the next instruction that is executed
            // so that we can get back to it
            // when we want to re-execute the condition
            PreLoopEdge *preLoopEdge = new PreLoopEdge();

            graphGenerator->newBB();
            derefTracker->makeNewDerefMap();
            handleConditional(forStatement->get_test_expr(), branchNode, comparisonNode);

            iterator->processComparison(comparisonNode);

            // finished handling the condition, so we can
            // mark any nodes below this point with the unroll factor
            pragmaParser->exitLoopCondition();

            if (pragmaParser->getUnrollFactor().first > 1) {
                PragmaNode *pragma = new UnrollPragmaNode(pragmaParser->getUnrollFactor().first);
                new UnrollPragmaEdge(pragma, comparisonNode);
            }

            graphGenerator->newBB();
            // handle the body of the for loop
            handleBB(bb->getStatementList());

            new ProgramlBranchEdge();

            if (forStatement->get_increment()->variantT() != V_SgNullExpression) {
                // unroll pragmas don't affect the loop increment
                pragmaParser->enterLoopInc();

                iterator->inIncrementRegion = true;

                graphGenerator->newBB(true);
                // handle the increment expression
                readExpression(forStatement->get_increment());

                iterator->inIncrementRegion = false;

                new ProgramlBranchEdge();

                pragmaParser->exitLoopInc();
            }

            // loop back to the first node we need to execute
            // to evaluate the condition,
            // and then mark the branch node as the predecessor
            new LoopBackEdge(preLoopEdge, branchNode);

            // unapply any pragmas from this bb
            pragmaParser->unstackPragmas();

            catchBreakStatements();

            graphGenerator->newBB();
            derefTracker->makeNewDerefMap();

            // if its a while statement
        } else if (SgWhileStmt *whileStmt = isSgWhileStmt(statement)) {
            pragmaParser->stackPragmas();

            // we need the branch making control flow edges
            // and the comparison for adding pragma nodes to
            // and they're output by reference, so here's where
            // we store them
            Node *branchNode = nullptr;
            Node *comparisonNode = nullptr;

            new ProgramlBranchEdge();

            // store the next instruction that is executed
            // so that we can get back to it
            // when we want to re-execute the condition
            PreLoopEdge *preLoopEdge = new PreLoopEdge();

            graphGenerator->newBB();
            SgStatement *condStatement = whileStmt->get_condition();
            SgExprStatement *condExprStatement = isSgExprStatement(condStatement);
            assert(condExprStatement);
            SgExpression *condExpr = condExprStatement->get_expression();
            handleConditional(condExpr, branchNode, comparisonNode);

            breakMergeEdges.push(std::queue<MergeStartEdge *>());

            // safely cast to the body to a bb
            SgBasicBlock *bb = isSgBasicBlock(whileStmt->get_body());
            if (!bb) {
                bb = SageBuilder::buildBasicBlock(whileStmt->get_body());
            }
            assert(bb);

            // get any pragmas in this bb and apply them
            pragmaParser->parsePragmas(bb);

            if (pragmaParser->getUnrollFactor().first > 1) {
                PragmaNode *pragma = new UnrollPragmaNode(pragmaParser->getUnrollFactor().first);
                new UnrollPragmaEdge(pragma, comparisonNode);            
            }

            graphGenerator->newBB();
            // handle the body of the for loop
            handleBB(bb->getStatementList());

            new ProgramlBranchEdge();

            // loop back to the first node we need to execute
            // to evaluate the condition,
            // and then mark the branch node as the predecessor
            new LoopBackEdge(preLoopEdge, branchNode);

            // unapply any pragmas from this bb
            pragmaParser->unstackPragmas();

            catchBreakStatements();
            derefTracker->makeNewDerefMap();

        } else if (SgDoWhileStmt *doWhileStmt = isSgDoWhileStmt(statement)) {
            throw std::runtime_error("Do While loops not currently supported");
        } else if (SgIfStmt *ifStmt = isSgIfStmt(statement)) {
            // merging control flow requires two edges
            // have to pass the first to the second
            // so its declared here
            MergeStartEdge *trueBodyMergeStartEdge;

            // branch node is an output by reference
            // this is where we store it
            Node *branchNode;

            // safely cast condition to an expression statement
            if (SgExprStatement *exprStatement = isSgExprStatement(ifStmt->get_conditional())) {
                // handle if statement condition
                handleConditional(exprStatement->get_expression(), branchNode);
            } else {
                throw std::runtime_error("Condition of an if statement wasn't an expression statement: " +
                                         ifStmt->get_conditional()->unparseToString());
            }
            // safely cast if body to a bb
            graphGenerator->newBB();

            ifStatementBreaks.push(false);
            if (SgBasicBlock *bb = isSgBasicBlock(ifStmt->get_true_body())) {
                // handle if body
                handleBB(bb->getStatementList());
            } else {
                std::vector<SgStatement *> statements;
                statements.push_back(ifStmt->get_true_body());
                handleBB(statements);
            }

            bool trueBodyBroke = ifStatementBreaks.top();
            ifStatementBreaks.pop();

            // if the if statement didn't break
            if (!trueBodyBroke) {
                new ProgramlBranchEdge();
                // mark that the previous node will need to merge
                trueBodyMergeStartEdge = new MergeStartEdge();
            }

            bool falseBodyBroke = false;
            // revert control flow predecessor to branch node
            new RevertControlFlowEdge(branchNode);
            // safely cast else body to a bb
            if (ifStmt->get_false_body()) {
                ifStatementBreaks.push(false);

                // handle else body
                graphGenerator->newBB();
                if (SgBasicBlock *bb = isSgBasicBlock(ifStmt->get_false_body())) {
                    handleBB(bb->getStatementList());
                } else {
                    std::vector<SgStatement *> statements;
                    statements.push_back(ifStmt->get_false_body());
                    handleBB(statements);
                }

                falseBodyBroke = ifStatementBreaks.top();
                ifStatementBreaks.pop();

                // if the if statement didn't break
                if (!falseBodyBroke) {
                    new ProgramlBranchEdge();
                }
            }

            // if both broke, don't look at rest of bb statements
            if (falseBodyBroke && trueBodyBroke) {
                break;
            } else if (!trueBodyBroke) {
                // mark that the next node to execute
                // has to merge the node marked in trueBodyMergeStartEdge
                new MergeEndEdge(trueBodyMergeStartEdge);
            }

        } else if (auto breakStatement = isSgBreakStmt(statement)) {
            new ProgramlBranchEdge();
            if (ifStatementBreaks.empty()) {
                std::runtime_error("Found a break statement not inside an if statement");
            }
            ifStatementBreaks.top() = true;
            breakMergeEdges.top().push(new MergeStartEdge());
        } else {
            throw std::runtime_error("Unsupported top level statement found: " + statement->unparseToString());
        }
    }

    if (numberOfReturnStatements > 1) {
        throw std::runtime_error("A maximum of one return statement per function is supported");
    }
}

Node *AstParser::handleFunctionCall(SgFunctionCallExp *funcCall) {
    // get the function declaration
    SgFunctionDeclaration *funcDec = AIR::getFuncDecFromCall(funcCall);

    if (graphGenerator->checkArg(INLINE_FUNCTIONS)) {
        // start parameter index at 0
        int paramIndex = 0;

        // foreach parameter in the function call
        for (SgExpression *argExpr : funcCall->get_args()->get_expressions()) {
            // get the matching parameter in the function declaration
            // using the param index
            SgInitializedName *varDec = funcDec->get_args()[paramIndex];

            // and link the parameter variable to
            // whatever expression was passed to that parameter
            // could be a constant, a variable, another function call
            variableMapper->setParamNode(varDec, argExpr);

            // and update the param index
            paramIndex++;
        }

        graphGenerator->newBB();
        // process each line of the function body
        handleBB(funcDec->get_definition()->get_body()->getStatementList());

        // when inlined, dataflow edges come from the actual node
        return functionReturn;
    } else {

        FunctionCallNode *funcCallNode = new FunctionCallNode();
        FunctionCallEdge *funcCallEdge = new FunctionCallEdge(funcCallNode, funcDec);
        functionDecsNeeded.push(funcDec);

        if (!decsToCalls.count(funcDec)) {
            decsToCalls[funcDec] = std::vector<FunctionCallNode *>();
        }
        decsToCalls[funcDec].push_back(funcCallNode);

        for (SgExpression *argExpr : funcCall->get_args()->get_expressions()) {
            funcCallEdge->parameters.push_back(argExpr);
        }

        funcCallNode->setType(funcDec->get_orig_return_type()->findBaseType()->unparseToString());

        // when not inlined, dataflow edges come from the call node
        return funcCallNode;
    }
}

// Handle an expression
// Expressions are deeply nested, with each expression having other expressions
// as operands it is dependant on.
// Leaf nodes are array indexing, variable reads, constants, or function calls
Node *AstParser::readExpression(SgExpression *expr) {
    // if the expression is an assignment operator
    if (SgAssignOp *assignOp = isSgAssignOp(expr)) {
        // handle the rhs of the assignment
        Node *rhs = readExpression(assignOp->get_rhs_operand());
        // and write to the lhs of the assignment
        Node *writeNode = writeExpression(assignOp->get_lhs_operand(), rhs);

        new ControlFlowEdge(writeNode);

        return rhs;
        // if the expression is a variable reference
    } else if (SgVarRefExp *varRef = isSgVarRefExp(expr)) {
        // get the actual variable
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();

        // get the pointer/memory element
        Node *variableNode = variableMapper->readVariable(varDec);

        // reading from some variables doesn't add a read node
        if (variableMapper->nonReadVariables.count(variableNode) > 0) {
            return variableNode;
        }

        // if its actually a variable, you need a read node
        Node *read = new ReadNode(variableNode);

        new ControlFlowEdge(read);

        // scalar variables need an address edge from an alloca
        // if mem elements, all reads need an element edge
        new ReadMemoryElementEdge(variableNode, read);

        return read;

        // if its an array indexing
    } else if (SgPntrArrRefExp *arrayIndex = isSgPntrArrRefExp(expr)) {

        // see if derefence has happened before in this BB
        // if yes, returns the already existing deref node
        // if not, it will add all the nodes for derefencing to the graph
        // and return the deref node
        DerefNode *deref = getDerefNode(arrayIndex);

        Node *memoryElement = deref->memoryElement;
        // resolved edge must be added before read
        // as we must specify read address before we can read
        // so we write destination as nullptr
        //
        // Proxy programl:
        // the outcoming pointer from the getelempointer connects to the read
        // getelemptr -> read data -> destination
        //
        // otherwise
        // get address -> specify address node -> mem elem
        // mem elem -> read data -> destination
        //
        // no connection from get address to read data
        Edge *resolvedEdge = new ResolvedMemoryAddressEdge(deref, nullptr, memoryElement);

        Node *readDataNode = new ReadNode(deref->baseTypeDependency);
        new ControlFlowEdge(readDataNode);

        // destination must be set manually as the node must
        // be added after the edge
        resolvedEdge->destination = readDataNode;

        // if mem elements, all reads need an element edge
        new ReadMemoryElementEdge(memoryElement, readDataNode);

        // the read data node provides the out-going data
        // from all of this
        return readDataNode;

    } else if (SgPointerDerefExp *pointerDeref = isSgPointerDerefExp(expr)) {
        SgVarRefExp *varRef = isSgVarRefExp(pointerDeref->get_operand());
        assert(varRef);
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
        Node *variable = variableMapper->readVariable(varDec);

        Node *readNode = new ReadNode(variable);
        new ParameterLoadDataFlowEdge(variable, readNode);

        new ControlFlowEdge(readNode);

        new ReadMemoryElementEdge(variable, readNode);

        return readNode;
        // if its a +=, -=, *= or /=
    } else if (Utils::isUpdateOp(expr->variantT())) {
        isUpdateOp = true;
        // add the correct arithmetic node
        Node *arithmeticNode = readArithmeticExpression(expr);

        // cast to a binary operation to get access to member variables
        SgBinaryOp *binaryOp = isSgBinaryOp(expr);

        // and write to the lhs of the assignment
        Node *writeNode = writeExpression(binaryOp->get_lhs_operand(), arithmeticNode);
        new ControlFlowEdge(writeNode);
        isUpdateOp = false;
        return writeNode;
        // if its an addition, subtraction, multiplication or division
    } else if (Utils::isBinaryArithmeticNode(expr->variantT())) {
        // Lots of operations consist partially of arithmetic nodes
        // so its in a function for reuse
        Node *arithmeticNode = readArithmeticExpression(expr);
        return arithmeticNode;
        // if its a function call
    } else if (SgFunctionCallExp *funcCall = isSgFunctionCallExp(expr)) {
        // function calls can be statements or expressions
        // so the same function is called from both places
        return handleFunctionCall(funcCall);
        // if its a ++ or --
    } else if (Utils::isIncOrDecOp(expr->variantT())) {
        // cast to a unary op to get access to member variables
        SgUnaryOp *unaryOp = isSgUnaryOp(expr);
        // read from the lhs
        Node *lhs = readExpression(unaryOp->get_operand());

        // add a constant
        Node *rhs;
        if (expr->variantT() == V_SgPlusPlusOp) {
            rhs = new ConstantNode("1", lhs->getType());
        } else {
            rhs = new ConstantNode("-1", lhs->getType());
        }

        // add the correct arithmetic node
        Node *arithmeticNode = processArithmeticExpression(V_SgPlusPlusOp, lhs, rhs);

        new ControlFlowEdge(arithmeticNode);

        // and write to the lhs
        Node *writeNode = writeExpression(unaryOp->get_operand(), arithmeticNode);
        new ControlFlowEdge(writeNode);

        return writeNode;
        // if its an integer constant
    } else if (SgIntVal *intVal = isSgIntVal(expr)) {
        TypeStruct intType = TypeStruct(DataType::INTEGER, 32);
        // add it to the graph
        Node *constant = new ConstantNode(std::to_string(intVal->get_value()), intType);
        return constant;

    } else if (SgLongLongIntVal *intVal = isSgLongLongIntVal(expr)) {
        // add it to the graph
        TypeStruct longIntType = TypeStruct(DataType::INTEGER, 64);
        Node *constant = new ConstantNode(std::to_string(intVal->get_value()), longIntType);
        return constant;
        // if its a boolean constant
    } else if (SgBoolValExp *boolVal = isSgBoolValExp(expr)) {
        TypeStruct boolType = TypeStruct(DataType::INTEGER, 1);
        Node *constant = new ConstantNode("Bool: " + std::to_string(boolVal->get_value()), boolType);
        return constant;
    } else if (SgDoubleVal *doubleVal = isSgDoubleVal(expr)) {
        TypeStruct doubleType = TypeStruct(DataType::FLOAT, 64);
        Node *constant = new ConstantNode(std::to_string(doubleVal->get_value()), doubleType);
        return constant;
    } else if (SgCastExp *castExpr = isSgCastExp(expr)) {
        std::string type = castExpr->get_type()->findBaseType()->unparseToString();

        Node *input = readExpression(castExpr->get_operand());

        if (input->getVariant() == NodeVariant::CONSTANT) {
            input->setType(type);
            return input;
        } else {
            return input;
        }
    } else if (expr->variantT() == V_SgDotExp || expr->variantT() == V_SgArrowExp) {
        SgBinaryOp *binaryOp = isSgBinaryOp(expr);
        assert(binaryOp);

        if(graphGenerator->checkArg(PROXY_PROGRAML)){
            DerefNode *dotDeref = getDotNode(binaryOp);

            if (binaryOp->get_rhs_operand()->get_type()->variantT() == V_SgArrayType) {
                Node *conversionDeref = new DerefNode();
                TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                conversionDeref->setType(pointerType);
                new ControlFlowEdge(conversionDeref);
                new DataFlowEdge(dotDeref, conversionDeref);

                Node *constantA = new ConstantNode("0", pointerType);
                Node *constantB = new ConstantNode("0", pointerType);

                new DataFlowEdge(constantA, conversionDeref);
                new DataFlowEdge(constantB, conversionDeref);

                return conversionDeref;
            } else {
                Node *read = new ReadNode(dotDeref->baseTypeDependency);
                new ResolvedMemoryAddressEdge(dotDeref, read, dotDeref->memoryElement);
                new ReadMemoryElementEdge(dotDeref->memoryElement, read);

                new ControlFlowEdge(read);

                return read;
            }
        }

        return readExpression(binaryOp->get_lhs_operand());

    } else if (auto commaExpr = isSgCommaOpExp(expr)) {
        readExpression(commaExpr->get_lhs_operand());
        return readExpression(commaExpr->get_rhs_operand());
    } else if (auto minusExpr = isSgMinusOp(expr)) {
        Node *input = readExpression(minusExpr->get_operand());
        if (input->getVariant() == NodeVariant::CONSTANT) {
            return input;
        }
        if (input->getType().dataType == DataType::FLOAT) {
            Node *node = new FNegNode();
            new DataFlowEdge(input, node);
            new ControlFlowEdge(node);
            return node;
        } else {
            return input;
        }

    } else if (auto sizeOfExpr = isSgSizeOfOp(expr)) {
        SgExpression *input = sizeOfExpr->get_operand_expr();

        bool found = false;
        while (!found) {
            if (auto varRef = isSgVarRefExp(input)) {
                found = true;
            } else if (auto dotExpr = isSgDotExp(input)) {
                input = dotExpr->get_rhs_operand();
            } else if (auto arrowExpr = isSgArrowExp(input)) {
                input = arrowExpr->get_rhs_operand();
            } else {
                throw std::runtime_error("Unsupport expr in size of expr: " + input->unparseToString());
            }
        }
        SgType *inputType = input->get_type();
        assert(inputType);

        int value;
        if (inputType->variantT() == V_SgArrayType) {
            SgArrayType *arrayType = isSgArrayType(inputType);
            assert(arrayType);
            std::cerr << "array type" << std::endl;
            int numElements = arrayType->get_number_of_elements();
            std::cerr << numElements << std::endl;
            int byteWidth;
            SgType *baseType = arrayType->get_base_type()->findBaseType();
            if (baseType->variantT() == V_SgTypeUnsignedChar) {
                byteWidth = 1;
            } else {
                throw std::runtime_error("Unsupported base type for sizeof expression: " + baseType->unparseToString());
            }
            value = byteWidth * numElements;
        } else {
            throw std::runtime_error("Unsupported top level type for sizeof expression: " +
                                     inputType->unparseToString());
        }
        TypeStruct sizeDesc = TypeStruct(DataType::INTEGER, 64);
        Node *node = new ConstantNode(std::to_string(value), sizeDesc);
        return node;
    } else if (auto nullExpr = isSgNullExpression(expr)) {
        return nullptr;
    } else if (auto addressOfExpr = isSgAddressOfOp(expr)) {
        SgExpression *input = addressOfExpr->get_operand();
        if (auto varRef = isSgVarRefExp(input)) {
            auto varDec = varRef->get_symbol()->get_declaration();
            return variableMapper->readVariable(varDec);
        } else if (auto arrayRef = isSgPntrArrRefExp(input)) {
            auto binaryOp = isSgBinaryOp(arrayRef);
            assert(binaryOp);
            Node *deref = getDerefNode(binaryOp);

            return deref;
        } else {
            throw std::runtime_error("Unsupported input for address of operation: " + input->unparseToString());
        }
    } else if (auto selectOp = isSgConditionalExp(expr)) {
        Node *condition = readExpression(selectOp->get_conditional_exp());

        Node *pred = condition;
        if (condition->getVariant() != NodeVariant::COMPARISON) {
            // there's actually a few types of nodes here to
            // convert to i1
            // TODO: check which node to actually add
            // instead of always a comparison
            ComparisonNode *comp = new ComparisonNode();

            Node *constant = new ConstantNode("0", condition->getType());
            ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(condition, constant, comp);
            comp->edge = edge;

            new ControlFlowEdge(comp);

            pred = comp;
        }

        Node *branch = new BranchNode();
        new ControlFlowEdge(branch);
        new DataFlowEdge(pred, branch);

        Node *a = readExpression(selectOp->get_true_exp());

        new ProgramlBranchEdge();

        MergeStartEdge *mergeStart = new MergeStartEdge();

        new RevertControlFlowEdge(branch);

        Node *b = readExpression(selectOp->get_false_exp());

        new ProgramlBranchEdge();

        new MergeEndEdge(mergeStart);

        SelectNode *select = new SelectNode();

        ArithmeticUnitEdge *arithmeticEdge = new ArithmeticUnitEdge(a, b, select);
        select->edge = arithmeticEdge;

        new ControlFlowEdge(select);
        return select;
    } else {
        throw std::runtime_error("Unsupported expression: " + expr->unparseToString());
    }
}

// if statements don't need the comparison returned
void AstParser::handleConditional(SgExpression *expr, Node *&branchNodeOut) {
    Node *comparisonNode;
    handleConditional(expr, branchNodeOut, comparisonNode);
}

// add all nodes for the comparison to the graph
// and return the branch and comparison by ref
// branch for looping control flow
// and comparison for adding pragma nodes to
void AstParser::handleConditional(SgExpression *expr, Node *&branchNodeOut, Node *&comparisonNodeOut) {
    if (SgFunctionCallExp *funcCall = isSgFunctionCallExp(expr)) {
        comparisonNodeOut = handleFunctionCall(funcCall);
    } else if (Utils::isComparisonOp(expr->variantT())) {
        SgBinaryOp *compOp = isSgBinaryOp(expr);
        Node *rhs = readExpression(compOp->get_rhs_operand());
        Node *lhs = readExpression(compOp->get_lhs_operand());

        ComparisonNode *comparisonNode = new ComparisonNode();

        comparisonNode->binaryComp = true;
        comparisonNode->lhs = lhs;
        comparisonNode->rhs = rhs;

        comparisonNodeOut = comparisonNode;

        ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(lhs, rhs, comparisonNode);
        comparisonNode->edge = edge;
    } else if (SgNotOp *notOp = isSgNotOp(expr)) {
        Node *input = readExpression(notOp->get_operand());

        comparisonNodeOut = new UnaryOpNode("Not");
        comparisonNodeOut->setType(input->getType());
        new DataFlowEdge(input, comparisonNodeOut);
    } else if (SgVarRefExp *varRef = isSgVarRefExp(expr)) {
        comparisonNodeOut = readExpression(varRef);
    } else if (auto *cast = isSgCastExp(expr)) {
        Node *lhs = readExpression(cast->get_operand());

        Node *rhs = new ConstantNode("0", lhs->getType());

        ComparisonNode *comparisonNode = new ComparisonNode();

        comparisonNode->binaryComp = true;
        comparisonNode->lhs = lhs;
        comparisonNode->rhs = rhs;

        comparisonNodeOut = comparisonNode;

        ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(lhs, rhs, comparisonNode);
        comparisonNode->edge = edge;
    } else {
        throw std::runtime_error("Unsupported conditional expression found: " + expr->unparseToString());
    }

    assert(comparisonNodeOut);
    new ControlFlowEdge(comparisonNodeOut);

    branchNodeOut = new BranchNode();

    new DataFlowEdge(comparisonNodeOut, branchNodeOut);
    new ControlFlowEdge(branchNodeOut);
}

DerefNode *AstParser::getDerefNode(SgBinaryOp *arrayIndex) {
    DerefNode *deref = nullptr;

    if (!(deref = derefTracker->getDerefNode(arrayIndex))) {
        SgExpression *lhs = arrayIndex->get_lhs_operand();

        if (SgPntrArrRefExp *prevDerefExpr = isSgPntrArrRefExp(lhs)){
            if(!graphGenerator->checkArg(PROXY_PROGRAML)){
                Node *rhs = readExpression(arrayIndex->get_rhs_operand());
                DerefNode *deref = getDerefNode(prevDerefExpr);
                new DataFlowEdge(rhs, deref);
                return deref;
            }
        } 


        // add a node for the address calculation
        deref = new DerefNode();
        derefTracker->saveDerefNode(arrayIndex, deref);



        bool resolvedParent = false;
        if (lhs->variantT() == V_SgDotExp || lhs->variantT() == V_SgArrowExp) {
            SgBinaryOp *binaryOp = isSgBinaryOp(lhs);
            assert(binaryOp);
            if(graphGenerator->checkArg(PROXY_PROGRAML)){
                DerefNode *dotNode = getDotNode(binaryOp);
                deref->memoryElement = dotNode->memoryElement;

                deref->baseTypeDependency = dotNode->baseTypeDependency;
                deref->typeDependency = dotNode;

                new ParameterLoadDataFlowEdge(dotNode, deref);
                resolvedParent = true;
            } else{
                lhs = binaryOp->get_lhs_operand();
            }
        }

        if(!resolvedParent){
            if (SgPntrArrRefExp *prevDerefExpr = isSgPntrArrRefExp(lhs)) {
                DerefNode *prevDeref = getDerefNode(prevDerefExpr);
                deref->memoryElement = prevDeref->memoryElement;
                deref->typeDependency = prevDeref;
                deref->baseTypeDependency = prevDeref->baseTypeDependency;
                new ParameterLoadDataFlowEdge(prevDeref, deref);
            
            } else if (SgVarRefExp *varRef = isSgVarRefExp(lhs)) {
                SgInitializedName *varDec = varRef->get_symbol()->get_declaration();

                Node *array = variableMapper->readVariable(varDec);
                new ParameterLoadDataFlowEdge(array, deref);

                std::cerr << "INFO: Setting memory element of deref to read memory element regardless of read/write"
                        << std::endl;
                deref->memoryElement = array;
                deref->typeDependency = array;
                deref->baseTypeDependency = array;
            }  else {
                throw std::runtime_error("array indexing indexed something unknown");
            }
        }
        // get the index value
        Node *rhs = readExpression(arrayIndex->get_rhs_operand());

        new SextDataFlowEdge(rhs, deref);
        new ControlFlowEdge(deref);
    }

    return deref;
}

DerefNode *AstParser::getDotNode(SgBinaryOp *binaryOp) {

    SgExpression *rhsOp = binaryOp->get_rhs_operand();
    SgVarRefExp *rightVarRef = isSgVarRefExp(rhsOp);

    assert(rightVarRef);

    SgInitializedName *rightVar = rightVarRef->get_symbol()->get_declaration();

    SgType *structType = binaryOp->get_lhs_operand()->get_type()->findBaseType();
    StructFieldNode *structField = variableMapper->getStructField(structType, rightVar);
    
    StructAccessNode *accessNode = new StructAccessNode();
    Node *structAddressSource = nullptr;
    Node *memoryElement = nullptr;
    if (auto lhsVarRef = isSgVarRefExp(binaryOp->get_lhs_operand())) {
        SgInitializedName *leftVar = lhsVarRef->get_symbol()->get_declaration();
        structAddressSource = variableMapper->readVariable(leftVar);
        memoryElement = structAddressSource;
        new ParameterLoadDataFlowEdge(structAddressSource, accessNode);
    } else if (auto lhsArrayDeref = isSgPntrArrRefExp(binaryOp->get_lhs_operand())) {
        structAddressSource = getDerefNode(lhsArrayDeref);
        SgVarRefExp *varRef = isSgVarRefExp(lhsArrayDeref->get_lhs_operand());
        assert(varRef);
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
        assert(varDec);
        memoryElement = Edges::graphGenerator->variableMapper->readVariable(varDec);
        new DataFlowEdge(structAddressSource, accessNode);
    }
    new ControlFlowEdge(accessNode);

    StructAccessEdge *dotEdge = new StructAccessEdge(accessNode);

    accessNode->memoryElement = memoryElement;
    accessNode->setType(structField->getImmediateType());
    accessNode->baseTypeDependency = structField;

    return accessNode;
}

Node *evaluateConstantArithmetic(SgExpression *expr, Node *lhs, Node *rhs) {
    std::string stringOp = Utils::getArithmeticNodeEncoding(expr->variantT());
    ConstantNode *lhsConstant = dynamic_cast<ConstantNode *>(lhs);
    ConstantNode *rhsConstant = dynamic_cast<ConstantNode *>(rhs);

    lhsConstant->folded = true;
    rhsConstant->folded = true;

    double a = lhsConstant->getValue();
    double b = rhsConstant->getValue();
    std::string stringResult;
    if (stringOp == "Multiplication") {
        double result = a * b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "*" << b << "=" << result << std::endl;
        stringResult = std::to_string(result);
    } else if (stringOp == "Addition") {
        double result = a + b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "+" << b << "=" << result << std::endl;
        stringResult = std::to_string(result);
    } else if (stringOp == "Subtraction") {
        double result = a - b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "-" << b << "=" << result << std::endl;
        stringResult = std::to_string(result);
    } else if (stringOp == "Division") {
        double result = a / b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "/" << b << "=" << result << std::endl;
        stringResult = std::to_string(result);
    } else if (stringOp == "LeftShift") {
        double result = int(a) << int(b);
        std::cerr << "folding:" << std::endl;
        std::cerr << int(a) << "<<" << int(b) << "=" << result << std::endl;
        stringResult = std::to_string(result);
    } else if (stringOp == "RightShift") {
        double result = int(a) >> int(b);
        std::cerr << "folding:" << std::endl;
        std::cerr << int(a) << ">>" << int(b) << "=" << result << std::endl;
        stringResult = std::to_string(result);
    } else {
        throw std::runtime_error("Found unexpected arithmetic encoding: " + stringOp);
    }

    Node *node = new ConstantNode(stringResult, lhsConstant->getType());
    return node;
}

Node *AstParser::readArithmeticExpression(SgExpression *expr) {
    if (SgBinaryOp *binaryOp = isSgBinaryOp(expr)) {
        Node *node;

        Node *lhs = readExpression(binaryOp->get_lhs_operand());
        Node *rhs = readExpression(binaryOp->get_rhs_operand());

        if (lhs->getVariant() == NodeVariant::CONSTANT && rhs->getVariant() == NodeVariant::CONSTANT) {
            return evaluateConstantArithmetic(expr, lhs, rhs);
        }

        try {
            node = processArithmeticExpression(binaryOp->variantT(), lhs, rhs);
            new ControlFlowEdge(node);
        } catch (std::runtime_error e) {
            std::cout << e.what() + binaryOp->unparseToString() << std::endl;
        }
        return node;
    } else {
        throw std::runtime_error("Arithmetic node was not a binary operation: " + expr->unparseToString());
    }
}

Node *AstParser::processArithmeticExpression(VariantT variant, Node *lhs, Node *rhs) {
    ArithmeticNode *node = new ArithmeticNode(variant);

    ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(lhs, rhs, node);
    node->arithmeticEdge = edge;

    return node;
}

Node *AstParser::addWrite(Node *variable, Node *rhs, Node *typeDependency) {
    assert(typeDependency);
    WriteNode *writeNode = new WriteNode(typeDependency);

    Edge *dataflowEdge = new ImplicitCastDataFlowEdge(rhs, writeNode);
    dataflowEdge->order = 1;
    writeNode->immediateInput = rhs;

    // and make the write an input to that node
    new WriteMemoryElementEdge(writeNode, variable);
    return writeNode;
}

Node *AstParser::writeExpression(SgNode *lhs, Node *rhs) {
    // are we writing to a variable
    if (SgVarRefExp *varRef = isSgVarRefExp(lhs)) {
        // get the declaration
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();

        // get the node that writes to this variable should connect to
        Node *variable = variableMapper->writeVariable(varDec);

        return addWrite(variable, rhs, variable);
        // if its an array index
    } else if (SgPntrArrRefExp *arrayIndex = isSgPntrArrRefExp(lhs)) {
        DerefNode *derefNode = getDerefNode(arrayIndex);

        // add a memory element edge to the write
        // element edges only show up if not proxying programl
        // get the node that writes to this variable should connect to
        Node *write = addWrite(derefNode->memoryElement, rhs, derefNode->baseTypeDependency);

        // add a resolved memory address edge between the deref and the write
        //
        // if proxying programl, normal memory address edge
        // deref -> writeData
        //
        // otherwise
        // deref -> specifyAddress -> memoryElement
        //
        // no connection to writeData
        new ResolvedMemoryAddressEdge(derefNode, write, derefNode->memoryElement);

        return write;
    } else if (SgPointerDerefExp *pointerDeref = isSgPointerDerefExp(lhs)) {
        auto varRef = isSgVarRefExp(pointerDeref->get_operand());
        assert(varRef);
        auto varDec = varRef->get_symbol()->get_declaration();
        assert(varDec);

        Node *variable = variableMapper->writeVariable(varDec);

        TypeStruct pointerTypeDesc = TypeStruct(DataType::INTEGER, 64);
        ConstantNode *fakeConstant = new ConstantNode("0", pointerTypeDesc);
        fakeConstant->folded = true;
        Node *read = new ReadNode(fakeConstant);
        new ControlFlowEdge(read);

        new MemoryAddressEdge(variable, read);

        Node *write = addWrite(read, rhs, variable);
        new MemoryAddressEdge(read, write);
        return write;
    } else if (lhs->variantT() == V_SgDotExp || lhs->variantT() == V_SgArrowExp) {
        SgBinaryOp *binaryOp = isSgBinaryOp(lhs);
        assert(binaryOp);

        if(graphGenerator->checkArg(PROXY_PROGRAML)){

            DerefNode *dotNode = getDotNode(binaryOp);

            Node *write = addWrite(dotNode->memoryElement, rhs, dotNode->typeDependency);

            new ResolvedMemoryAddressEdge(dotNode, write, dotNode->memoryElement);
            return write;
        }
        return writeExpression(binaryOp->get_lhs_operand(), rhs);
    } else if (SgInitializedName *varDec = isSgInitializedName(lhs)) {
        Node *variable = variableMapper->writeVariable(varDec);
        return addWrite(variable, rhs, variable);
    } else {
        throw std::runtime_error("Unsupported expression to write to: " + lhs->unparseToString());
    }
}

ReturnEdge *AstParser::handleFuncDec(SgFunctionDeclaration *funcDec, Node *external) {
    FunctionStartEdge *startEdge = new FunctionStartEdge(external);
    startEdge->functionName = funcDec->get_name();
    functionStartEdgeMap[funcDec] = startEdge;

    // add possible return node
    ReturnEdge *functionReturnEdge;

    if (funcDec->get_definition()) {
        SgBasicBlock *bb = funcDec->get_definition()->get_body();
        pragmaParser->parsePragmas(bb);

        graphGenerator->newBB();

        for (SgInitializedName *arg : funcDec->get_parameterList()->get_args()) {
            Node *parameter = variableMapper->createParameterVariableNode(arg);
            new ParameterInitializeEdge(parameter);
        }

        handleBB(bb->getStatementList());

        // add possible return node
        functionReturnEdge = new ReturnEdge(functionReturn, funcDec);
    } else {
        // add possible return node
        functionReturnEdge = new UndefinedFunctionEdge(funcDec->get_name(), graphGenerator->getFunctionID());
    }

    new ControlFlowEdge(external);

    return functionReturnEdge;
}

LocalScalarNode *AstParser::getIteratorFromForInit(SgStatementPtrList statements) {
    if (statements.size() > 1) {
        throw std::runtime_error("More than 1 statement in for statement init");
    }

    SgStatement *statement = statements[0];

    if (SgVariableDeclaration *varDecStatement = isSgVariableDeclaration(statement)) {
        SgInitializedNamePtrList declarations = varDecStatement->get_variables();
        if (declarations.size() > 1) {
            throw std::runtime_error("More than 1 variable declared in for statement init");
        }

        SgInitializedName *varDec = declarations[0];
        makeVariable(varDec);
        Node *variableNode = variableMapper->readVariable(varDec);

        if (LocalScalarNode *onChipVar = dynamic_cast<LocalScalarNode *>(variableNode)) {
            onChipVar->hasIteratorInit = true;
            return onChipVar;
        }

    } else if (SgExprStatement *exprStatement = isSgExprStatement(statement)) {
        SgExpression *expr = exprStatement->get_expression();
        if (SgCommaOpExp *commaExpr = isSgCommaOpExp(expr)) {
            // if comma op, check the left side of the comma for the iterator
            // why would you use a comma in a for init I don't know
            // but they do is aes256, and the iterator is on the left
            expr = commaExpr->get_lhs_operand();
        }

        if (SgAssignOp *assignOp = isSgAssignOp(expr)) {
            if (SgVarRefExp *varRef = isSgVarRefExp(assignOp->get_lhs_operand())) {
                SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
                Node *variableNode = variableMapper->readVariable(varDec);

                if (LocalScalarNode *onChipVar = dynamic_cast<LocalScalarNode *>(variableNode)) {
                    onChipVar->hasIteratorInit = true;
                    return onChipVar;
                }
            }
        }
    }

    throw std::runtime_error("Unsupported for init statement found: " + statement->unparseToString());
}

FunctionStartEdge *AstParser::getFunctionStartEdge(SgFunctionDeclaration *funcDec) {
    if (functionStartEdgeMap.count(funcDec)) {
        return functionStartEdgeMap[funcDec];
    }

    throw std::runtime_error("Tried to get the start edge of function, but it wasn't in the map: " +
                             funcDec->get_name());
}
ReturnEdge *AstParser::getFunctionReturnEdge(SgFunctionDeclaration *funcDec) {
    if (functionReturnEdgeMap.count(funcDec)) {
        return functionReturnEdgeMap[funcDec];
    }

    throw std::runtime_error("Tried to get the return edge of function, but it wasn't in the map: " +
                             funcDec->get_name());
}

void AstParser::parseAst(SgFunctionDefinition *topLevelFuncDef) {
    SgFunctionDeclaration *topLevelFuncDec = topLevelFuncDef->get_declaration();
    graphGenerator->setGroupName("External");
    graphGenerator->setFuncDec(topLevelFuncDec);
    Node *external = new ExternalNode(graphGenerator);

    graphGenerator->newBB();

    graphGenerator->setGroupName(topLevelFuncDec->get_name());

    handleFuncDec(topLevelFuncDec, external);
    variableMapper->finishedMain = true;

    while (!functionDecsNeeded.empty()) {
        SgFunctionDeclaration *funcDec = functionDecsNeeded.front();
        functionDecsNeeded.pop();

        graphGenerator->setFuncDec(funcDec);

        if (functionDecsComplete.count(funcDec)) {
            continue;
        }
        functionDecsComplete.insert(funcDec);

        graphGenerator->setGroupName(funcDec->get_name());
        graphGenerator->enterNewFunction();

        ReturnEdge *functionReturnEdge = handleFuncDec(funcDec, external);

        functionReturnEdgeMap[funcDec] = functionReturnEdge;
    }

    pragmaParser->parseInlinePragmas(functionDecsComplete);

    while (!pragmaParser->inlinedFunctions.empty()) {
        SgFunctionDeclaration *funcDec = pragmaParser->inlinedFunctions.front();
        pragmaParser->inlinedFunctions.pop();

        Node *pragma = nullptr;

        assert(decsToCalls.count(funcDec));
        for (FunctionCallNode *funcCall : decsToCalls[funcDec]) {
            if(graphGenerator->checkArg(ABSORB_PRAGMAS)){
                funcCall->inlined = true;
            } else {
                graphGenerator->stateNode = funcCall;
                if(!pragma){
                    pragma = new InlinedFunctionPragmaNode();
                }
                new InlineFunctionPragmaEdge(pragma, funcCall);
            }
        }
    }

    graphGenerator->registerCalls(topLevelFuncDec, 1);
    for(SgFunctionDeclaration *funcDec : functionDecsComplete){
        for (FunctionCallNode *funcCall : decsToCalls[funcDec]) {
            graphGenerator->registerCalls(funcDec, funcCall->unrollFactor.full);
        }
    }    
}

} // namespace GNN
