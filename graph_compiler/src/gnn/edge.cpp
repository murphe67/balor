#include "edge.h"
#include "args.h"

namespace GNN {

GraphGenerator *Edges::graphGenerator = nullptr;
Node *Edges::previousControlFlowNode = nullptr;
std::queue<Edge *> Edges::previousControlFlowNodeChangeListeners;

Node *Edges::getPreviousControlFlowNode() { return previousControlFlowNode; }

void Edges::updatePreviousControlFlowNode(Node *node) {
    previousControlFlowNode = node;
    while (!previousControlFlowNodeChangeListeners.empty()) {
        Edge *listener = previousControlFlowNodeChangeListeners.front();
        previousControlFlowNodeChangeListeners.pop();
        listener->runDeferred();
    }
}

void Edges::addPreviousControlFlowNodeChangeListener(Edge *edge) { previousControlFlowNodeChangeListeners.push(edge); }

Edge::Edge(Node *source, Node *destination) : source(source), destination(destination) {
    Edges::graphGenerator->edges.push_back(this);

    // give to unique pointer to manage memory automatically
    std::unique_ptr<Edge> edge_unique = std::unique_ptr<Edge>(this);
    // pass to vector on object so it passes out of scope at the right time
    Edges::graphGenerator->edges_unq.push_back(std::move(edge_unique));
}

WriteMemoryElementEdge::WriteMemoryElementEdge(Node *source, Node *destination) : Edge(source, destination) {
    if (destination->getVariant() == NodeVariant::LOCAL_SCALAR) {
        if (WriteNode *write = dynamic_cast<WriteNode *>(source)) {
            Node *immediateInput = write->immediateInput;

            LocalScalarNode *scalar = dynamic_cast<LocalScalarNode *>(destination);
            scalar->addBound(immediateInput);
        }
    }
}

void ControlFlowEdge::run() {
    if (Edges::graphGenerator->checkArg(ONLY_MEMORY_CONTROL_FLOW)) {
        bool memoryNode = destination->getVariant() == NodeVariant::MEMORY;
        bool branchNode = destination->getVariant() == NodeVariant::BRANCH;
        bool externalNode = destination->getVariant() == NodeVariant::EXTERNAL;
        bool returnNode = destination->getVariant() == NodeVariant::RETURN;
        bool callNode = destination->getVariant() == NodeVariant::CALL;

        if (!(memoryNode || branchNode || externalNode || returnNode || callNode)) {
            return;
        }
    }

    Edges::printSubControlFlowEdge(Edges::getPreviousControlFlowNode(), destination, false);
    Edges::updatePreviousControlFlowNode(destination);
}

void UnrollPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 0);
    }
}

void ArrayPartitionPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 1);
    }
}

void ResourceAllocationPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 2);
    }
}

void InlineFunctionPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 3);
    }
}

void DataFlowEdge::run() {
    Edges::graphGenerator->stateNode = destination;
    bool sourceIsConstant = source->getVariant() == NodeVariant::CONSTANT;
    bool sourceIsParameter = source->getVariant() == NodeVariant::ALLOCA_INITIALIZER;
    bool sourceIsGlobalArray = source->getVariant() == NodeVariant::GLOBAL_ARRAY;
    bool globalArrayIsValue = Edges::graphGenerator->checkArg(PROXY_PROGRAML);
    bool sourceIsValue = sourceIsConstant || sourceIsParameter || (sourceIsGlobalArray && globalArrayIsValue);
    bool hideValues = Edges::graphGenerator->checkArg(HIDE_VALUES);
    bool absorbTypes = Edges::graphGenerator->checkArg(ABSORB_TYPES);

    bool hasSource = !(sourceIsValue && hideValues);

    bool sourceToDest = absorbTypes && hasSource;
    bool sourceToType = (!absorbTypes) && hasSource;
    bool typeToDest = !absorbTypes;

    if (typeToDest) {
        TypeStruct sourceType = source->getImmediateType();
        std::cerr << sourceType.toString() << std::endl;

        if (!sourceType.stringOverride) {
            // source->print();
            // destination->print();
            // assert(sourceType.bitwidth > 0);
        }

        Node *typeNode = new TypeNode(sourceType, sourceIsConstant);
        typeNode->print();

        Edges::printSubDataFlowEdge(typeNode, destination, order);
        if (sourceToType) {
            Edges::printSubDataFlowEdge(source, typeNode);
        }
    } else if (sourceToDest) {
        Edges::printSubDataFlowEdge(source, destination, order);
    }
}

void ReadMemoryElementEdge::run() {
    bool allocas = !Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS);
    bool elementIsLocalScalar = source->getVariant() == NodeVariant::LOCAL_SCALAR;
    bool elementIsParameterScalar = source->getVariant() == NodeVariant::PARAMETER_SCALAR;
    bool elementIsScalar = elementIsLocalScalar || elementIsParameterScalar;

    // array elements have memory address edges from the getelemptr node
    // scalar elements have them directly from the memory element
    // which is presenting itself as an alloca
    if (allocas) {
        if (elementIsScalar) {
            (new MemoryAddressEdge(source, destination))->run();
        }
    } else {
        MemoryAddressEdge *edge = new MemoryAddressEdge(source, destination);
        edge->typeDependency = source;
        edge->run();
    }
}

void WriteMemoryElementEdge::run() {
    bool allocas = !Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS);
    bool elementIsLocalScalar = destination->getVariant() == NodeVariant::LOCAL_SCALAR;
    bool elementIsParameterScalar = destination->getVariant() == NodeVariant::PARAMETER_SCALAR;
    bool elementIsScalar = elementIsLocalScalar || elementIsParameterScalar;

    // array elements have memory address edges from the getelemptr node
    // scalar elements have them directly from the memory element
    // which is presenting itself as an alloca
    //
    // also for writes, the edge still still comes from the alloca
    // so its reversed for programl
    if (allocas) {
        if (elementIsScalar) {
            (new MemoryAddressEdge(destination, source))->run();
        }
    } else {
        MemoryAddressEdge *edge = new MemoryAddressEdge(source, destination);
        edge->typeDependency = destination;
        edge->run();
    }
}

void MemoryAddressEdge::run() {
    Edges::graphGenerator->stateNode = destination;
    bool sourceIsConstant = source->getVariant() == NodeVariant::CONSTANT;
    bool sourceIsParameter = source->getVariant() == NodeVariant::ALLOCA_INITIALIZER;
    bool sourceIsGlobalArray = source->getVariant() == NodeVariant::GLOBAL_ARRAY;
    bool globalArrayIsValue = Edges::graphGenerator->checkArg(PROXY_PROGRAML);
    bool sourceIsValue = sourceIsConstant || sourceIsParameter || (sourceIsGlobalArray && globalArrayIsValue);
    bool hideValues = Edges::graphGenerator->checkArg(HIDE_VALUES);
    bool absorbTypes = Edges::graphGenerator->checkArg(ABSORB_TYPES);

    bool hasSource = !(sourceIsValue && hideValues);

    bool sourceToDest = absorbTypes && hasSource;
    bool sourceToType = (!absorbTypes) && hasSource;
    bool typeToDest = !absorbTypes;

    if (typeToDest) {
        TypeStruct sourceType = source->getImmediateType();

        Node *typeNode;
        if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
            typeNode = new TypeNode(source->getImmediateType(), sourceIsConstant);
            typeNode->print();
        } else {
            typeNode = new TypeNode(getElemType(), sourceIsConstant);
            typeNode->print();
        }

        Edges::printSubMemoryAddressEdge(typeNode, destination);
        if (sourceToType) {
            Edges::printSubMemoryAddressEdge(source, typeNode);
        }
    } else if (sourceToDest) {
        Edges::printSubMemoryAddressEdge(source, destination);
    }
}

void SpecifyAddressEdge::run() {
    Edges::graphGenerator->stateNode = destination;
    if (!Edges::graphGenerator->checkArg(ABSORB_TYPES)) {
        Node *typeNode = new TypeNode(source->getType(), false);
        typeNode->print();

        Edges::printSubMemoryAddressEdge(source, typeNode);
        Edges::printSubMemoryAddressEdge(typeNode, destination);
    } else {
        Edges::printSubMemoryAddressEdge(source, destination);
    }
}

TypeStruct MemoryAddressEdge::getElemType() {
    if (typeDependency) {
        return typeDependency->getType();
    }
    throw std::runtime_error("Memory element edge didn't have its type dependency set");
}
void ResolvedMemoryAddressEdge::run() {
    if (!destination) {
        throw std::runtime_error("Destination must be manually set on resolved memory address edges.");
    }

    if (Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        if (dynamic_cast<DerefNode *>(source)) {
            Edges::graphGenerator->stateNode = destination;
            Node *writeNode = new SpecifyAddressNode();
            writeNode->unrollFactor = source->unrollFactor;
            writeNode->pipelined = source->pipelined;
            writeNode->print();
            (new ControlFlowEdge(writeNode))->run();
            (new DataFlowEdge(source, writeNode))->run();

            (new SpecifyAddressEdge(writeNode, memoryElement))->run();
        } else {
            throw std::runtime_error("Resolved memory address edge didn't come from a dereference node");
        }
    } else {
        (new MemoryAddressEdge(source, destination))->run();
    }
}

void ProgramlBranchEdge::run() {
    if (!Edges::graphGenerator->checkArg(REMOVE_SINGLE_TARGET_BRANCHES)) {
        Node *branch = new BranchNode();
        branch->print();

        (new ControlFlowEdge(branch))->run();
    }
}

void SextDataFlowEdge::run() {
    bool addSexts = true;
    int bitLimit = 64;
    if (Edges::graphGenerator->checkArg(REMOVE_SEXTS)) {
        addSexts = false;
    } else {

        if (Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
            bitLimit = 32;
        }

        if (widthOverridden) {
            bitLimit = newWidth;
        }

        if (source->getVariant() == NodeVariant::CONSTANT) {
            source->setType(TypeStruct(source->getSextType().dataType, bitLimit));
            addSexts = false;
        } else {
            TypeStruct type = source->getSextType();
            if (type.bitwidth >= bitLimit) {
                addSexts = false;
            }
        }
    }

    if (addSexts) {
        Edges::graphGenerator->stateNode = destination;
        Node *sext = new SextNode(bitLimit, source->getSextType().isUnsigned);
        sext->print();

        (new DataFlowEdge(source, sext))->run();
        Edge *edge = new DataFlowEdge(sext, destination);
        edge->order = 1;
        edge->run();

        (new ControlFlowEdge(sext))->run();
    } else {
        Edge *edge = new DataFlowEdge(source, destination);
        edge->order = 1;
        edge->run();
    }
}

void ParameterLoadDataFlowEdge::run() {
    // parameter loads only added for allocas
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        Edges::graphGenerator->stateNode = destination;
        bool externalArray = source->getVariant() == NodeVariant::EXTERNAL_ARRAY;
        bool parameterArray = source->getVariant() == NodeVariant::PARAMETER_ARRAY;
        std::cerr << "pldf: " << externalArray << std::endl;
        if (externalArray || parameterArray) {

            TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
            Node *read = new ReadNode(pointerType);
            read->print();

            (new ControlFlowEdge(read))->run();
            (new MemoryAddressEdge(source, read))->run();
            (new DataFlowEdge(read, destination))->run();
        } else {
            (new MemoryAddressEdge(source, destination))->run();

            if (Nodes::graphGenerator->checkArg(PROXY_PROGRAML)) {
                TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                Node *node = new ConstantNode("Local Array Stack Pointer", pointerType);
                node->print();

                (new DataFlowEdge(node, destination))->run();
            }
        }
    }
}

void ParameterInitializeEdge::run() {
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        Edges::graphGenerator->stateNode = source;

        (new ControlFlowEdge(source))->run();

        Node *previousNode = source;
        if (source->getVariant() == NodeVariant::STRUCT) {
            Node *bitcast = new BitcastNode();
            bitcast->print();
            (new ControlFlowEdge(bitcast))->run();

            (new DataFlowEdge(source, bitcast))->run();

            previousNode = bitcast;
        }

        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
        Node *store = new WriteNode(pointerType);
        store->print();

        (new ControlFlowEdge(store))->run();
        (new MemoryAddressEdge(previousNode, store))->run();

        Node *initialValueNode;
        if (source->getVariant() == NodeVariant::PARAMETER_SCALAR) {
            initialValueNode = new AllocaInitializerNode(source->getType());
        } else {
            TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
            initialValueNode = new AllocaInitializerNode(pointerType);
        }

        initialValueNode->print();

        (new DataFlowEdge(initialValueNode, store, 1))->run();

        Edges::updatePreviousControlFlowNode(store);
    }
}

void VariableDeclareEdge::run() {
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        (new ControlFlowEdge(source))->run();
    }
}

void ReturnEdge::run() {
    if (!Edges::graphGenerator->checkArg(INLINE_FUNCTIONS)) {
        Node *returnNode = getNode();
        Edges::graphGenerator->newBB();
        Edges::graphGenerator->stateNode = nullptr;

        returnNode->bbID = Edges::graphGenerator->getBBID();

        setNodeVariables(returnNode);
        returnNode->print();

        Edges::graphGenerator->newBB();

        (new ControlFlowEdge(returnNode))->run();

        for (Node *returnLocation : returnLocations) {
            Edges::printSubFunctionCallEdge(returnNode, returnLocation);
        }

        if (functionReturn) {
            SgType *returnType = funcDec->get_orig_return_type()->findBaseType();
            TypeStruct returnTypeDesc = TypeStruct(returnType->unparseToString());
            returnNode->setType(returnTypeDesc);
            (new ImplicitCastDataFlowEdge(functionReturn, returnNode))->run();
        } else {
            TypeStruct returnTypeDesc = TypeStruct();
            returnNode->setType(returnTypeDesc);
        }
    }
}

void ReturnEdge::setNodeVariables(Node *node) {
    Node *pred = Edges::getPreviousControlFlowNode();
    node->functionID = pred->functionID;
    node->groupName = pred->groupName;
}

Node *ReturnEdge::getNode() {
    if (!privateNode) {
        privateNode = new ReturnNode();
    }
    return privateNode;
}

void UndefinedFunctionEdge::setNodeVariables(Node *node) {
    node->functionID = funcID;
    node->groupName = functionName;
}

Node *UndefinedFunctionEdge::getNode() {
    if (!privateNode) {
        privateNode = new UndefinedFunctionNode(functionName);
    }
    return privateNode;
}

void LoopBackEdge::run() {
    (new BackControlFlowEdge(preLoopEdge->loopConditionStart))->run();
    Edges::updatePreviousControlFlowNode(branch);
}

// Run doesn't know where to merge to yet
// So it stores the second place to merge from in source2
// and tells the Edges class to call runDeferred after
// a control flow edge is added from source2
void MergeEndEdge::run() {
    source2 = Edges::getPreviousControlFlowNode();
    Edges::addPreviousControlFlowNodeChangeListener(this);
}

// After a control flow edge is added from source2
// we can get the current control flow node
// and add an edge from source1 to it
void MergeEndEdge::runDeferred() {
    Node *destination = Edges::getPreviousControlFlowNode();
    Node *source1 = openEdge->source1;

    // only add an extra control flow edge if something happened
    // between starting and ending the merge
    if (source1 != source2) {
        Edges::printSubControlFlowEdge(source1, destination);
    }
}

void FunctionStartEdge::run() {
    // all functions should have the external node as their predecessor
    Edges::updatePreviousControlFlowNode(source);
    Edges::addPreviousControlFlowNodeChangeListener(this);
}

void FunctionStartEdge::runDeferred() {
    Node *startNode = Edges::getPreviousControlFlowNode();
    // start at 1 as there's already an edge from external
    int edgeID = 1;
    for (Node *callLocation : callLocations) {
        Edges::printSubFunctionCallEdge(callLocation, startNode, edgeID);
        edgeID++;
    }
}

void FunctionCallEdge::run() {
    if (!Edges::graphGenerator->checkArg(INLINE_FUNCTIONS)) {
        Edges::graphGenerator->stateNode = funcCallNode;

        int parameterEdgeID = 0;
        for (SgExpression *expr : parameters) {
            SgInitializedName *originalParam = funcDec->get_parameterList()->get_args()[parameterEdgeID];
            SgType *paramType = originalParam->get_type();
            TypeStruct parameterTypeDesc = TypeStruct(paramType->findBaseType()->unparseToString());
            if (paramType->variantT() == V_SgArrayType || paramType->variantT() == V_SgPointerType) {
                parameterTypeDesc = TypeStruct(DataType::INTEGER, 64);
            }

            ConstantNode *paramTypeDependency = new ConstantNode("0", parameterTypeDesc);
            paramTypeDependency->folded = true;

            Edges::graphGenerator->nodes = std::vector<Node *>();
            Edges::graphGenerator->edges = std::vector<Edge *>();

            bool foundException = false;
            if (SgVarRefExp *varRef = isSgVarRefExp(expr)) {
                SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
                Node *variableRead = Edges::graphGenerator->variableMapper->readVariable(varDec);
                if (Edges::graphGenerator->checkArg(DROP_FUNC_CALL_PROC)) {
                    NodeVariant variant = variableRead->getVariant();
                    bool localArray = variant == NodeVariant::LOCAL_ARRAY;
                    bool externalArray = variant == NodeVariant::EXTERNAL_ARRAY;
                    bool parameterScalar = variant == NodeVariant::PARAMETER_SCALAR;
                    bool localScalar = variant == NodeVariant::LOCAL_SCALAR;
                    if (localArray || externalArray || parameterScalar || localScalar) {
                        foundException = true;
                        (new ReadMemoryElementEdge(variableRead, funcCallNode))->run();
                        (new WriteMemoryElementEdge(funcCallNode, variableRead))->run();
                    }
                } else {
                    bool isExternal = variableRead->getVariant() == NodeVariant::EXTERNAL_ARRAY;
                    bool isParameter = variableRead->getVariant() == NodeVariant::PARAMETER_ARRAY;

                    if (isExternal || isParameter) {
                        foundException = true;

                        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                        Node *readNode = new ReadNode(pointerType);
                        readNode->print();

                        (new ControlFlowEdge(readNode))->run();
                        (new MemoryAddressEdge(variableRead, readNode))->run();

                        Edge *edge = new ImplicitCastDataFlowEdge(readNode, funcCallNode, paramTypeDependency);
                        edge->order = parameterEdgeID;
                        edge->run();
                        parameterEdgeID++;

                    } else if (variableRead->getVariant() == NodeVariant::LOCAL_ARRAY) {
                        foundException = true;

                        Node *derefNode = new DerefNode();
                        derefNode->print();

                        (new ControlFlowEdge(derefNode))->run();

                        (new MemoryAddressEdge(variableRead, derefNode))->run();

                        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                        Node *valueNode = new ConstantNode("0?", pointerType);
                        valueNode->print();

                        // print first unknown edge
                        (new DataFlowEdge(valueNode, derefNode))->run();

                        // print second unknown edge
                        (new DataFlowEdge(valueNode, derefNode))->run();

                        Edge *edge = new ImplicitCastDataFlowEdge(derefNode, funcCallNode, paramTypeDependency);
                        edge->order = parameterEdgeID;
                        edge->run();
                        parameterEdgeID++;
                    }
                }
            }
            Edges::graphGenerator->derefTracker->makeNewDerefMap();
            if (!foundException) {
                Node *parameterRead = Edges::graphGenerator->astParser->readExpression(expr);
                std::vector<Node *> nodesFrozen = Edges::graphGenerator->nodes;
                for (Node *node : nodesFrozen) {
                    node->print();
                }
                std::vector<Edge *> edgesFrozen = Edges::graphGenerator->edges;
                for (Edge *edge : edgesFrozen) {
                    edge->run();
                }
                Edge *edge = new ImplicitCastDataFlowEdge(parameterRead, funcCallNode, paramTypeDependency);
                edge->order = parameterEdgeID;
                edge->run();
                parameterEdgeID++;
            }
        }

        (new ControlFlowEdge(funcCallNode))->run();

        FunctionStartEdge *startEdge = Edges::graphGenerator->astParser->getFunctionStartEdge(funcDec);
        startEdge->callLocations.push_back(funcCallNode);

        ReturnEdge *returnEdge = Edges::graphGenerator->astParser->getFunctionReturnEdge(funcDec);
        returnEdge->returnLocations.push_back(funcCallNode);
    }
}

void BackControlFlowEdge::run() {
    Edges::printSubControlFlowEdge(destination, Edges::getPreviousControlFlowNode(), true);
    Edges::updatePreviousControlFlowNode(destination);
}

void ArithmeticUnitEdge::run() {
    TypeStruct lhsType = lhs->getSextType();
    TypeStruct rhsType = rhs->getSextType();
    int unitMinBitwidth = unit->minBitwidth();

    if (lhsType.stringOverride || rhsType.stringOverride) {
        throw std::runtime_error("Overridden types cannot be used in an arithmetic unit: " + lhsType.overriddenString +
                                 " " + rhsType.overriddenString);
    }

    // signed ICMP has a minimum of i32
    // unsigned ICMP has a minimum of i8
    if(Edges::graphGenerator->checkArg(PROXY_PROGRAML)){
        if (!lhsType.isUnsigned || !rhsType.isUnsigned) {
            if (ComparisonNode *comparison = dynamic_cast<ComparisonNode *>(unit)) {
                unitMinBitwidth = 32;
            }
        }
    }


    if (lhsType.dataType == DataType::FLOAT && rhsType.dataType == DataType::FLOAT) {
        // no cast
    } else if (lhsType.dataType == DataType::INTEGER && rhsType.dataType == DataType::INTEGER) {
        // no cast
    } else if (lhsType.dataType == DataType::INTEGER && rhsType.dataType == DataType::FLOAT) {
        // cast lhs to float
        Node *node = new CastToFloatNode(rhsType);
        node->print();
        (new ControlFlowEdge(node))->run();

        (new DataFlowEdge(lhs, node))->run();
        (new DataFlowEdge(node, unit))->run();

        (new DataFlowEdge(rhs, unit, 1))->run();
        return;
    } else if (lhsType.dataType == DataType::FLOAT && rhsType.dataType == DataType::INTEGER) {
        // cast rhs to float
        Node *node = new CastToFloatNode(lhsType);
        node->print();
        (new ControlFlowEdge(node))->run();

        (new DataFlowEdge(rhs, node, 1))->run();
        (new DataFlowEdge(node, unit))->run();

        (new DataFlowEdge(lhs, unit))->run();
        return;
    }

    int maxBitwidth = lhsType.bitwidth > rhsType.bitwidth ? lhsType.bitwidth : rhsType.bitwidth;
    maxBitwidth = maxBitwidth > unitMinBitwidth ? maxBitwidth : unitMinBitwidth;

    // TODO: manage truncates
    if (lhsType.bitwidth < maxBitwidth) {
        (new SextDataFlowEdge(lhs, unit, maxBitwidth))->run();
    } else {
        (new DataFlowEdge(lhs, unit))->run();
    }

    if (rhsType.bitwidth < maxBitwidth) {
        Edge *edge = new SextDataFlowEdge(rhs, unit, maxBitwidth);
        edge->order = 1;
        edge->run();
    } else {
        (new DataFlowEdge(rhs, unit, 1))->run();
    }

    // }
}

TypeStruct ArithmeticUnitEdge::getType() {
    DataType outputType;
    TypeStruct lhsType = lhs->getSextType();
    TypeStruct rhsType = rhs->getSextType();
    int unitMinBitwidth = unit->minBitwidth();
    if (lhsType.dataType == DataType::FLOAT || rhsType.dataType == DataType::FLOAT) {
        outputType = DataType::FLOAT;
    } else {
        outputType = DataType::INTEGER;
    }

    int maxBitwidth = lhsType.bitwidth > rhsType.bitwidth ? lhsType.bitwidth : rhsType.bitwidth;
    maxBitwidth = maxBitwidth > unitMinBitwidth ? maxBitwidth : unitMinBitwidth;

    return TypeStruct(outputType, maxBitwidth);
}

ImplicitCastDataFlowEdge::ImplicitCastDataFlowEdge(Node *lhs, Node *rhs) : Edge(lhs, rhs), typeDependency(rhs) {}

ImplicitCastDataFlowEdge::ImplicitCastDataFlowEdge(Node *lhs, Node *rhs, Node *typeDependency)
    : Edge(lhs, rhs), typeDependency(typeDependency) {}

void ImplicitCastDataFlowEdge::run() {
    TypeStruct type = typeDependency->getType();
    if (type.dataType == DataType::FLOAT && source->getType().dataType == DataType::INTEGER) {
        Node *node = new CastToFloatNode(type);
        node->print();
        (new ControlFlowEdge(node))->run();

        (new DataFlowEdge(source, node))->run();
        (new DataFlowEdge(node, destination, order))->run();
        return;
    }

    if(!Edges::graphGenerator->checkArg(REMOVE_SEXTS)){
        int incomingBits = source->getType().bitwidth;
        int castBits = type.bitwidth;
        if (incomingBits > castBits) {
            Node *truncate = new TruncateNode(type);
            truncate->print();
            (new DataFlowEdge(source, truncate))->run();
            (new DataFlowEdge(truncate, destination, order))->run();
            return;
        } else if (incomingBits < castBits) {
            Edge *edge = new SextDataFlowEdge(source, destination, castBits);
            edge->order = order;
            edge->run();
            return;
        } 
    } 

    (new DataFlowEdge(source, destination, order))->run();
}

void StructAccessEdge::run() {
    if(Edges::graphGenerator->checkArg(PROXY_PROGRAML)){
        Edges::graphGenerator->stateNode = accessNode;
        TypeStruct intType = TypeStruct(DataType::INTEGER, 32);
        Node *startAddress = new ConstantNode("0", intType);
        startAddress->print();
        (new DataFlowEdge(startAddress, accessNode))->run();

        Node *index = new ConstantNode("struct index", intType);
        index->print();
        (new DataFlowEdge(index, accessNode))->run();
    }
}


} // namespace GNN