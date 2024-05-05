#include "node.h"
#include "args.h"
#include <limits>

namespace GNN {

TypeStruct::TypeStruct(const std::string &stringIn) {
    if (stringIn == "int" || stringIn == "signed int" || stringIn == "unsigned int") {
        dataType = DataType::INTEGER;
        bitwidth = 32;
    } else if (stringIn == "unsigned long" || stringIn == "unsigned long long" || stringIn == "signed long") {
        dataType = DataType::INTEGER;
        bitwidth = 64;
    } else if (stringIn == "signed char") {
        dataType = DataType::INTEGER;
        bitwidth = 8;
    } else if (stringIn == "unsigned char") {
        dataType = DataType::INTEGER;
        bitwidth = 8;
        isUnsigned = true;
    } else if (stringIn == "double") {
        dataType = DataType::FLOAT;
        bitwidth = 64;
    } else if (stringIn == "float") {
        dataType = DataType::FLOAT;
        bitwidth = 32;
    } else if (stringIn == "void") {
        isVoid = true;
    } else if (stringIn == "bool") {
        dataType = DataType::INTEGER;
        bitwidth = 8;
    } else {
        throw std::runtime_error("Cannot make TypeDescription from type: " + stringIn);
    }
}

GraphGenerator *Nodes::graphGenerator = nullptr;
int Nodes::nodeID = 0;

void Nodes::setNodeID(Node *node) {
    node->id = nodeID;
    nodeID++;
}

void Nodes::resetNodeID() { nodeID = 0; }

Node::Node() {
    std::cerr << "add node" << std::endl;

    // give to unique pointer to manage memory automatically
    std::unique_ptr<Node> node_unique = std::unique_ptr<Node>(this);
    // pass to vector on object so it passes out of scope at the right time
    Nodes::graphGenerator->nodes_unq.push_back(std::move(node_unique));

    groupName = Nodes::graphGenerator->getGroupName();
    pipelined = Nodes::graphGenerator->pragmaParser->getPipelined();
    previouslyPipelined = Nodes::graphGenerator->pragmaParser->getPreviouslyPipelined();

    if(!previouslyPipelined){
        unrollFactor = Nodes::graphGenerator->pragmaParser->getUnrollFactor();
    } else {
        PragmaParser *pragmaParser = Nodes::graphGenerator->pragmaParser.get();
        unrollFactor = pragmaParser->getTripcount();
        unrollFactor.full /= pragmaParser->getPipelineTripcount();
        unrollFactor.first /= pragmaParser->getPipelineTripcount();
        if(unrollFactor.second > 1){
            unrollFactor.second /= pragmaParser->getPipelineTripcount();
        }

        if(unrollFactor.third > 1){
            unrollFactor.third /= pragmaParser->getPipelineTripcount();
        }
    }


    tripcount = Nodes::graphGenerator->pragmaParser->getTripcount();

    funcDec = Nodes::graphGenerator->getFuncDec();

    bbID = Nodes::graphGenerator->getBBID();
    functionID = Nodes::graphGenerator->getFunctionID();

    // Add to raw pointer vector for actually use
    Nodes::graphGenerator->nodes.push_back(this);

    std::cerr << "end add node" << std::endl;
}

std::string Node::getTypeToPrint() {
    try {
        TypeStruct type = getOutputType();
        if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
            if(type.dataType == DataType::FLOAT){
                return "double";
            }
        }

        return type.toString();
    } catch (std::runtime_error &e) {
        return "NA";
    }
}

void addDataInNodeToAlloca(Node *alloca) {
    if (!Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        Nodes::graphGenerator->stateNode = alloca;
        TypeStruct inputType = TypeStruct(DataType::INTEGER, 32);
        Node *dataIn = new ConstantNode("0", inputType);
        dataIn->print();
        new DataFlowEdge(dataIn, alloca);
    }
}

void ConstantNode::print() {
    if (!Nodes::graphGenerator->checkArg(HIDE_VALUES)) {
        if (!folded) {
            Nodes::setNodeID(this);

            NodePrinter printer(this, "lightyellow");
            printer.attributes["nodeType"] = "DO_NOT_USE";
            printer.attributes["keyText"] = "constantValue";
            printer.attributes["label"] = value;

            printer.attributes.erase("datatype");
            printer.attributes.erase("bitwidth");

            printer.print();
        }
    }
}

void GlobalArrayNode::print() {
    if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
        constNode->print();
    } else {
        Nodes::setNodeID(this);

        NodePrinter printer(this, "0.33 0.1 1");


        printer.attributes["nodeType"] = "instruction";
        printer.attributes["keyText"] = "globalArray";
        printer.attributes["label"] = "Global Array: " + value;

        printer.print();
    }
}

TypeStruct GlobalArrayNode::getType(){
    return constNode->getType();
}

TypeStruct ConstantNode::getSextType(){
    if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
        return getType();
    } 
    if(type.dataType == DataType::FLOAT){
        return getType();
    }
    return TypeStruct(type.dataType, 0);
}

void AllocaInitializerNode::print() {
    if (!Nodes::graphGenerator->checkArg(HIDE_VALUES)) {
        Nodes::setNodeID(this);

        NodePrinter printer(this, "lightyellow");
        printer.attributes["nodeType"] = "DO_NOT_USE";
        printer.attributes["keyText"] = "parameterValue";
        printer.attributes["label"] = "Parameter";

        printer.attributes.erase("datatype");
        printer.attributes.erase("bitwidth");

        printer.print();
    }
}

void BranchNode::print() {
    Nodes::setNodeID(this);
    NodePrinter nodePrinter(this, "0.75 0.1 1");

    std::string label = "Branch";
    nodePrinter.attributes["label"] = label;
    nodePrinter.attributes["keyText"] = "br";

    nodePrinter.print();
}

void ReadNode::print() {
    Nodes::setNodeID(this);

    std::string description = "Load";
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        description = "Read";
    }

    NodePrinter printer(this, "0.584 0.1 1");
    printer.attributes["label"] = description;
    printer.attributes["keyText"] = "load";

    printer.print();
}

void WriteNode::print() {
    Nodes::setNodeID(this);
    std::string description = "Store";
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        description = "Write";
    }

    NodePrinter printer(this, "0.584 0.1 1");
    printer.attributes["label"] = description;
    printer.attributes["keyText"] = "store";

    printer.print();
}

void ComparisonNode::print() {
    Nodes::setNodeID(this);
    std::string description = "Comparison";

    NodePrinter printer(this, "0 0.1 1");
    printer.attributes["label"] = description;
    if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
        if (getType().dataType == DataType::INTEGER) {
            printer.attributes["keyText"] = "icmp";
        } else {
            printer.attributes["keyText"] = "fcmp";
        }
    } else {
        printer.attributes["keyText"] = "cmp";
    }


    printer.print();
}

TypeStruct ComparisonNode::getType() {
    assert(edge);
    return edge->getType();
}

void SextNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "0.083 0.1 1");
    if (isUnsigned) {
        printer.attributes["label"] = "Zext";
        printer.attributes["keyText"] = "zext";
    } else {
        printer.attributes["label"] = "Sext";
        printer.attributes["keyText"] = "sext";
    }

    printer.print();
}

TypeStruct SextNode::getType() { return TypeStruct(DataType::INTEGER, bitwidth); }

void ArithmeticNode::print() {
    Nodes::setNodeID(this);
    std::string description = opType;

    NodePrinter printer(this, "0 0.1 1");
    std::string keytext;
    if (opType == "Addition") {
        keytext = "add";
    } else if (opType == "Division") {
        keytext = "div";
    } else if (opType == "Multiplication") {
        keytext = "mul";
    } else if (opType == "Subtraction") {
        keytext = "sub";
    } else if (opType == "RightShift") {
        keytext = "ashr";
    } else if (opType == "LeftShift") {
        keytext = "shl";
    } else if (opType == "BitAnd") {
        keytext = "and";
    } else if (opType == "BitXor") {
        keytext = "xor";
    } else if (opType == "Or") {
        keytext = "or";
    } else if (opType == "Xor") {
        keytext = "xor";
    } else {
        keytext = opType;
    }

    if (Nodes::graphGenerator->checkArg(PROXY_PROGRAML)) {
        if (getType().dataType == DataType::FLOAT) {
            keytext = "f" + keytext;
        }
    }

    printer.attributes["keyText"] = keytext;
    printer.attributes["label"] = description;

    printer.print();
}

int ArithmeticNode::minBitwidth() {
    if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
        if (opType == "Xor" || opType == "BitXor") {
            return 32;
        }
    }

    return 0;
}

TypeStruct ArithmeticNode::getType() {
    assert(arithmeticEdge);
    return arithmeticEdge->getType();
}

void DerefNode::print() {
    Nodes::setNodeID(this);
    std::string description = "Get Element Ptr";
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        description = "Get Address";
    }

    NodePrinter printer(this, "0.833 0.05 1");
    printer.attributes["keyText"] = "getelementptr";
    printer.attributes["label"] = description;

    printer.print();
}

void DerefNode::setType(TypeStruct typeDesc) {
    isTypeSet = true;
    type = typeDesc;
}

TypeStruct DerefNode::getType() {
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        return TypeStruct(DataType::INTEGER, 32);
    } else {
        if (isTypeSet) {
            return type;
        }
        if (!typeDependency) {
            return TypeStruct(DataType::INTEGER, 64);
        }
        TypeStruct prevType = typeDependency->getImmediateType();
        if (!prevType.stringOverride) {
            return TypeStruct(DataType::INTEGER, 64);
        } else {
            if (prevType.overriddenString.back() == ']') {
                int backIndex = 2;
                int length = prevType.overriddenString.length();
                while (prevType.overriddenString[length - backIndex] != '[') {
                    backIndex++;
                }

                if (prevType.overriddenString[length - backIndex - 1] == ']') {
                    TypeStruct subArrayType = TypeStruct();
                    subArrayType.overrideType(prevType.overriddenString.substr(0, length - backIndex));
                    return subArrayType;
                }
                return TypeStruct(DataType::INTEGER, 64);
            } else {
                throw std::runtime_error("Overridden array type didn't end in ]: " + prevType.overriddenString);
            }
        }
    }
}

void LocalScalarNode::print() {
    Nodes::setNodeID(this);
    addDataInNodeToAlloca(this);
    if(type.dataType == DataType::INTEGER){
        if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
            if (Edges::graphGenerator->checkArg(REDUCE_ITERATOR_BITWIDTH)) {
                double boundsMin = std::numeric_limits<double>::max();
                double boundsMax = std::numeric_limits<double>::lowest();
                for(double bound : bounds){
                    boundsMax = bound > boundsMax ? bound : boundsMax;
                    boundsMin = bound < boundsMin ? bound : boundsMin;
                }

                if (fixedSizeIterator && hasIteratorInit && writtenToAsIterator) {
                    int range = boundsMax - boundsMin;
                    int bitwidth = ceil(log2(range + 1));

                    type = TypeStruct(DataType::INTEGER, bitwidth);
                }
            }
        }
    }

    NodePrinter printer(this, "0.33 0.1 1");
    
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        printer.attributes["keyText"] = "localScalar";
        std::string label = "Local Scalar: " + description;

        printer.attributes["label"] = label;
    } else {
        printer.attributes["keyText"] = "alloca";
        printer.attributes["label"] = "Alloca: " + description;
    }

    printer.print();
}

TypeStruct LocalScalarNode::getImmediateType() {
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        return getType();
    }
    return TypeStruct(DataType::INTEGER, 64);
}

TypeStruct ParameterScalarNode::getImmediateType() {
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        return getType();
    }
    return TypeStruct(DataType::INTEGER, 64);
}

void PragmaNode::print() {
    if (!Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Nodes::setNodeID(this);
        NodePrinter printer(this, "white");

        printer.attributes["nodeType"] = "pragma";
        std::string label = "Pragma: " + keyText;
        if (factor != "0") {
            label += "\n" + factor;
        }
        printer.attributes["label"] = label;
        printer.attributes["keyText"] = keyText;
        printer.attributes["numeric"] = factor;

        printer.print();
    }
}

void ExternalNode::print() {
    bool drawExternal = true;
    bool control_flow = !Nodes::graphGenerator->checkArg(IGNORE_CONTROL_FLOW);
    bool call_edges = !Nodes::graphGenerator->checkArg(IGNORE_CALL_EDGES);
    bool inlined_functions = Nodes::graphGenerator->checkArg(INLINE_FUNCTIONS);

    // no control flow also means no call edges
    if (!control_flow) {
        drawExternal = false;
    }

    // if non-inline and non-connected, no external
    if (!call_edges && !inlined_functions) {
        drawExternal = false;
    }

    if (drawExternal) {
        Nodes::setNodeID(this);

        NodePrinter printer(this, "white");

        printer.attributes["keyText"] = "[external]";
        printer.attributes["label"] = "External";

        printer.print();

        // push the external node closer to the top of the graph
        std::cout << "subgraph cluster_External {" << std::endl;
        std::cout << "{rank=min; node" + std::to_string(id) + "}" << std::endl;
        std::cout << "}" << std::endl;
    }
}

void TypeNode::print() {
    Nodes::setNodeID(this);
    std::string color = "azure2";
    std::string nodeType = "variable";
    if (constant) {
        color = "white";
        nodeType = "constant";
    }

    NodePrinter printer(this, color);

    printer.attributes["label"] = getTypeToPrint();
    printer.attributes["nodeType"] = nodeType;
    printer.attributes["shape"] = "diamond";

    printer.attributes["keyText"] = getTypeToPrint();

    // by default it will be set to the unroll factor
    // of the BB
    // which doesn't make sense for constants
    if (Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        printer.attributes["numeric"] = "1";
    }

    printer.print();
}

void ReturnNode::print() {
    if (!Nodes::graphGenerator->checkArg(INLINE_FUNCTIONS)) {
        Nodes::setNodeID(this);

        NodePrinter printer(this, "white");
        printer.attributes["label"] = "Return";
        printer.attributes["keyText"] = "ret";

        printer.print();
    }
}

void SubParameterArrayNode::print() {
    Nodes::setNodeID(this);

    addDataInNodeToAlloca(this);

    NodePrinter printer(this, "0.33 0.1 1");
    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        printer.attributes["label"] = "Parameter: " + variableName;
        printer.attributes["keyText"] = "arrayParameter";
    } else {
        printer.attributes["label"] = "Alloca: " + variableName;
        printer.attributes["keyText"] = "alloca";
    }

    if(!Nodes::graphGenerator->checkArg(ONE_HOT_TYPES)){
        assert(Nodes::graphGenerator->checkArg(ABSORB_TYPES));

        // this node can be a pointer, which would have 1 element
        // so only change if array
        printer.attributes["arrayWidth"] = std::to_string(numElements);
    }

    printer.print();
}

void ParameterScalarNode::print() {
    Nodes::setNodeID(this);
    addDataInNodeToAlloca(this);
    std::string description = "Alloca: ";
    std::string keyText = "alloca";

    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        description = "External Scalar: ";
        keyText = "externalScalar";
    }

    NodePrinter printer(this, "0.33 0.1 1");
    printer.attributes["label"] = description + variableName;

    printer.attributes["keyText"] = keyText;
    printer.print();
}

void LocalArrayNode::print() {
    Nodes::setNodeID(this);

    addDataInNodeToAlloca(this);

    NodePrinter printer(this, "0.33 0.1 1");

    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        printer.attributes["label"] = "Local Array: " + variableName;
        printer.attributes["keyText"] = "localArray";
    } else {
        printer.attributes["label"] = "Alloca: " + variableName;
        printer.attributes["keyText"] = "alloca";
    }

    if(!Nodes::graphGenerator->checkArg(ONE_HOT_TYPES)){
        assert(Nodes::graphGenerator->checkArg(ABSORB_TYPES));
        std::string dataType = "float";
        std::string typeDesc = getType().toString();
        if(typeDesc == "i1"){
            dataType = "bool";
        } else if(getType().dataType == DataType::INTEGER){
            dataType = "int";
        }
        printer.attributes["datatype"] = dataType;

        std::string bitwidth = typeDesc;
        bitwidth.erase(0, 1);
        printer.attributes["bitwidth"] = bitwidth;

        printer.attributes["arrayWidth"] = std::to_string(numElements);
    }


    printer.print();
}

void ExternalArrayNode::print() {
    Nodes::setNodeID(this);

    addDataInNodeToAlloca(this);

    NodePrinter printer(this, "0.33 0.1 1");

    if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        printer.attributes["label"] = "External Array: " + variableName;
        printer.attributes["keyText"] = "externalArray";
    } else {
        printer.attributes["label"] = "Alloca: " + variableName;
        printer.attributes["keyText"] = "alloca";
    }


    if(!Nodes::graphGenerator->checkArg(ONE_HOT_TYPES)){
        assert(Nodes::graphGenerator->checkArg(ABSORB_TYPES));


        printer.attributes["arrayWidth"] = std::to_string(numElements);

    }

    printer.print();
}

void FunctionCallNode::print() {
    if (!Nodes::graphGenerator->checkArg(INLINE_FUNCTIONS)) {
        Nodes::setNodeID(this);
        NodePrinter printer(this, "white");
        printer.attributes["keyText"] = "call";
        printer.attributes["label"] = "Function Call";

        if(inlined){
            printer.attributes["inlined"] = "inlined";
        }

        printer.print();
    }
}

void SpecifyAddressNode::print() {
    Nodes::setNodeID(this);
    std::string description = "Specify Address To Read/Write";

    NodePrinter printer(this, "0.584 0.1 1");
    printer.attributes["label"] = description;
    printer.attributes["keyText"] = "specifyAddress";

    printer.print();
}

void StructNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "0.33 0.1 1");
    printer.attributes["label"] = "Alloca";
    printer.attributes["keyText"] = "alloca";

    printer.print();
}

void BitcastNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "white");
    printer.attributes["label"] = "Bitcast";
    printer.attributes["keyText"] = "bitcast";

    printer.print();
}

void BreakNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "white");
    printer.attributes["label"] = "Break";
    printer.attributes["keyText"] = "break";

    printer.print();
}

void CastNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "white");
    printer.attributes["label"] = "Cast";
    printer.attributes["keyText"] = "cast";

    printer.print();
}

void UnaryOpNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "0 0.1 1");
    printer.attributes["label"] = opType;
    printer.attributes["keyText"] = opType;

    printer.print();
}

void TruncateNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "white");
    printer.attributes["label"] = "Truncate";
    printer.attributes["keyText"] = "trunc";

    printer.print();
}

void UndefinedFunctionNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "white");
    printer.attributes["label"] = "Undefined Function: " + name;
    if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
        printer.attributes["keyText"] = "; undefined function";
    } else {
        printer.attributes["keyText"] = name;
    }

    printer.print();
}

void CastToFloatNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "white");
    printer.attributes["label"] = "Cast To Float";
    printer.attributes["keyText"] = "sitofp";

    printer.print();
}

void AddressOfNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "0 0.1 1");
    printer.attributes["label"] = "Get Address";
    printer.attributes["keyText"] = "getAddress";

    printer.print();
}

void SelectNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "0 0.1 1");
    printer.attributes["label"] = "select";
    printer.attributes["keyText"] = "phi";

    printer.print();
}

TypeStruct SelectNode::getType() {
    assert(edge);
    return edge->getType();
}

void FNegNode::print() {
    Nodes::setNodeID(this);

    NodePrinter printer(this, "0 0.1 1");
    printer.attributes["label"] = "Negate";
    printer.attributes["keyText"] = "fneg";

    printer.print();
}

void LocalScalarNode::addBound(Node *bound) {
    // std::cerr << "adding bound" << std::endl;
    // bool constant = bound->getVariant() == NodeVariant::CONSTANT;
    // std::cerr << "constant: " << constant << std::endl;
    // std::cerr << "in bound region: " << inIteratorBoundsRegion << std::endl;
    // std::cerr << "in increment region: " << inIncrementRegion << std::endl;
    if (inIteratorBoundsRegion) {
        if (ConstantNode *constant = dynamic_cast<ConstantNode *>(bound)) {
            double value = constant->getValue();
            bounds.push_back(value);
        } else {
            fixedSizeIterator = false;
        }
    } else if (!inIncrementRegion) {
        writtenToAsIterator = false;
    }
}

void LocalScalarNode::processComparison(Node *comparison) {
    inIteratorBoundsRegion = true;
    if (ComparisonNode *comparisonNode = dynamic_cast<ComparisonNode *>(comparison)) {
        if (comparisonNode->binaryComp) {
            bool simpleComparison = false;
            Node *bound;
            if (comparisonNode->lhs = this) {
                simpleComparison = true;
                bound = comparisonNode->rhs;
            } else if (comparisonNode->rhs = this) {
                simpleComparison = true;
                bound = comparisonNode->lhs;
            }

            if (simpleComparison) {
                addBound(bound);
                inIteratorBoundsRegion = false;
                return;
            }
        }
    }
    fixedSizeIterator = false;
}

void StructAccessNode::print() {
    Nodes::setNodeID(this);

    std::string description = "Struct Access";
    std::string keyText = "struct_access";
    if(Nodes::graphGenerator->checkArg(PROXY_PROGRAML)){
        description = "Get Element Ptr";
        if (Nodes::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
            description = "Get Address";
        }
        keyText = "getelementptr";
    }

    NodePrinter printer(this, "0.833 0.05 1");
    printer.attributes["keyText"] = keyText;
    printer.attributes["label"] = description;

    printer.print();
}


} // namespace GNN