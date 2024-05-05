#ifndef GNN_NODE_H
#define GNN_NODE_H

#include "edge.h"
#include "graphGenerator.h"
#include "nodePrinter.h"
#include "nodeUtils.h"
#include "rose.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// std::string typeConversion(const std::string type) {
//     if (type == "i8" || type == "i32" || type == "i64" || type == "f32" || type == "f64" || type == "i1") {
//         return type;
//     }
//     if (type == "int" || type == "signed int" || type == "unsigned int") {
//         return "i32";
//     } else if (type == "unsigned long" || type == "unsigned long long") {
//         return "i64";
//     } else if (type == "signed char" || type == "unsigned char") {
//         return "i8";
//     } else if (type == "double") {
//         return "f64";
//     } else {
//         std::cerr << "Unreformatted type: " << type << std::endl;
//         return type;
//     }
// }

// std::string typeConversion(TypeStruct type) {
//     if (type == V_SgTypeSignedInt) {
//         return "i32";
//     } else if (type == V_SgTypeSignedLong) {
//         return "i64";
//     } else if (type == V_SgTypeSignedChar || type == V_SgTypeUnsignedChar) {
//         return "i8";
//     } else if (type == V_SgTypeDouble) {
//         return "f64";
//     } else if (type == V_SgTypeVoid){
//         return "void";
//     } else {
//         throw std::runtime_error("Unreformatted type: " + type);
//     }
// }
} // namespace

namespace GNN {

class GraphGenerator;
class Edge;
class ControlFlowEdge;
class DerefTracker;
class Node;
class ArithmeticUnitEdge;

enum class NodeVariant {
    DEFAULT,
    EXTERNAL,
    LOCAL_ARRAY,
    EXTERNAL_ARRAY,
    PARAMETER_ARRAY,
    LOCAL_SCALAR,
    PARAMETER_SCALAR,
    CONSTANT,
    ALLOCA_INITIALIZER,
    MEMORY,
    BRANCH,
    RETURN,
    CALL,
    STRUCT,
    COMPARISON,
    ARITHMETIC,
    GLOBAL_ARRAY
};

enum class DataType { INTEGER, FLOAT };

class TypeStruct {
  public:
    TypeStruct(DataType dataType, int bitwidth) : dataType(dataType), bitwidth(bitwidth) {}
    TypeStruct() : isVoid(true) {}
    TypeStruct(const std::string &stringIn);

    void overrideType(const std::string &stringIn) {
        isVoid = false;
        stringOverride = true;
        overriddenString = stringIn;
    }

    DataType dataType;
    int bitwidth = 0;
    bool isVoid = false;
    bool isUnsigned = false;
    bool stringOverride = false;
    std::string overriddenString;

    std::string toString() {
        if(isVoid){
          return "void";
        }
        if (stringOverride) {
            return overriddenString;
        }
        if (dataType == DataType::INTEGER) {
            return "i" + std::to_string(bitwidth);
        } else {
            return "f" + std::to_string(bitwidth);
        }
    }
};

struct StackedFactor {
  float full;
  float first;
  float second;
  float third;
};

//-------------------------------------------
//       Base Types to Inherit From
//-------------------------------------------

class Nodes {
  public:
    static GraphGenerator *graphGenerator;
    static void setNodeID(Node *node);
    static void resetNodeID();

  private:
    static int nodeID;
};

class Node {
  public:
    virtual void print() = 0;
    virtual TypeStruct getType() { throw std::runtime_error("Type was pulled from node without type"); }

    virtual std::string getTypeToPrint();
    virtual TypeStruct getImmediateType() { return getType(); }
    virtual TypeStruct getSextType() { return getImmediateType(); }
    virtual TypeStruct getOutputType() { return getImmediateType(); }

    StackedFactor unrollFactor;

    virtual void setType(TypeStruct type) { this->type = type; }

    std::string groupName;

    int id;
    int partitionFactor1 = 0;
    int partitionFactor2 = 0;
    StackedFactor tripcount;
    bool pipelined;
    bool previouslyPipelined;

    int bbID;
    int functionID;

    std::string extraNote = "";

    virtual int minBitwidth() { return 0; }

    SgFunctionDeclaration *funcDec;

    std::string partitionType1 = "none";
    std::string partitionType2 = "none";
    bool inlined = false;


    virtual NodeVariant getVariant() { return NodeVariant::DEFAULT; }

  protected:
    Node();
    TypeStruct type;
};

//----------------------------------------
//   Basic Nodes for all Graph Types
//----------------------------------------

class ConstantNode : public Node {
  public:
    ConstantNode(const std::string &value, TypeStruct type) : value(value) { setType(type); }
    std::string value;

    bool folded = false;

    bool canGetValue() {
        try {
            getValue();
            return true;
        } catch (...) {
            return false;
        }
    }
    double getValue() {
        try {
            return std::stod(value);
        } catch (...) {
            throw std::runtime_error("Tried to get int from non-numerical constant");
        }
    }

    void print() override;
    NodeVariant getVariant() override { return NodeVariant::CONSTANT; }

    TypeStruct getType() override { return type; }
    TypeStruct getSextType() override;
};

class AllocaInitializerNode : public Node {
  public:
    AllocaInitializerNode(TypeStruct type) { setType(type); }
    void print() override;

    NodeVariant getVariant() override { return NodeVariant::ALLOCA_INITIALIZER; }
    TypeStruct getType() override { return type; }
};

class BranchNode : public Node {
  public:
    void print() override;
    NodeVariant getVariant() override { return NodeVariant::BRANCH; }
};
class ReadNode : public Node {
  public:
    ReadNode(Node *typeDependency) : typeDependency(typeDependency) {}
    ReadNode(TypeStruct type) { setType(type); }
    void print() override;

    Node *typeDependency = nullptr;

    NodeVariant getVariant() override { return NodeVariant::MEMORY; }

    TypeStruct getType() override {
        if (typeDependency) {
            return typeDependency->getType();
        }
        return type;
    }
};

class WriteNode : public Node {
  public:
    WriteNode(Node *typeDependency) : typeDependency(typeDependency) {}
    WriteNode(TypeStruct type) { setType(type); }
    void print() override;

    Node *immediateInput = nullptr;
    Node *typeDependency = nullptr;

    NodeVariant getVariant() override { return NodeVariant::MEMORY; }

    TypeStruct getType() override {
        if (typeDependency) {
            return typeDependency->getType();
        }
        return type;
    }
};

class DerefNode : public Node {
  public:
    DerefNode() {}
    void print() override;

    Node *memoryElement = nullptr;
    Node *typeDependency = nullptr;
    Node *baseTypeDependency = nullptr;

    void setType(TypeStruct typeDesc) override;

    TypeStruct getType();

  private:
    bool isTypeSet = false;
};

class ComparisonNode : public Node {
  public:
    void print() override;

    ArithmeticUnitEdge *edge;

    bool binaryComp = false;
    Node *lhs = nullptr;
    Node *rhs = nullptr;

    TypeStruct getImmediateType() override { return TypeStruct(DataType::INTEGER, 1); }
    TypeStruct getType();
    TypeStruct getOutputType() override { return getType(); }

    // int minBitwidth() { return 32; }

    NodeVariant getVariant() override { return NodeVariant::COMPARISON; }
};

class ArithmeticNode : public Node {
  public:
    ArithmeticNode(VariantT variant) : opType(Utils::getArithmeticNodeEncoding(variant)) {}

    std::string opType;
    void print() override;

    ArithmeticUnitEdge *arithmeticEdge = nullptr;

    TypeStruct getType() override;
    int minBitwidth() override;
    NodeVariant getVariant() override { return NodeVariant::ARITHMETIC; }
};

class ExternalNode : public Node {
  public:
    ExternalNode(GraphGenerator *graphGenerator) {}
    void print() override;

    virtual NodeVariant getVariant() { return NodeVariant::EXTERNAL; }
};

//-------------------------------------------------
//     Variable Nodes
//-------------------------------------------------

class LocalArrayNode : public Node {
  public:
    LocalArrayNode(const std::string &variableName, TypeStruct arrayType)
        : variableName(variableName), arrayType(arrayType) {}
    std::string variableName;
    TypeStruct arrayType;

    NodeVariant getVariant() override { return NodeVariant::LOCAL_ARRAY; }

    TypeStruct getImmediateType() override { return arrayType; }
    TypeStruct getType() override { return type; }
    TypeStruct getSextType() override { return TypeStruct(DataType::INTEGER, 64); }

    int numElements = 1;

    void print() override;
};

class LocalScalarNode : public Node {
  public:
    LocalScalarNode(const std::string &description) : description(description) {}
    std::string description;

    void print() override;

    // are both bounds constants
    bool fixedSizeIterator = true;

    // is the variable inititialized in a for init
    bool hasIteratorInit = false;

    // is the variable only written in for init and for increment
    bool writtenToAsIterator = true;

    // are we in a code region where the iterator can be bounded
    // e.g. init or condition
    bool inIteratorBoundsRegion = false;

    bool inIncrementRegion = false;

    void addBound(Node *bound);

    std::vector<double> bounds;

    void processComparison(Node *comparison);

    TypeStruct getImmediateType() override;

    TypeStruct getType() override { return type; }

    NodeVariant getVariant() override { return NodeVariant::LOCAL_SCALAR; }
};

class ParameterScalarNode : public Node {
  public:
    ParameterScalarNode(const std::string &variableName) : variableName(variableName) {}
    std::string variableName;

    NodeVariant getVariant() override { return NodeVariant::PARAMETER_SCALAR; }

    TypeStruct getImmediateType() override;

    void print() override;
    TypeStruct getType() override { return type; }
};

class ExternalArrayNode : public Node {
  public:
    ExternalArrayNode(const std::string &variableName) : variableName(variableName) {}
    std::string variableName;

    NodeVariant getVariant() override { return NodeVariant::EXTERNAL_ARRAY; }

    void print() override;
    TypeStruct getType() override { return type; }
    TypeStruct getImmediateType() override { return TypeStruct(DataType::INTEGER, 64); }

    int numElements;

};

class SubParameterArrayNode : public Node {
  public:
    SubParameterArrayNode(const std::string &variableName) : variableName(variableName) {}
    std::string variableName;

    virtual NodeVariant getVariant() { return NodeVariant::PARAMETER_ARRAY; }

    void print() override;
    TypeStruct getType() override { return type; }
    TypeStruct getImmediateType() override { return TypeStruct(DataType::INTEGER, 64); }

    int numElements;
};

//-----------------------------------------------
//          Programl Proxy Nodes
//-----------------------------------------------

class PragmaNode : public Node {
  public:
    PragmaNode() {}
    PragmaNode(int factor) : factor(std::to_string(factor)) {}

    std::string factor = "0";
    std::string keyText;

    void print() override;
};

class UnrollPragmaNode : public PragmaNode {
  public:
    UnrollPragmaNode(int factor) : PragmaNode(factor) { keyText = "unroll"; }
};

class ArrayPartitionPragmaNode : public PragmaNode {
  public:
    ArrayPartitionPragmaNode(std::string partitionStyle, int factor, int dim) : PragmaNode(factor) {
        keyText = partitionStyle + "ArrayPartition" + std::to_string(dim);
    }
};

class Bram1P_ResourceAllocationPragmaNode : public PragmaNode {
  public:
    Bram1P_ResourceAllocationPragmaNode() { keyText = "resourceAllocation_bram1p"; }
};

class Bram2P_ResourceAllocationPragmaNode : public PragmaNode {
  public:
    Bram2P_ResourceAllocationPragmaNode() { keyText = "resourceAllocation_bram2p"; }
};

class InlinedFunctionPragmaNode : public PragmaNode {
  public:
    InlinedFunctionPragmaNode() { keyText = "inlinedFunction"; }
};

class SextNode : public Node {
  public:
    SextNode(int bitwidth, bool isUnsigned) : bitwidth(bitwidth), isUnsigned(isUnsigned) {}

    int bitwidth;
    bool isUnsigned;

    void print() override;

    TypeStruct getType() override;
};

class TypeNode : public Node {
  public:
    TypeNode(TypeStruct type, bool constant) : constant(constant) { setType(type); }

    bool constant;

    void print() override;
    TypeStruct getType() { return type; }
};
class ReturnNode : public Node {
  public:
    void print() override;
    NodeVariant getVariant() override { return NodeVariant::RETURN; }
    TypeStruct getType() override { return type; }
};

class FunctionCallNode : public Node {
  public:
    void print() override;
    TypeStruct getType() override { return type; }
    NodeVariant getVariant() override { return NodeVariant::CALL; }
};

class SpecifyAddressNode : public Node {
  public:
    void print() override;
    TypeStruct getType() override { return TypeStruct(DataType::INTEGER, 32); }

    NodeVariant getVariant() override { return NodeVariant::MEMORY; }
};

class StructNode : public Node {
  public:
    void print() override;
    // WARNING: not sure if this getType is generalized, it doesn't look it
    // not sure if its ever called?
    TypeStruct getType() override { assert(false); }
    TypeStruct getImmediateType() override {
        TypeStruct structType = TypeStruct();
        structType.overrideType("struct*");
        return structType;
    }
    TypeStruct getSextType() override { return TypeStruct(DataType::INTEGER, 64); }

    NodeVariant getVariant() override { return NodeVariant::STRUCT; }
};

class BitcastNode : public Node {
  public:
    void print() override;
    TypeStruct getType() override { return TypeStruct(DataType::INTEGER, 64); }
};

class StructFieldNode : public Node {
  public:
    StructFieldNode(int index) : index(index) {}
    int index;
    void print() override {}
    TypeStruct getType() override { return type; }
};

class StructArrayFieldNode : public StructFieldNode {
  public:
    StructArrayFieldNode(int index, TypeStruct arrayType) : arrayType(arrayType), StructFieldNode(index) {}
    TypeStruct arrayType;

    TypeStruct getImmediateType() override { return arrayType; }
};

class BreakNode : public Node {
  public:
    void print() override;
};

class CastNode : public Node {
  public:
    void print() override;
    TypeStruct getType() override { return type; }
};

class UnaryOpNode : public Node {
  public:
    UnaryOpNode(const std::string &opType) : opType(opType) {}
    std::string opType;
    void print() override;
    TypeStruct getType() override { return type; }
};

class TruncateNode : public Node {
  public:
    TruncateNode(TypeStruct outputType) { setType(outputType); }
    TypeStruct getType() { return type; }

    void print() override;
};

class UndefinedFunctionNode : public Node {
  public:
    UndefinedFunctionNode(const std::string &name) : name(name) {}
    std::string name;

    void print() override;
};

class CastToFloatNode : public Node {
  public:
    CastToFloatNode(TypeStruct type) { setType(type); }
    TypeStruct getType() override { return type; }

    void print() override;
};

class AddressOfNode : public Node {
  public:
    void print() override;
    TypeStruct getType() override { return TypeStruct(DataType::INTEGER, 64); }
};

class SelectNode : public Node {
  public:
    void print() override;
    ArithmeticUnitEdge *edge;
    TypeStruct getType() override;
};

class GlobalArrayNode : public Node {
  public:
    GlobalArrayNode(const std::string &value, TypeStruct elementType, TypeStruct arrayType)
        : arrayType(arrayType), value(value) {constNode = new ConstantNode(value, elementType);}

    void print() override;

    std::string value;
    ConstantNode *constNode;

    TypeStruct arrayType;
    TypeStruct getImmediateType() override { return arrayType; }

    TypeStruct getType() override;
    TypeStruct getSextType() override { return TypeStruct(DataType::INTEGER, 64); }

    NodeVariant getVariant() override { return NodeVariant::GLOBAL_ARRAY; }
};

class FNegNode : public Node {
  public:
    void print() override;
    TypeStruct getType() { return TypeStruct(DataType::FLOAT, 64); }
};

class StructAccessNode : public DerefNode {
  public:
    void print() override;
};

} // namespace GNN

#endif