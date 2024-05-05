#include "nodePrinter.h"
#include "args.h"
#include "node.h"

namespace {
std::string addPragmaToLabel(GNN::Node *node, std::string label) {
    if (node->unrollFactor.full > 1) {
        label += "\n Unroll: " + std::to_string(node->unrollFactor.full);
    }
    if (node->partitionFactor1 > 0) {
        label += "\n Partition Factor 1: " + std::to_string(node->partitionFactor1);
    }
    if (node->partitionFactor2 > 0) {
        label += "\n Partition Factor 2: " + std::to_string(node->partitionFactor2);
    }
    if(node->tripcount.full > 1){
        label += "\n Tripcount: " + std::to_string(node->tripcount.full);
    }
    if (node->partitionType1 != "none") {
        label += "\n Partition 1: " + node->partitionType1;
    }
    if (node->partitionType2 != "none") {
        label += "\n Partition 2: " + node->partitionType2;
    }
    if (node->inlined){
        label += "\n Inlined";
    }    
    if (node->pipelined){
        label += "\n Pipelined";
    }
    return label;
}

void output(std::string message) { 
    std::cout << message << std::endl; 
}

std::string toVariableType(GNN::Node *node) {
    GNN::TypeStruct type;
    try{
        type = node->getImmediateType();
    } catch(...){
        return "NA";
    }
    if (type.isVoid) {
        return "void";
    }
    if (type.dataType == GNN::DataType::INTEGER) {
        return "int";
    } else if (type.dataType == GNN::DataType::FLOAT) {
        return "float";
    }
    throw std::runtime_error("toVariableType reached unreachable control flow");
}

std::string typeToBitwidth(GNN::Node *node) {
    GNN::TypeStruct type;
    try{
        type = node->getImmediateType();
    } catch(...){
        return "0";
    }
    if (type.isVoid) {
        return "0";
    }
    return std::to_string(type.bitwidth);
}

} // namespace

namespace GNN {
NodePrinter::NodePrinter(Node *node, const std::string &color) : color(color) {
    this->node = node;
    attributes["group"] = node->groupName;
    attributes["nodeType"] = "instruction";

    if (Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        attributes["partitionFactor1"] = std::to_string(node->partitionFactor1);
        attributes["partitionFactor2"] = std::to_string(node->partitionFactor2);
        attributes["fullUnrollFactor"] = std::to_string(node->unrollFactor.full);
        attributes["unrollFactor1"] = std::to_string(node->unrollFactor.first);
        attributes["unrollFactor2"] = std::to_string(node->unrollFactor.second);
        attributes["unrollFactor3"] = std::to_string(node->unrollFactor.third);
        attributes["partition1"] = node->partitionType1;
        attributes["partition2"] = node->partitionType2;
        attributes["inlined"] = "not_inlined";
        attributes["tripcount"] = std::to_string(node->tripcount.full);
    } else {
        attributes["numeric"] = "0";
    }

    if (Nodes::graphGenerator->checkArg(ABSORB_TYPES)) {
        if (!Nodes::graphGenerator->checkArg(DONT_DISPLAY_TYPES)) {
            if (Nodes::graphGenerator->checkArg(ONE_HOT_TYPES)) {
                attributes["datatype"] = node->getTypeToPrint();
            } else {
                attributes["datatype"] = toVariableType(node);
                attributes["bitwidth"] = typeToBitwidth(node);
                attributes["arrayWidth"] = "1";
            }
        }
    }

    if (Nodes::graphGenerator->checkArg(ADD_BB_ID)) {
        attributes["bbID"] = std::to_string(node->bbID);
    }
    if (Nodes::graphGenerator->checkArg(ADD_FUNC_ID)) {
        attributes["funcID"] = std::to_string(node->functionID);
    }
    if(Nodes::graphGenerator->checkArg(ADD_NUM_CALLS)){
        attributes["numCalls"] = std::to_string(Nodes::graphGenerator->getCallsNums(node->funcDec));
        attributes["numCallSites"] = std::to_string(Nodes::graphGenerator->getCallSiteNums(node->funcDec));
    }
}

void NodePrinter::print() {
    std::string out = "node" + std::to_string(node->id);
    out += " [";
    out += "style=filled fillcolor=\"" + color + "\" ";

    if (Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        attributes["label"] = addPragmaToLabel(node, attributes["label"]);
    }

    if (attributes.count("datatype")) {
        attributes["label"] += "\n" + attributes["datatype"];
    }
    if (attributes.count("bitwidth")) {
        attributes["label"] += "\n" + attributes["bitwidth"] + " bits";
    }
    if (attributes.count("arrayWidth")) {
        attributes["label"] += "\n Array Width: " + attributes["arrayWidth"];
    }
    if (attributes.count("bbID")) {
        attributes["label"] += "\n BB ID: " + attributes["bbID"];
    }
    if (attributes.count("funcID")) {
        attributes["label"] += "\n Func ID: " + attributes["funcID"];
    }

    if (!Nodes::graphGenerator->checkArg(ADD_NODE_TYPE)) {
        attributes.erase("nodeType");
    }

    if (attributes.count("nodeType")) {
        attributes["label"] += "\n Node Type: " + attributes["nodeType"];
    }

    if(attributes.count("numCalls")){
        attributes["label"] += "\n Num Calls: " + attributes["numCalls"];
    }

    if(attributes.count("numCallSites")){
        attributes["label"] += "\n Num Call Sites: " + attributes["numCallSites"];
    }


    if (!node->extraNote.empty()) {
        attributes["label"] += "\n" + node->extraNote;
    }

    for (std::pair<std::string, std::string> attribute : attributes) {
        out += attribute.first;
        out += "=\"";
        out += attribute.second;
        out += "\" ";
    }
    out += "]";
    output(out);
}
} // namespace GNN