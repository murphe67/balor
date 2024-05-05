#include "pragmaParser.h"
#include "rose.h"

#include <boost/algorithm/string.hpp>

namespace GNN {

void PragmaParser::parseInlinePragmas(std::set<SgFunctionDeclaration *> funcDecs) {
    for (SgFunctionDeclaration *funcDec : funcDecs) {
        if (funcDec->get_definition()) {
            SgBasicBlock *bb = funcDec->get_definition()->get_body();

            functionInlined = false;
            parsePragmas(bb);

            if (functionInlined) {
                inlinedFunctions.push(funcDec);
            }
        }
    }
}

void PragmaParser::parsePragmas(SgBasicBlock *bb) {
    std::cerr << "parse pragmas" << std::endl;
    std::vector<SgNode *> pragmas = NodeQuery::querySubTree(bb, V_SgPragmaDeclaration, AstQueryNamespace::ChildrenOnly);

    unrollFactor = 1;
    tripcount = 1;

    for (SgNode *pragmaNode : pragmas) {
        // cast to SgPragma to get access to member variables
        SgPragma *pragma = isSgPragmaDeclaration(pragmaNode)->get_pragma();

        std::string pragmaText = pragma->get_name();
        // convert the pragma text (everything after #pragma) to uppercase
        std::string pragmaTextUpper = boost::algorithm::to_upper_copy(pragma->get_name());

        // split the pragma into word tokens using boost to prevent whitespace issues
        std::vector<std::string> pragmaTextVectorUpper;
        boost::algorithm::split(pragmaTextVectorUpper, pragmaTextUpper, boost::is_any_of(" ="));

        std::vector<std::string> pragmaTextVector;
        boost::algorithm::split(pragmaTextVector, pragmaText, boost::is_any_of(" ="));
        if (pragmaTextVectorUpper.size() > 1) {
            if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "INTERFACE") {
                std::string variableName;
                std::string bundle;
                bool foundBundle = false;
                bool foundVariable = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "PORT") {
                        variableName = pragmaTextVector[i + 1];
                    }
                    if (pragmaTextVectorUpper[i] == "BUNDLE") {
                        bundle = pragmaTextVector[i + 1];
                    }
                }
                if (!foundBundle || !foundVariable) {
                    throw std::runtime_error("Couldn't find variable or bundle on interface pragma");
                }
                variableToPortType[variableName] = bundle;
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "UNROLL") {
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "FACTOR") {
                        try {
                            unrollFactor = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read unroll factor from pragma");
                        }
                    }
                }
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "PIPELINE") {
                pipelined = true;
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "RESOURCE") {
                std::string core;
                std::string variable;
                bool foundCore = false;
                bool foundVariable = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "CORE") {
                        foundCore = true;
                        core = pragmaTextVector[i + 1];
                    }
                    if (pragmaTextVectorUpper[i] == "VARIABLE") {
                        foundVariable = true;
                        variable = pragmaTextVector[i + 1];
                    }
                }
                if (!foundCore || !foundVariable) {
                    throw std::runtime_error("Couldn't find core or variable on resource pragma");
                }

                graphGenerator->variableMapper->resourceTypeMap[variable] = core;
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "ARRAY_PARTITION") {
                std::string type;
                int factor;
                int dim;
                std::string variable;
                bool foundType = false;
                bool foundVariable = false;
                bool foundFactor = false;
                bool foundDim = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "TYPE") {
                        foundType = true;
                        type = pragmaTextVector[i + 1];
                    } else if (pragmaTextVectorUpper[i] == "VARIABLE") {
                        foundVariable = true;
                        variable = pragmaTextVector[i + 1];
                    } else if (pragmaTextVectorUpper[i] == "FACTOR") {
                        try {
                            foundFactor = true;
                            factor = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read factor from array partition pragma");
                        }
                    } else if (pragmaTextVectorUpper[i] == "DIM") {
                        try {
                            foundDim = true;
                            dim = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read dim from array partition pragma");
                        }
                    }
                }
                bool found = false;
                if (foundType && foundVariable && foundFactor && foundDim) {
                    found = true;
                } else if (foundType && foundVariable && foundDim && type == "complete") {
                    found = true;
                    factor = 1;
                } else {
                    throw std::runtime_error("Couldn't find one of type, variable, factor or dim on resource pragma");
                }
                graphGenerator->variableMapper->arrayPartitionMap[variable].push(std::make_tuple(type, factor, dim));
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "INLINE") {
                if (pragmaTextVectorUpper[2] == "ON") {
                    functionInlined = true;
                }
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "TRIPCOUNT") {
                bool foundAvg = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "AVG") {
                        try {
                            foundAvg = true;
                            tripcount = std::stof(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read average tripcount from tripcount pragma");
                        }
                    }
                }
                if (!foundAvg) {
                    throw std::runtime_error("Couldn't find avg on tripcount pragma");
                }
            }
        }
    }

    unrollHierarchy.moveDown(unrollFactor);
    tripcountHierarchy.moveDown(tripcount);

    if(pipelined && !getPreviouslyPipelined()){
        pipelineTripcount = tripcountHierarchy.fullFactor;
    }
    std::cerr << "end parse pragmas" << std::endl;
}

void PragmaParser::stackPragmas() {
    pipelineStack.push(pipelined);
}

void PragmaParser::unstackPragmas() {
    unrollHierarchy.moveUp();
    tripcountHierarchy.moveUp();

    pipelined = pipelineStack.top();
    pipelineStack.pop();
}

void PragmaParser::enterLoopCondition() {
    unrollHierarchy.pauseFactor();
    tripcountHierarchy.pauseFactor();
}
void PragmaParser::exitLoopCondition() {
    unrollHierarchy.unpauseFactor();
    tripcountHierarchy.unpauseFactor();
}

void PragmaParser::enterLoopInc() {
    unrollHierarchy.pauseFactor();
}
void PragmaParser::exitLoopInc() {
    unrollHierarchy.unpauseFactor();
}

StackedFactor PragmaParser::getUnrollFactor() { 
    if(graphGenerator->stateNode){
        return graphGenerator->stateNode->unrollFactor;
    }
    return {unrollHierarchy.fullFactor,  unrollHierarchy.factor1, unrollHierarchy.factor2, unrollHierarchy.factor3}; 
}

StackedFactor PragmaParser::getTripcount(){
    if(graphGenerator->stateNode){
        return graphGenerator->stateNode->tripcount;
    }
    return {tripcountHierarchy.fullFactor, tripcountHierarchy.factor1, tripcountHierarchy.factor2, tripcountHierarchy.factor3}; 
}

bool PragmaParser::getPipelined() { return pipelined; }

bool PragmaParser::getPreviouslyPipelined() { 
    if(pipelineStack.empty()){
        return false;
    }
    return pipelineStack.top();

}

float PragmaParser::getPipelineTripcount(){
    return pipelineTripcount;
}

std::string PragmaParser::getPortType(const std::string &variable) {
    if (variableToPortType.count(variable)) {
        return variableToPortType[variable];
    }
    return "1 Port Ram";
}
} // namespace GNN

