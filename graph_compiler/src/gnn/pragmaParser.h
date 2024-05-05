#ifndef GNN_PRAGMA_PARSER_H
#define GNN_PRAGMA_PARSER_H

#include "graphGenerator.h"
#include "node.h"
#include "rose.h"
#include <map>
#include <string>


namespace GNN {
class GraphGenerator;

struct StackedFactor;

class FactorHierarchy{
  public:
    float factor1 = 1;
    float factor2 = 1;
    float factor3 = 1;

    float fullFactor = 1;


    //sliding window: moving down moves values up
    void moveDown(int newFactor){
      fullFactor *= newFactor;

      factorStack.push(factor3);
      factorOver = factor3;
      factor3 = factor2;
      factor2 = factor1;
      factor1 = newFactor;
    }

    //sliding window: moving up moves values down
    void moveUp(){
      fullFactor /= factor1;

      factor1 = factor2;
      factor2 = factor3;
      factor3 = factorStack.top();
      factorStack.pop();
    }

    void pauseFactor(){
      fullFactor /= factor1;

      factorUnder = factor1;
      factor1 = factor2;
      factor2 = factor3;
      factor3 = factorOver;
    }

    void unpauseFactor(){
      factor3 = factor2;
      factor2 = factor1;
      factor1 = factorUnder;

      fullFactor *= factor1;
    }

  private:
    float factorUnder = 1;
    float factorOver = 1;

    std::stack<int> factorStack;
};

class PragmaParser {
  public:
    PragmaParser(GraphGenerator *graphGenerator) : graphGenerator(graphGenerator) {}


    GraphGenerator *graphGenerator;

    void parsePragmas(SgBasicBlock *bb);
    void parseInlinePragmas(std::set<SgFunctionDeclaration *> funcDecs);

    std::string getPortType(const std::string &variable);

    void stackPragmas();
    void unstackPragmas();

    void enterLoopCondition();
    void exitLoopCondition();

    void enterLoopInc();
    void exitLoopInc();

    StackedFactor getUnrollFactor();
    bool getPipelined();
    bool getPreviouslyPipelined();

    float getPipelineTripcount();

    StackedFactor getTripcount();

    bool functionInlined = false;
    std::queue<SgFunctionDeclaration *> inlinedFunctions;

  private:
    std::map<std::string, std::string> variableToPortType;


    FactorHierarchy unrollHierarchy;
    FactorHierarchy tripcountHierarchy;

    float tripcount = 1;
    int unrollFactor;

    bool pipelined = false;
    bool previouslyPipelined = false;
    float pipelineTripcount;


    // std::stack<float> tripcountStack;
    // float realTripcount;


    std::stack<bool> pipelineStack;
};

} // namespace GNN

#endif