#ifndef GNN_NODE_PRINTER_H
#define GNN_NODE_PRINTER_H

#include "node.h"
#include <map>
#include <string>

namespace GNN {

class Node;

class NodePrinter {
  public:
    NodePrinter(Node *node, const std::string &color);
    std::string color;
    std::map<std::string, std::string> attributes;

    Node *node;

    void print();
};

} // namespace GNN

#endif