import argparse
import pygraphviz as pgv
import sys

def reorderNodes(input_file, output_file):
    # Create a graph from the DOT file
    graph = pgv.AGraph(input_file)

    # Dictionary to store nodes grouped by their 'group'
    nodes_by_group = {}

    # List to store edges
    edges = []

    # Iterate over nodes in the graph
    for node in graph.nodes():
        group = node.attr.get('group', 'default')

        # Append the DOT line to the corresponding group in the dictionary
        nodes_by_group.setdefault(group, []).append(str(node))

    clusters = {}
    for node in graph.nodes():
        group = node.attr['group']
        if group != "External":
            if group not in clusters:
                clusters[group] = graph.add_subgraph(name=f"cluster_{group}", label=group)
            clusters[group].add_node(node)
        
    # for cluster in clusters.values():
    #     cluster.graph_attr["splines"] ="curved"
    # Write the DOT file with clusters
    graph.write(output_file)


def main():
    parser = argparse.ArgumentParser(description="Run programl on .cpp file")
    parser.add_argument("input_file", help="Path to the input dot file")
    parser.add_argument("output_file", help="Path to the output dot file")

    args = parser.parse_args()
    reorderNodes(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

