import argparse
from pathlib import Path

from nid.ast.graph_builder.common.definitions import PythonNodeEdgeDefinitions
from nid.ast.graph_builder.common.parser import GraphParser

from nid.validation.ast_node_examples import PythonCodeExamplesForNodes
from nid.visualization import visualize


def main(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for node, code in PythonCodeExamplesForNodes.examples.items():
        print(node)
        variety = "GraphParser"
        parser = GraphParser(PythonNodeEdgeDefinitions)
        nodes, edges = parser.parse(code).as_df()
        visualize(
            nodes, 
            edges, 
            args.output_dir / f"{node}_{variety}.png", 
            show_reverse=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()
    main(args)
