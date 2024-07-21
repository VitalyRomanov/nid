import argparse
from pathlib import Path

from nid import select_graph_builder, into_graph
from nid.validation.ast_node_examples import PythonCodeExamplesForNodes
from nid.visualization import visualize


def main(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parser = select_graph_builder(args.variety)

    for node, code in PythonCodeExamplesForNodes.examples.items():
        print(node)
        graph = into_graph(code, parser)
        nodes, edges, offsets = graph.as_df()
        visualize(
            nodes, 
            edges, 
            args.output_dir / f"{node}_{parser.__class__.__name__}.png",
            show_reverse=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variety", choices=["GraphParser", "GraphParserV1", "GraphParserV2", "GraphParserV3"])
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()
    main(args)
