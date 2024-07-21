from nid.ast.graph_builder import select_graph_builder
from nid.ast.graph_builder.common.graph import ParsedGraph


def into_graph(source_code: str, parser) -> ParsedGraph:
    if isinstance(parser, str):
        parser = select_graph_builder(parser)

    graph = parser.parse(source_code)
    return graph
