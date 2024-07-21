from nid.ast.graph_builder.common.definitions import PythonNodeEdgeDefinitions
from nid.ast.graph_builder.common.parser import GraphParser
from nid.ast.graph_builder.v1.parser import GraphParserV1
from nid.ast.graph_builder.v2.parser import GraphParserV2
from nid.ast.graph_builder.v3.definitions import PythonNodeEdgeDefinitionsV3
from nid.ast.graph_builder.v3.parser import GraphParserV3


def select_graph_builder(
        parser_name, add_reverse_edges: bool = True, save_node_strings: bool = True,
        add_mention_instances: bool = False, parse_constants: bool = False
) -> GraphParser:
    assert parser_name in ["GraphParser", "GraphParserV1", "GraphParserV2", "GraphParserV3"]

    if parser_name == "GraphParser":
        parser = GraphParser(
            PythonNodeEdgeDefinitions,
        )
    elif parser_name == "GraphParserV1":
        parser = GraphParserV1()
    elif parser_name == "GraphParserV2":
        parser = GraphParserV2()
    elif parser_name == "GraphParserV3":
        parser = GraphParserV3(
            PythonNodeEdgeDefinitionsV3,
            add_reverse_edges=add_reverse_edges,
            save_node_strings=save_node_strings,
            add_mention_instances=add_mention_instances,
            parse_constants=parse_constants,
        )
    else:
        raise ValueError(f"Invalid parser: {parser_name}")

    return parser
