import ast
from typing import Type, Set

from nid.ast.graph_builder import PythonNodeEdgeDefinitions
from nid.ast.graph_builder.common.definitions import get_available_fields_for_ast_node


def get_known_ast_node_types(definition_class: Type[PythonNodeEdgeDefinitions]) -> Set[str]:
    return (
            set(definition_class.node_types()) |
            definition_class.operand_nodes |
            definition_class.control_flow_nodes
    )


def is_ast_node_type(node_type_name: str) -> bool:
    return hasattr(ast, node_type_name)


def generate_available_edges(node_edge_definition_class: Type[PythonNodeEdgeDefinitions]):
    node_types = node_edge_definition_class.node_types()
    for nt in sorted(node_types):
        if is_ast_node_type(nt):
            fl = sorted(get_available_fields_for_ast_node(nt))
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")


def generate_utilized_edges(node_edge_definition_class: Type[PythonNodeEdgeDefinitions]):
    d = dict()
    d.update(node_edge_definition_class.ast_node_type_edges)
    d.update(node_edge_definition_class.overridden_node_type_edges)
    for nt in sorted(d.keys()):
        if is_ast_node_type(nt):
            fl = sorted(d[nt])
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")
