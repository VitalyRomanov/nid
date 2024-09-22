import ast
from typing import Iterable, Dict, List, Type, Set

from nid.ast.graph_builder import PythonNodeEdgeDefinitions


class NodeTypeNameError(Exception):
    ...


def get_available_fields_for_ast_node(ast_node_type_name: str) -> List[str]:
    ast_node_type = getattr(ast, ast_node_type_name, None)
    if ast_node_type is None:
        raise NodeTypeNameError(f"No AST node type found for {ast_node_type_name}")

    if hasattr(ast_node_type, "_fields"):
        return list(ast_node_type._fields)


def iterate_ast_node_types() -> Iterable[str]:
    for item in dir(ast):
        if item != "AST" and hasattr(getattr(ast, item), "_fields"):
            yield item


def get_all_node_edge_associations() -> Dict[str, List[str]]:
    ast_node_type_fields = {}

    for ast_note_type_name in iterate_ast_node_types():
        ast_node_type_fields[ast_note_type_name] = get_available_fields_for_ast_node(ast_note_type_name)

    return ast_node_type_fields


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
