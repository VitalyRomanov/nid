from typing import Type

from nid.ast.graph_builder.common.definitions import PythonNodeEdgeDefinitions, iterate_ast_node_types, \
    get_available_fields_for_ast_node, get_known_ast_node_types


# TODO move to tests?
def verify_current_python_version_support(definitions_class: Type[PythonNodeEdgeDefinitions]):
    known_node_types = get_known_ast_node_types(definitions_class)

    unknown = {}
    for ast_node_type_name in iterate_ast_node_types():
        if ast_node_type_name not in known_node_types:
            unknown[ast_node_type_name] = get_available_fields_for_ast_node(ast_node_type_name)

    return unknown
