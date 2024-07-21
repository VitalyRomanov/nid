from typing import Type

from nid.ast.graph_builder.common.definitions import PythonNodeEdgeDefinitions


def verify_current_python_version_support(edge_definitions_class: Type[PythonNodeEdgeDefinitions]):
    import ast
    items = dir(ast)

    known_node_types = (
        set(edge_definitions_class.node_types()) |
        edge_definitions_class.operand_nodes |
        edge_definitions_class.control_flow_nodes
    )

    unknown = {}
    for item in items:
        if (
                item not in known_node_types and
                hasattr(getattr(ast, item), "_fields")
        ):
            unknown[item] = list(getattr(ast, item)._fields)

    unknown.pop("AST", None)
    return unknown
