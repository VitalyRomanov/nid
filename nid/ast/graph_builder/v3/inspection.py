import ast
from typing import Type

from nid.ast.graph_builder.v3.definitions import PythonNodeEdgeDefinitions


def generate_available_edges(node_edge_definition_class: Type[PythonNodeEdgeDefinitions]):
    node_types = node_edge_definition_class.node_types()
    for nt in sorted(node_types):
        if hasattr(ast, nt):
            fl = sorted(getattr(ast, nt)._fields)
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
        if hasattr(ast, nt):
            fl = sorted(d[nt])
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")
