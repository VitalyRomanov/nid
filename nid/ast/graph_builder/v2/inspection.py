from nid import ast
from nid.ast.graph_builder.v2.definitions import PythonNodeEdgeDefinitions


def generate_available_edges():
    node_types = PythonNodeEdgeDefinitions.node_types()
    for nt in sorted(node_types):
        if hasattr(ast, nt):
            fl = sorted(getattr(ast, nt)._fields)
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")


def generate_utilized_edges():
    d = dict()
    d.update(PythonNodeEdgeDefinitions.ast_node_type_edges)
    d.update(PythonNodeEdgeDefinitions.overridden_node_type_edges)
    for nt in sorted(d.keys()):
        if hasattr(ast, nt):
            fl = sorted(d[nt])
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")
