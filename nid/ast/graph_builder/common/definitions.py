import ast
from collections import defaultdict
from enum import Enum
from itertools import chain
from typing import Dict, List, Optional, Set, Type, Any, Iterable

GraphNodeId = str
NodeImage = Dict[str, Any]
EdgeImage = Dict[str, Any]


class PythonNodeEdgeDefinitions:
    _node_type_enum_initialized: bool = False
    _edge_type_enum_initialized: bool = False
    _node_type_enum: Optional[Type[Enum]] = None
    _edge_type_enum: Optional[Type[Enum]] = None

    ast_node_type_edges = get_all_node_edge_associations()
    overridden_node_type_edges = defaultdict(list)
    context_edge_names = defaultdict(list)
    extra_edge_types = {"next"}

    # exceptions needed when we do not want to filter some edge types using a simple rule `_rev`
    reverse_edge_exceptions = {
    }

    iterable_nodes = {  # parse_iterable
        "List", "Tuple", "Set"
    }

    named_nodes = {
        "Name", "NameConstant"  # parse_name
    }

    constant_nodes = {
        "Constant"  # parse_Constant
    }

    operand_nodes = {  # parse_op_name
        "And", "Or", "Not", "Is", "Gt", "Lt", "GtE", "LtE", "Eq", "NotEq", "Ellipsis", "Add", "Mod",
        "Sub", "UAdd", "USub", "Div", "Mult", "MatMult", "Pow", "FloorDiv", "RShift", "LShift", "BitAnd",
        "BitOr", "BitXor", "IsNot", "NotIn", "In", "Invert"
    }

    control_flow_nodes = {  # parse_control_flow
        "Continue", "Break", "Pass"
    }

    # extra node types exist for keywords and attributes to prevent them from
    # getting mixed with local variable mentions
    extra_node_types = set()

    @classmethod
    def regular_node_types(cls) -> Set[str]:
        return set(cls.ast_node_type_edges.keys())

    @classmethod
    def overridden_node_types(cls) -> Set[str]:
        return set(cls.overridden_node_type_edges.keys())

    @classmethod
    def node_types(cls) -> List[str]:
        return list(  # TODO why not set?
            cls.regular_node_types() |
            cls.overridden_node_types() |
            cls.iterable_nodes | cls.named_nodes | cls.constant_nodes |
            cls.operand_nodes | cls.control_flow_nodes | cls.extra_node_types
        )

    @classmethod
    def scope_edges(cls) -> Set[str]:
        return set(map(lambda x: x, chain(*cls.context_edge_names.values())))  # "defined_in_" +

    @classmethod
    def auxiliary_edges(cls) -> Set[str]:
        direct_edges = cls.scope_edges() | cls.extra_edge_types
        reverse_edges = cls.compute_reverse_edges(direct_edges)
        return direct_edges | reverse_edges

    @classmethod
    def compute_reverse_edges(cls, direct_edges) -> Set[str]:
        reverse_edges = set()
        for edge in direct_edges:
            if edge in cls.reverse_edge_exceptions:
                reverse = cls.reverse_edge_exceptions[edge]
                if reverse is not None:
                    reverse_edges.add(reverse)
            else:
                reverse_edges.add(edge + "_rev")
        return reverse_edges

    @classmethod
    def edge_types(cls) -> List[str]:  # TODO why not set
        direct_edges = list(
            set(chain(*cls.ast_node_type_edges.values())) |
            set(chain(*cls.overridden_node_type_edges.values())) |
            cls.scope_edges() |
            cls.extra_edge_types
            # | cls.named_nodes | cls.constant_nodes |
            # cls.operand_nodes | cls.control_flow_nodes | cls.extra_node_types
        )

        reverse_edges = list(cls.compute_reverse_edges(direct_edges))
        return direct_edges + reverse_edges

    def __init__(self):
        raise Exception(f"Cannot instantiate. {self.__class__.__name__} is a static class")

    @classmethod
    def make_node_type_enum(cls) -> Type[Enum]:
        if not cls._node_type_enum_initialized:
            cls._node_type_enum = Enum("NodeTypes", " ".join(cls.node_types()))  # type: ignore
            cls._node_type_enum_initialized = True
        # assert cls._node_type_enum is not None  # silence type checker
        return cls._node_type_enum  # type: ignore

    @classmethod
    def make_edge_type_enum(cls) -> Type[Enum]:
        if not cls._edge_type_enum_initialized:
            cls._edge_type_enum = Enum("EdgeTypes", " ".join(cls.edge_types()))  # type: ignore
            cls._edge_type_enum_initialized = True
        # assert cls._edge_type_enum is not None  # silence type checker
        return cls._edge_type_enum  # type: ignore
