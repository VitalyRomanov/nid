from enum import Enum
from typing import Optional, Union

from nid.ast.graph_builder.common.parser import GraphNodeId
from nid.ast.string_tools import string_hash


class GraphNode:
    name: str
    type: Enum
    string: Optional[str]
    node_hash: Optional[str]

    def __init__(self, name, type, string=None, **kwargs):
        self.name = name
        self.type = type
        self.string = string
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other) -> bool:
        return self.name == other.name and self.type == other.type

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        return (self.name, self.type).__hash__()

    def setprop(self, key, value):
        setattr(self, key, value)

    @property
    def hash_id(self):
        if not hasattr(self, "node_hash") or self.node_hash is None:
            self.node_hash = string_hash(f"{self.type.name.strip()}_{self.name.strip()}")
        return self.node_hash


class GraphEdge:
    src: GraphNodeId
    dst: GraphNodeId
    type: Enum
    scope: Optional[GraphNodeId]
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    edge_hash: Optional[str] = None

    def __init__(
            self, src: GraphNodeId, dst: GraphNodeId, type, scope: Optional[GraphNodeId] = None,
    ):
        self.src = src
        self.dst = dst
        self.type = type
        self.scope = scope
        self.offset_start = None
        self.offset_end = None

    def assign_positions(self, positions, prefix: Optional[str] = None):
        if prefix is None:
            if positions is not None:
                self.offset_start = positions[0]
                self.offset_end = positions[1]
        else:
            if positions is not None:
                setattr(self, f"{prefix}_offset_start", positions[0])
                setattr(self, f"{prefix}_offset_end", positions[1])

    def make_reverse(self, *args, **kwargs):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.__dict__[item]

    @property
    def hash_id(self):
        if not hasattr(self, "edge_hash") or self.edge_hash is None:
            self.edge_hash = string_hash(f"{self.src}_{self.dst}_{self.type}")
        return self.edge_hash
