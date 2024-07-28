import hashlib
from typing import Optional, Dict, Set

import pandas as pd

from nid.ast.graph_builder.v1.definitions import PythonSharedNodes
from nid.ast.graph_builder.v1.primitives import GNode


class NodeIdResolver:
    def __init__(self):
        self.node_ids = {}
        self.new_nodes = []
        self.stashed_nodes = []

        self._resolver_cache = dict()

    def stash_new_nodes(self):
        """
        Put new nodes into temporary storage.
        :return: Nothing
        """
        self.stashed_nodes.extend(self.new_nodes)
        self.new_nodes = []

    def get_node_id(self, type_name):
        return hashlib.md5(type_name.encode('utf-8')).hexdigest()

    def resolve_node_id(self, node: GNode, **kwargs) -> GNode:
        """
        Resolve node id from name and type, create new node is no nodes like that found.
        :param node: node
        :param kwargs:
        :return: updated node (return object with the save reference as input)
        """
        if not hasattr(node, "id") or node.id is None:
            node_repr = f"{node.name.strip()}_{node.type.strip()}"

            if node_repr in self.node_ids:
                node_id = self.node_ids[node_repr]
                node.setprop("id", node_id)
            else:
                new_id = self.get_node_id(node_repr)
                self.node_ids[node_repr] = new_id

                if not PythonSharedNodes.is_shared(node) and not node.name == "unresolved_name":
                    assert "0x" in node.name

                self.new_nodes.append(
                    {
                        "id": new_id,
                        "type": node.type,
                        "name": node.name,
                        "scope": pd.NA,
                        "string": node.string
                    }
                )
                if hasattr(node, "scope"):
                    self.resolve_node_id(node.scope)
                    self.new_nodes[-1]["scope"] = node.scope.id
                node.setprop("id", new_id)
        return node

    def prepare_for_write(self, from_stashed: bool = False):
        nodes = self.new_nodes_for_write(from_stashed)[
            ['id', 'type', 'name', 'scope', 'string']
        ]

        return nodes

    def new_nodes_for_write(self, from_stashed: bool = False) -> Optional[pd.DataFrame]:

        new_nodes = pd.DataFrame(self.new_nodes if not from_stashed else self.stashed_nodes)
        if len(new_nodes) == 0:
            return None

        new_nodes = new_nodes[
            ['id', 'type', 'name', 'scope', 'string']
        ].astype({"scope": "string", "id": "string"})
        new_nodes["node_hash"] = new_nodes["id"]

        return new_nodes

    def adjust_ast_node_types(self, mapping: Dict[str, str], from_stashed: bool = False) -> None:
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        for node in nodes:
            node["type"] = mapping.get(node["type"], node["type"])

    def drop_nodes(self, node_ids_to_drop: Set, from_stashed: bool = False) -> None:
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        position = 0
        while position < len(nodes):
            if nodes[position]["id"] in node_ids_to_drop:
                nodes.pop(position)
            else:
                position += 1

    def map_mentioned_in_to_global(self, mapping: Dict[str, str], from_stashed: bool = False) -> None:
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        for node in nodes:
            node["scope"] = mapping.get(node["scope"], node["scope"])
