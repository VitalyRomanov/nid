from dataclasses import dataclass
from typing import List, Dict, Union

import pandas as pd

from nid.ast.graph_builder.common.dataframe import nodes_edges_to_df


@dataclass
class ParsedGraph:
    source_code: str
    nodes: List[Dict[str, Union[int, str]]]
    edges: List[Dict[str, Union[int, str]]]

    def _get_offsets(self, edges):
        offsets = edges[["src", "offset_start", "offset_end", "scope"]] \
            .dropna() \
            .rename({
            "src": "node_id"  # , "offset_start": "start", "offset_end": "end" #, "scope": "mentioned_in"
        }, axis=1)

        # assert len(offsets) == offsets["node_id"].nunique()  # there can be several offsets for constants

        return edges, offsets

    def _assign_node_strings(self, nodes, offsets):
        # used to convert ot df
        start_map = {}
        end_map = {}
        for node_id, part in offsets.groupby("node_id"):
            if len(part) > 1:
                continue
            start_map[node_id] = part["offset_start"].iloc[0]
            end_map[node_id] = part["offset_end"].iloc[0]

        existing_string = dict(zip(nodes["id"], nodes["string"]))
        existing_start = dict(zip(nodes["id"], nodes["offset_start"]))
        existing_end = dict(zip(nodes["id"], nodes["offset_end"]))

        def assign_string(id_):
            assert self.source_code is not None
            if id_ in start_map:
                node_string = self.source_code[start_map[id_]: end_map[id_]]
                if id_ not in existing_string:
                    assert node_string == existing_string[id_]
            elif id_ in existing_string:
                node_string = existing_string[id_]
            else:
                node_string = pd.NA
            return node_string

        def assign_start(id_):
            if id_ in start_map:
                start = existing_start[id_]
                if id_ not in existing_start:
                    assert start == existing_start[id_]
            elif id_ in existing_start:
                start = existing_start[id_]
            else:
                start = pd.NA
            return start

        def assign_end(id_):
            if id_ in end_map:
                end = existing_end[id_]
                if id_ not in existing_end:
                    assert end == existing_end[id_]
            elif id_ in existing_end:
                end = existing_end[id_]
            else:
                end = pd.NA
            return end

        nodes["string"] = nodes["id"].apply(assign_string)
        nodes["offset_start"] = nodes["id"].apply(assign_start)
        nodes["offset_end"] = nodes["id"].apply(assign_end)

        return nodes

    def as_df(self):
        # import pandas as pd
        # nodes = pd.DataFrame.from_records(self.nodes)
        # edges = pd.DataFrame.from_records(self.edges)
        nodes, edges, offsets = nodes_edges_to_df(self.nodes, self.edges, make_table=True)
        edges, offsets = self._get_offsets(edges)
        nodes = self._assign_node_strings(nodes, offsets)

        return nodes, edges, offsets
