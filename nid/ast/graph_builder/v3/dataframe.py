from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd

from nid.ast.graph_builder.v3.primitives import GraphEdge, GraphNode


def nodes_edges_to_df(
        nodes: List[GraphNode], edges: List[GraphEdge], make_table=True
    ) -> Tuple[Union[List[Dict], pd.DataFrame], Union[List[Dict], pd.DataFrame], Union[List[Dict], pd.DataFrame]]:
    edge_specification: Dict[str, Tuple[str, str, Optional[Callable]]] = {
        "id": ("edge_hash", "string", None),
        "src": ("src", "string", None),
        "dst": ("dst", "string", None),
        "type": ("type", "string", lambda x: x.name),
        "scope": ("scope", "string", None),
        "offset_start": ("offset_start", "Int64", None),
        "offset_end": ("offset_end", "Int64", None),
    }

    node_specification: Dict[str, Tuple[str, str, Optional[Callable]]] = {
        "id": ("node_hash", "string", None),
        "name": ("name", "string", None),
        "type": ("type", "string", lambda x: x.name),
        "scope": ("scope", "string", None),
        "string": ("string", "string", None),
        "offset_start": ("offset_start", "Int64", None),
        "offset_end": ("offset_end", "Int64", None),
    }

    offset_specification: Dict[str, Tuple[str, str, Optional[Callable]]] = {
        "node_id": ("node_id", "string", None),
        "offset_start": ("offset_start", "Int64", None),
        "offset_end": ("offset_end", "Int64", None),
        "scope": ("scope", "string", None),
    }

    def format_for_table(collection: List[Union[GraphNode, GraphEdge]]) -> List[Dict[str, Any]]:
        entries = []
        for record in collection:
            entries.append(record.__dict__)
        return entries

    def create_table(collection, specification):
        table = pd.DataFrame.from_records(collection)

        column_order = []
        for trg_col, (src_col, dtype, preproc_fn) in specification.items():
            trg = table[src_col]
            if preproc_fn is not None:
                trg = trg.apply(preproc_fn)
            table[trg_col] = trg.astype(dtype)
            column_order.append(trg_col)
        return table[column_order]

    def get_offsets(edges) -> List[Dict[str, Any]]:
        offsets = []
        for edge in edges:
            if edge["offset_start"] is not None:
                offsets.append({
                    "node_id": edge["src"],
                    "offset_start": edge["offset_start"],
                    "offset_end": edge["offset_end"],
                    "scope": edge["scope"],
                })
        return offsets

    nodes = format_for_table(nodes)  # type: ignore
    edges = format_for_table(edges)  # type: ignore
    offsets = get_offsets(edges)

    if make_table is True:
        nodes = create_table(nodes, node_specification) if len(nodes) > 0 else None  # type: ignore
        edges = create_table(edges, edge_specification) if len(edges) > 0 else None  # type: ignore
        offsets = create_table(offsets, offset_specification) if len(offsets) > 0 else None
        edges.drop_duplicates("id", inplace=True)  # type: ignore

    return nodes, edges, offsets  # type: ignore
