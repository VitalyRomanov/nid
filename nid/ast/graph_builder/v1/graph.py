from typing import List

import pandas as pd

from nid.ast.graph_builder.common.graph import ParsedGraph
from nid.ast.annotator_utils import adjust_offsets2
from nid.ast.graph_builder import GraphParser
from nid.ast.graph_builder.common.definitions import EdgeImage
from nid.ast.graph_builder.common.utils import has_valid_syntax
from nid.ast.graph_builder.v1.node_resolver import NodeIdResolver
from nid.ast.string_tools import get_cum_lens, get_byte_to_char_map, to_offsets, string_hash


class GraphFormatter(GraphParser):
    def get_edges(self):
        edges = []
        all_edges, top_node_name = self._parse_node(self._root)
        edges.extend(all_edges)

        cum_lens = get_cum_lens(self._original_source, as_bytes=True)
        byte2char = get_byte_to_char_map(self._original_source)

        def format_offsets(edge):
            def into_offset(range):
                try:
                    return to_offsets(
                        self._original_source, [(*range, None)],  # type: ignore
                        cum_lens=cum_lens, b2c=byte2char, as_bytes=True
                    )[-1][:2]
                except:
                    return None

            if "line" in edge:
                edge["offsets"] = into_offset(
                    (edge["line"], edge["end_line"], edge["col_offset"], edge["end_col_offset"])
                )
                edge.pop("line")
                edge.pop("end_line")
                edge.pop("col_offset")
                edge.pop("end_col_offset")
            else:
                edge["offsets"] = None
            if "var_line" in edge:
                edge["var_offsets"] = into_offset(
                    (edge["var_line"], edge["var_end_line"], edge["var_col_offset"], edge["var_end_col_offset"])
                )
                edge.pop("var_line")
                edge.pop("var_end_line")
                edge.pop("var_col_offset")
                edge.pop("var_end_col_offset")

        for edge in edges:
            format_offsets(edge)

        return edges

    def _standardize_new_edges(
            self, edges: List[EdgeImage], node_resolver: NodeIdResolver,
            # mention_tokenizer: MentionTokenizer
    ) -> List[EdgeImage]:
        """
        Tokenize relevant node names, assign id to every node, collapse edge representation to id-based
        :param edges: list of edges
        :param node_resolver: helper class that tracks node ids
        :return:
        """
        # :param mention_tokenizer: helper class that performs tokenization of relevant nodes

        # edges = mention_tokenizer.replace_mentions_with_subwords(edges)

        resolve_node_id = lambda node: node_resolver.resolve_node_id(node)
        extract_id = lambda node: node if isinstance(node, str) else node.id

        for edge in edges:
            edge["src"] = resolve_node_id(edge["src"])
            edge["dst"] = resolve_node_id(edge["dst"])
            if "scope" in edge:
                edge["scope"] = resolve_node_id(edge["scope"])

        for edge in edges:
            edge["src"] = extract_id(edge["src"])
            edge["dst"] = extract_id(edge["dst"])
            if "scope" in edge:
                edge["scope"] = extract_id(edge["scope"])
            else:
                edge["scope"] = pd.NA
            edge["file_id"] = pd.NA
            edge["edge_hash"] = string_hash(f"{edge['src']}_{edge['dst']}_{edge['type']}")

        return edges

    def _process_code_without_index(
            self, source: str,
            node_resolver: NodeIdResolver,
            # mention_tokenizer: MentionTokenizer,
            track_offsets: bool = False,
    ):
        self._initialize_state(source)
        edges = self.get_edges()

        if len(edges) == 0:
            return ParsedGraph(source, [], [])

        # tokenize names, replace nodes with their ids
        edges = self._standardize_new_edges(edges, node_resolver)

        if track_offsets:
            def get_valid_offsets(edges):
                """
                :param edges: Dictionary that represents edge. Information is tored in edges but is related to source node
                :return: Information about location of this edge (offset) in the source file in fromat (start, end, node_id)
                """
                return [
                    (edge["offsets"][0], edge["offsets"][1], (edge["src"], edge["dst"], edge["type"]), edge["scope"])
                    for edge in edges
                    if edge["offsets"] is not None]

            def get_node_offsets(offsets):
                return [(offset[0], offset[1], offset[2][0], offset[3]) for offset in offsets]

            def offsets_to_edge_mapping(offsets):
                return {offset[2]: (offset[0], offset[1]) for offset in offsets}

            def attach_offsets_to_edges(edges, offsets_edge_mapping):
                for edge in edges:
                    repr = (edge["src"], edge["dst"], edge["type"])
                    if repr in offsets_edge_mapping:
                        offset = offsets_edge_mapping[repr]
                        edge["offset_start"] = offset[0]
                        edge["offset_end"] = offset[1]

            # recover ast offsets for the current file
            valid_offsets = get_valid_offsets(edges)
            ast_offsets = get_node_offsets(valid_offsets)
            attach_offsets_to_edges(edges, offsets_to_edge_mapping(valid_offsets))
        else:
            ast_offsets = None

        return edges, ast_offsets

    def parse(self, source_code: str):
        node_resolver = NodeIdResolver()
        # mention_tokenizer = MentionTokenizer(bpe_tokenizer_path, create_subword_instances, connect_subwords)
        all_ast_edges = []
        all_offsets = []

        source_code_ = source_code.lstrip()
        initial_strip = source_code[:len(source_code) - len(source_code_)]

        if not has_valid_syntax(source_code):
            raise SyntaxError()

        edges, ast_offsets = self._process_code_without_index(
            source_code, node_resolver, # mention_tokenizer,
            track_offsets=True
        )

        if ast_offsets is not None:
            adjust_offsets2(ast_offsets, len(initial_strip))

        if edges is None:
            raise ValueError("No graph can be generated from the source code")

        #### afterprocessing

        # for edge in edges:
        #     edge["file_id"] = source_code_id

        #### finish afterprocessing

        all_ast_edges.extend(edges)

        def format_offsets(ast_offsets, target):
            """
            Format offset as a record and add to the common storage for offsets
            :param ast_offsets:
            :param target: List where all other offsets are stored.
            :return: Nothing
            """
            if ast_offsets is not None:
                for offset in ast_offsets:
                    target.append({
                        "file_id": 0,  # source_code_id,
                        "start": offset[0],
                        "end": offset[1],
                        "node_id": offset[2],
                        "scope": offset[3],
                        "string": source_code[offset[0]: offset[1]],
                        "package": "0",  # package
                    })

        format_offsets(ast_offsets, target=all_offsets)

        node_resolver.stash_new_nodes()

        all_ast_nodes = node_resolver.new_nodes_for_write(from_stashed=True)

        if all_ast_nodes is None:
            return None, None, None

        def prepare_edges(all_ast_edges):
            all_ast_edges = pd.DataFrame(all_ast_edges)
            all_ast_edges.drop_duplicates(["type", "src", "dst"], inplace=True)
            all_ast_edges = all_ast_edges.query("src != dst")
            all_ast_edges["id"] = all_ast_edges["edge_hash"]

            column_order = ["id", "type", "src", "dst", "file_id", "scope"]
            if "offset_start" in all_ast_edges.columns:
                column_order.append("offset_start")
                column_order.append("offset_end")

            all_ast_edges = all_ast_edges[column_order] \
                .rename({'src': 'src', 'dst': 'dst', 'scope': 'scope'}, axis=1) \
                .astype({'file_id': 'Int32', "scope": 'string'})

            # all_ast_edges["id"] = range(len(all_ast_edges))
            all_ast_edges["edge_hash"] = all_ast_edges["id"]
            return all_ast_edges

        all_ast_edges = prepare_edges(all_ast_edges)

        if len(all_offsets) > 0:
            all_offsets = pd.DataFrame(all_offsets)
        else:
            all_offsets = None

        node2id = dict(zip(all_ast_nodes["id"], range(len(all_ast_nodes))))

        def map_columns_to_int(table, dense_columns, sparse_columns):
            types = {column: "int64" for column in dense_columns}
            types.update({column: "Int64" for column in sparse_columns})

            for column, dtype in types.items():
                table[column] = table[column].apply(node2id.get).astype(dtype)

        # map_columns_to_int(all_ast_nodes, dense_columns=["id"], sparse_columns=["scope"])
        # map_columns_to_int(
        #     all_ast_edges,
        #     dense_columns=["src", "dst"],
        #     sparse_columns=["scope"]
        # )
        # if all_offsets is not None:
        #     map_columns_to_int(all_offsets, dense_columns=["node_id"], sparse_columns=["scope"])

        return ParsedGraph(source_code, all_ast_nodes.to_dict("records"), all_ast_edges.to_dict("records"))
