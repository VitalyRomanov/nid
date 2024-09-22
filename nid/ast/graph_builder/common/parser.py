import ast
from enum import Enum
import logging
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Iterable, Optional, Tuple, Type, List, Union, Dict

from nid.ast.graph_builder.common.definitions import PythonNodeEdgeDefinitions, GraphNodeId, EdgeImage, NodeImage
from nid.ast.graph_builder.common.graph import ParsedGraph
from nid.ast.graph_builder.common.identifiers import IdentifierPool
from nid.ast.graph_builder.common.inspection import get_available_fields_for_ast_node
from nid.ast.string_tools import to_offsets, get_cum_lens, get_byte_to_char_map


class AbstractGraphParser(ABC):
    @abstractmethod
    def parse(self, source_code: str) -> ParsedGraph:
        raise NotImplementedError()


class GraphParser(AbstractGraphParser):
    _original_source: Optional[str]
    _source_lines: Optional[List[str]]
    _root: Optional[ast.AST]
    _cum_lens: Optional[List[int]]
    _byte2char: Optional[Dict[int, int]]

    _node_types: Type[Enum]
    _edge_types: Type[Enum]
    _node_pool: Optional[Dict[GraphNodeId, Any]]
    _identifier_pool: IdentifierPool  # TODO should make Optional?

    _current_condition: List[GraphNodeId]
    _condition_status: List[Union[str, Enum]]
    _scope: List[GraphNodeId]

    _edges: List[EdgeImage]

    def __init__(self, graph_definitions: Type[PythonNodeEdgeDefinitions], **kwargs):
        self._graph_definitions = graph_definitions
        self._node_types = graph_definitions.make_node_type_enum()
        self._edge_types = graph_definitions.make_edge_type_enum()
        self._node_pool = dict()
        self._identifier_pool = IdentifierPool()

        self._current_condition = []
        self._condition_status = []
        self._scope = []

    @property
    def _latest_scope(self) -> Optional[GraphNodeId]:
        if len(self._scope) > 0:
            return self._scope[-1]
        else:
            return None

    @property
    def _latest_scope_name(self) -> Optional[str]:
        assert self._node_pool is not None

        if len(self._scope) > 0:
            scope = self._node_pool[self._scope[-1]]
            return scope["name"]
        else:
            return None

    def _make_edge(
            self, src: GraphNodeId, dst: GraphNodeId, type: Union[Enum, str], scope: Optional[GraphNodeId], **kwargs
    ) -> EdgeImage:
        edge = {
            "src": src,
            "dst": dst,
            "type": type,
            "scope": scope
        }
        edge.update(kwargs)
        return edge

    def _make_node(self, name: str, type: Union[Enum, str], scope: Optional[GraphNodeId], **kwargs) -> NodeImage:
        node = {
            "name": name,
            "type": type,
            "scope": scope,
        }
        node.update(kwargs)
        return node

    def _get_source_from_range(self, start: int, end: int) -> str:
        assert self._original_source is not None, "Parser was not initialized with source code"
        return self._original_source[start: end]

    def _parse_body(self, nodes: Iterable[ast.AST]) -> List[EdgeImage]:
        assert self._node_pool is not None, "Node pool is not initialized"

        edges: List[EdgeImage] = []

        last_node = None
        for node in nodes:
            edges_, root_id = self._parse_node(node)
            if self._node_pool[root_id]["type"] in (
                    self._node_types["Constant"],
            ):
                # this happens when processing docstring, as a result a lot of nodes are connected to the node const
                continue  # in general, constant node has no affect as a body expression, can skip
            edges.extend(edges_)

            if last_node is not None:
                # add connection to capture the expression order
                self._add_edge(edges, src=last_node, dst=root_id, type=self._edge_types["next"],
                               scope=self._latest_scope)

            last_node = root_id

            for cond_name, cond_stat in zip(self._current_condition[-1:], self._condition_status[-1:]):
                self._add_edge(
                    edges, src=last_node, dst=cond_name, type=cond_stat, scope=self._latest_scope)  # "defined_in_" +
        return edges

    def _handle_span_exceptions(
            self, node: Union[ast.AST, str], positions: Dict[str, int], return_ast_positions: bool = False
    ) -> Union[Optional[Tuple[int, int]], Tuple[Optional[Tuple[int, int]], Dict[str, int]]]:
        """
        For some of ast nodes, adjust their spans so that the span references the keywords or 
        key token instead

        :param node: ast node or string representation
        :param positions: dictionary containing line and column offsets abtained from ast node
        :param return_ast_positions: whether to return the dictionary with the adjusted ast node positions
        :return: adjusted span or (adjusted span, adjusted ast node positions)
        """
        assert self._original_source is not None
        assert self._source_lines is not None

        line = positions["line"]
        end_line = positions["end_line"]
        col_offset = positions["col_offset"]
        end_col_offset = positions["end_col_offset"]

        # this defines how to handle the majority of remapping
        offset_reduction_spec: Dict[Type[ast.AST], Optional[Tuple[int, str]]] = {
            ast.ExceptHandler: (6, "except"),
            ast.Try: (3, "try"),
            ast.For: (3, "for"),
            ast.AsyncFor: (9, "async for"),
            ast.While: (5, "while"),
            ast.With: (4, "with"),
            ast.AsyncWith: (10, "async with"),
            ast.FunctionDef: (3, "def"),
            ast.AsyncFunctionDef: (9, "async def"),
            ast.ClassDef: (5, "class"),
            ast.Import: (6, "import"),
            ast.Delete: (3, "del"),
            ast.ImportFrom: (4, "from"),
            ast.List: (1, "["),
            ast.Dict: (1, "{"),
            ast.Set: (1, "{"),
            ast.Tuple: None,  # possible variants: (1,2) and 1,2
            ast.ListComp: (1, "["),
            ast.DictComp: (1, "{"),
            ast.SetComp: (1, "{"),
            ast.GeneratorExp: None,  # cannot handle if passed as argument or function
            ast.Starred: (1, "*"),
            ast.Return: (6, "return"),
            ast.Global: (6, "global"),
            ast.Nonlocal: (8, "nonlocal"),
            ast.Assert: (6, "assert"),
            ast.Lambda: (6, "lambda"),
            ast.Raise: (5, "raise"),
            ast.Await: (5, "await"),
            ast.Yield: (5, "yield"),
            ast.YieldFrom: (10, "yield from"),
        }

        exception_handled = False
        expected_string = None
        skip_check = False
        node_type = type(node)
        if node_type in offset_reduction_spec:
            spec = offset_reduction_spec[node_type]
            if spec is not None:
                char_len, expected_string = offset_reduction_spec[node_type]  # type: ignore
                end_line = line
                end_col_offset = col_offset + char_len  # type: ignore
                exception_handled = True
        elif isinstance(node, ast.If):
            # special case for if statement
            end_line = line
            if self._source_lines[line][col_offset] == "i":
                # expression "if ...."
                end_col_offset = col_offset + 2
                expected_string = "if"
            elif self._source_lines[line][col_offset] == "e":
                # expression "elif ...."
                end_col_offset = col_offset + 4
                expected_string = "elif"
            else:
                assert False, "Unexpected character for if statement"
            exception_handled = True
        elif isinstance(node, ast.arg):
            # some issues when there is a type annotation and there is new
            # line after colon. example variable:\n type_ann
            skip_check = True
        elif node_type in {
            ast.Name,  # Should not even try since type annotation extraction depends on this
            ast.Attribute,  # Should not even try since type ann extraction depends on this
            ast.Constant,
            ast.JoinedStr,
            ast.Expr
        }:  # do not bother
            # TODO can probably add these in the dict above
            pass
        elif node_type in {
            ast.Compare,  # can use comparator operator
            ast.BoolOp,  # could be multiline
            ast.BinOp,  # could be multiline
            ast.Assign,  # could be multiline
            ast.AnnAssign,  # could be multiline
            ast.AugAssign,  # could be multiline
            ast.Subscript,  # could be multiline
            ast.Call,  # need to parse
            ast.UnaryOp,  # need to parse
            ast.IfExp,  # need to parse
            ast.Pass,  # seem to be fine
            ast.Break,  # seem to be fine
            ast.Continue,  # seem to be fine
        }:  # potential
            # extend functionality in the future
            pass
        # else:
        #     assert False

        positions_ = positions
        positions = {
            "line": line,
            "end_line": end_line,
            "col_offset": col_offset,
            "end_col_offset": end_col_offset
        }
        positions_offset = self._into_offset(positions)
        assert positions_offset is not None, "Could not calculate offset"

        if skip_check is False:
            if exception_handled is False:
                try:
                    ast.parse(self._original_source[positions_offset[0]: positions_offset[1]])
                except SyntaxError:
                    try:
                        ast.parse("(" + self._original_source[positions_offset[0]: positions_offset[1]] + ")")
                    except SyntaxError:
                        logging.warning(
                            f"Problems with parsing range {positions_offset} "
                            f"on line {self._source_lines[positions['line']]}"
                        )
                        positions_offset = self._into_offset(positions_)
                        # raise Exception("Range parsed incorrectly")
            else:
                assert (
                    expected_string == self._original_source[positions_offset[0]: positions_offset[1]]
                ), f"{expected_string} != {self._original_source[positions_offset[0]: positions_offset[1]]}"

        if return_ast_positions:
            return positions_offset, positions
        return positions_offset

    def _into_offset(self, range: Union[Tuple[int, int, int, int], Dict[str, int]]) -> Optional[Tuple[int, int]]:
        if isinstance(range, dict):
            range = (range["line"], range["end_line"], range["col_offset"], range["end_col_offset"])

        assert len(range) == 4

        try:
            assert self._original_source is not None, "Parser was not initialized"
            return to_offsets(
                self._original_source, [(*range, None)], cum_lens=self._cum_lens, b2c=self._byte2char, as_bytes=True
            )[-1][:2]
        except:
            return None

    def _get_positions_from_node(self, node: Optional[Union[ast.AST, str]], full: bool = False) -> Optional[Tuple[int, int]]:
        return_positions = None
        if node is not None and hasattr(node, "lineno"):
            # it is sufficient to check whether the node has a lineno attribute
            positions: Dict[str, int] = {
                "line": node.lineno - 1,  # 1-based line number  # type: ignore
                "end_line": node.end_lineno - 1,  # 1-based line number  # type: ignore
                "col_offset": node.col_offset,  # type: ignore
                "end_col_offset": node.end_col_offset  # type: ignore
            }
            positions_ = self._into_offset(positions)
            if full is False:  # TODO a better name for variable `full`
                positions_ = self._handle_span_exceptions(node, positions)
            return_positions = positions_
        return return_positions

    def _add_edge(
            self, edges, src: GraphNodeId, dst: GraphNodeId, type, scope: Optional[GraphNodeId] = None,
            position_node: Optional[ast.AST] = None, var_position_node: Optional[ast.AST] = None, 
            position: Optional[Tuple[int, int]] = None
    ):
        new_edge = self._make_edge(src=src, dst=dst, type=type, scope=scope)

        edge_pos = self._get_positions_from_node(position_node)
        if edge_pos is not None:
            new_edge.update({
                "offset_start": edge_pos[0],
                "offset_end": edge_pos[1]
            })

        edge_var_pos = self._get_positions_from_node(var_position_node)
        if edge_var_pos is not None:
            new_edge.update({
                "var_offset_start": edge_var_pos[0],
                "var_offset_end": edge_var_pos[1]
            })

        if position is not None:
            assert position_node is None, "position conflict"
            # TODO override position?
            new_edge.update({
                "offset_start": position[0],
                "offset_end": position[1]
            })

        edges.append(new_edge)

        # TODO should this be kept here?
        # if self._add_reverse_edges is True:
        #     reverse = new_edge.make_reverse()
        #     if reverse is not None:
        #         edges.append(reverse)

    def _parse_operand(self, node: Union[ast.AST, str]) -> Tuple[GraphNodeId, List[EdgeImage]]:
        assert self._node_pool is not None
        # need to make sure upper level name is correct; handle @keyword; type placeholder for sourcetrail nodes?
        edges: List[EdgeImage] = []
        if isinstance(node, str):
            # fall here when parsing attributes, they are given as strings; should attributes be parsed into subwords?
            if "@" in node:  # TODO should replace this with some proper type
                node_name, node_type = node.split("@")
                node = self._get_node(name=node_name, type=self._node_types[node_type])
            else:
                node = self._get_node(name=node, type=type(node).__name__)
                # node = ast.Name(node)
                # edges_, node = self._parse_node(node)
                # edges.extend(edges_)
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = self._get_node(name=str(node), type=type(node).__name__)
        else:
            iter_e = self._parse_node(node)
            # if type(iter_e) == str:
            #     iter_ = iter_e
            # elif isinstance(iter_e, int):
            #     iter_ = iter_e
            # elif
            if type(iter_e) == tuple:
                ext_edges, name = iter_e
                assert isinstance(name, str) and name in self._node_pool
                edges.extend(ext_edges)
                iter_ = name
            else:
                # unexpected scenario
                raise Exception()

        return iter_, edges

    def _parse_and_add_operand(self, node_name, operand, type, edges):

        operand_name, ext_edges = self._parse_operand(operand)
        edges.extend(ext_edges)

        if not isinstance(type, self._edge_types):
            type = self._edge_types[type]

        self._add_edge(edges, src=operand_name, dst=node_name, type=type, scope=self._latest_scope,
                       position_node=operand)

    def _parse_in_context(self, cond_name: Union[str, List[str]], cond_stat: Union[str, Enum, List[Union[str, Enum]]], edges: List, body: Iterable[ast.AST]) -> None:
        if not isinstance(cond_name, list):
            cond_name = [cond_name]
            cond_stat = [cond_stat]  # type: ignore

        for cn, cs in zip(cond_name, cond_stat):  # type: ignore
            self._current_condition.append(cn)
            self._condition_status.append(cs)

        edges.extend(self._parse_body(body))

        for i in range(len(cond_name)):
            self._current_condition.pop(-1)
            self._condition_status.pop(-1)

    def _get_node(
            self, *, node: Optional[Union[ast.AST, str]] = None, name: Optional[str] = None, type: Optional[Union[Enum, str]] = None,
            positions: Optional[Tuple[int, int]] = None, scope: Optional[GraphNodeId] = None, add_random_identifier: bool = False,
            node_string: Optional[str] = None
    ) -> GraphNodeId:
        assert self._node_pool is not None, "Node pool not initialized"

        random_identifier: GraphNodeId = self._identifier_pool.get_new_identifier()

        if name is not None:
            assert name is not None and type is not None
            if add_random_identifier:
                name = f"{name}_{random_identifier}"
        else:
            assert node is not None
            name = f"{node.__class__.__name__}_{random_identifier}"
            type = self._node_types[node.__class__.__name__]

        if positions is None:
            positions = self._get_positions_from_node(node)

        if positions is not None:
            offset_start = positions[0]
            offset_end = positions[1]
        else:
            offset_start = None
            offset_end = None

        new_node = self._make_node(
            name=name, type=type, scope=scope, string=node_string,
            offset_start=offset_start, offset_end=offset_end
        )
        new_node_id = str(len(self._node_pool))
        self._node_pool[new_node_id] = new_node
        return new_node_id

    def _generic_parse(
            self, node: ast.AST, operands: Iterable[str], with_name: Optional[GraphNodeId] = None, **kwargs
        ) -> Tuple[List[Any], GraphNodeId]:
        edges: List[EdgeImage] = []

        if with_name is None:
            node_name = self._get_node(node=node)
        else:
            node_name = with_name

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"] and isinstance(node.__getattribute__(operand), Iterable) :
                self._parse_in_context(node_name, operand, edges, node.__getattribute__(operand))
            else:
                operand_: Optional[Union[ast.AST, Iterable[ast.AST]]] = node.__getattribute__(operand)
                if operand_ is not None:
                    if isinstance(operand_, Iterable) and not isinstance(operand_, str):
                        # TODO:
                        #  appears as leaf node if the iterable is empty. suggest adding an "EMPTY" element
                        for oper_ in operand_:
                            self._parse_and_add_operand(node_name, oper_, operand, edges)
                    else:
                        self._parse_and_add_operand(node_name, operand_, operand, edges)

        return edges, node_name

    def _parse_node(self, node: ast.AST) -> Tuple[List[EdgeImage], GraphNodeId]:
        return self._generic_parse(node, get_available_fields_for_ast_node(node.__name__))

    def _normalize_edges(self, edges: Iterable[EdgeImage]) -> List[EdgeImage]:
        """
        Convert all edge types to strings, assign edge ids.
        """
        edges_: List[EdgeImage] = list()
        for ind, edge in enumerate(edges):
            edges_.append(copy(edge))
            if not isinstance(edge["type"], str):
                edges_[-1]["type"] = edges_[-1]["type"].name
            edges_[-1]["edge_hash"] = str(ind)
        return edges_

    def _normalize_nodes(self, nodes: Dict[GraphNodeId, Any]) -> List[NodeImage]:
        """
        Convert all node types to strings, assign node ids.
        """
        nodes_: List[NodeImage] = []
        for id_, node in nodes.items():
            nodes_.append(copy(node))
            if not isinstance(node["type"], str):
                nodes_[-1]["type"] = nodes_[-1]["type"].name
            nodes_[-1]["node_hash"] = str(id_)
            nodes_[-1]["string"] = nodes_[-1].get("string")
        return nodes_

    def _initialize_state(self, source: str):
        self._edges = []
        self._node_pool = dict()
        self._original_source = source
        self._source_lines = source.split("\n")
        self._root = ast.parse(source)
        self._cum_lens = get_cum_lens(self._original_source, as_bytes=True)
        self._byte2char = get_byte_to_char_map(self._original_source)

    def parse(self, source: str) -> ParsedGraph:
        self._initialize_state(source)

        self._edges = self._normalize_edges(self._parse_node(self._root)[0])  # type: ignore
        graph = ParsedGraph(source, self._normalize_nodes(self._node_pool), self._edges)  # type: ignore
        return graph


if __name__ == "__main__":
    parser = GraphParser(PythonNodeEdgeDefinitions)
    from nid.validation.ast_node_examples import PythonCodeExamplesForNodes
    for example, code in PythonCodeExamplesForNodes.examples.items():
        parser.parse(code).as_df()
    # parser.parse(PythonCodeExamplesForNodes.examples["FunctionDef2"])