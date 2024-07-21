import ast
import logging
from pprint import pprint
from collections.abc import Iterable
from typing import Optional, Type, Dict

import pandas as pd

from nid.ast.graph_builder.common.parser import GraphParser, GraphNodeId
from nid.ast.graph_builder.v3.dataframe import nodes_edges_to_df
from nid.ast.graph_builder.v3.definitions import PythonNodeEdgeDefinitions
from nid.ast.graph_builder.v3.primitives import GraphEdge, GraphNode
from nid.ast.string_tools import get_cum_lens, get_byte_to_char_map, to_offsets
from nid.validation.ast_node_examples import PythonCodeExamplesForNodes


class GraphParserV3(GraphParser):
    _node_pool: Dict[GraphNodeId, GraphNode]

    def __init__(
            self, graph_definitions: Type[PythonNodeEdgeDefinitions],
            add_reverse_edges: bool = True, save_node_strings: bool = True,
            add_mention_instances: bool = False, parse_constants: bool = False,
            # parse_ctx=False,
            **kwargs
    ):
        super().__init__(graph_definitions)
        self._graph_definitions = graph_definitions
        self._add_reverse_edges = add_reverse_edges
        self._add_mention_instances = add_mention_instances
        self._parse_constants = parse_constants
        # self._parse_ctx = parse_ctx
        self._save_node_strings = save_node_strings
        self._set_node_class()
        self._set_edge_class()

    def _set_node_class(self):
        self._make_node = GraphNode  # type: ignore

    def _set_edge_class(builderself):  # type: ignore
        class _GraphEdge(GraphEdge):
            def make_reverse(self, *args, **kwargs):
                assert builderself._node_pool is not None
                reverse_type = builderself._graph_definitions.get_reverse_type(self.type.name)
                src_node = builderself._node_pool[self.src]
                if reverse_type is not None and not builderself._graph_definitions.is_shared_name_type(src_node.name,
                                                                                                   src_node.type):
                    return self.__class__(src=self.dst, dst=self.src, type=reverse_type, scope=self.scope)
                else:
                    return None

        builderself._make_edge = _GraphEdge  # type: ignore

    @property
    def _latest_scope_name(self):
        if len(self._scope) > 0:
            scope = self._node_pool[self._scope[-1]]
            return scope.name
        else:
            return None

    def _into_offset(self, range):
        if isinstance(range, dict):
            range = (range["line"], range["end_line"], range["col_offset"], range["end_col_offset"])

        assert len(range) == 4

        try:
            return to_offsets(
                self._original_source, [(*range, None)], cum_lens=self._cum_lens, b2c=self._byte2char, as_bytes=True
            )[-1][:2]
        except:
            return None

    def _handle_span_exceptions(self, node, positions):
        line = positions["line"]
        end_line = positions["end_line"]
        col_offset = positions["col_offset"]
        end_col_offset = positions["end_col_offset"]

        offset_reduction_spec = {
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
            ast.GeneratorExp: None,  # cannot if passed as argument ot function
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
                char_len, expected_string = offset_reduction_spec[node_type]
                end_line = line
                end_col_offset = col_offset + char_len
                exception_handled = True
        elif isinstance(node, ast.If):
            end_line = line
            if self._source_lines[line][col_offset] == "i":
                end_col_offset = col_offset + 2
                expected_string = "if"
            elif self._source_lines[line][col_offset] == "e":
                end_col_offset = col_offset + 4
                expected_string = "elif"
            else:
                assert False
            exception_handled = True
        elif isinstance(node, ast.arg):
            # some issues when there is a type annotation and there is new
            # line after colon. example variable:\n type_ann
            skip_check = True
        elif type(node) in {
            ast.Name,  # Should not even try since type annotation extraction depends on this
            ast.Attribute,  # Should not even try since type ann extraction depends on this
            ast.Constant,
            ast.JoinedStr,
            ast.Expr
        }:  # do not bother
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
            pass
        # else:
        #     assert False

        positions = {
            "line": line,
            "end_line": end_line,
            "col_offset": col_offset,
            "end_col_offset": end_col_offset
        }
        positions = self._into_offset(positions)

        if skip_check is False:
            if exception_handled is False:
                try:
                    ast.parse(self._original_source[positions[0]: positions[1]])
                except SyntaxError:
                    try:
                        ast.parse("(" + self._original_source[positions[0]: positions[1]] + ")")
                    except:
                        raise Exception("Range parsed incorrectly")
            else:
                assert (
                    expected_string == self._original_source[positions[0]: positions[1]]
                ), f"{expected_string} != {self._original_source[positions[0]: positions[1]]}"

        return positions

    def _get_positions_from_node(self, node, full=False):
        if node is not None and hasattr(node, "lineno"):
            positions = {
                "line": node.lineno - 1,
                "end_line": node.end_lineno - 1,
                "col_offset": node.col_offset,
                "end_col_offset": node.end_col_offset
            }
            positions_ = self._into_offset(positions)
            if full is False:
                positions_ = self._handle_span_exceptions(node, positions)
            positions = positions_
        else:
            positions = None
        return positions

    def _get_source_from_range(self, start, end):
        return self._original_source[start: end]

    def _get_node(
            self, *, node=None, name: Optional[str] = None, type=None,
            positions=None, scope=None, add_random_identifier: bool = False,
            node_string=None
    ) -> str:

        random_identifier = self._identifier_pool.get_new_identifier()

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

        if self._save_node_strings:
            if node_string is None:
                node_string = self._get_source_from_range(*positions) if positions is not None else None
            else:
                assert isinstance(node_string, str)

        if scope is None and self._graph_definitions.is_shared_name_type(name, type) is False:
            scope = self.latest_scope

        new_node = self._make_node(
            name=name, type=type, scope=scope, string=node_string,  # positions=positions,
            offset_start=offset_start, offset_end=offset_end
        )
        self._node_pool[new_node.hash_id] = new_node
        return new_node.hash_id

    def _parse(self, node):
        n_type = type(node).__name__
        if n_type in self._graph_definitions.ast_node_type_edges:
            return self.generic_parse(
                node,
                self._graph_definitions.ast_node_type_edges[n_type]
            )
        elif n_type in self._graph_definitions.overridden_node_type_edges:
            method_name = "parse_" + n_type
            return self.__getattribute__(method_name)(node)
        elif n_type in self._graph_definitions.iterable_nodes:
            return self.parse_iterable(node)
        elif n_type in self._graph_definitions.named_nodes:
            return self.parse_name(node)
        elif n_type in self._graph_definitions.constant_nodes:
            return self.parse_Constant(node)
        elif n_type in self._graph_definitions.operand_nodes:
            return self.parse_op_name(node)
        elif n_type in self._graph_definitions.control_flow_nodes:
            return self.parse_control_flow(node)
        elif n_type in self._graph_definitions.ctx_nodes:
            return self.parse_ctx(node)
        else:
            logging.error(f"Failed parsing:")
            logging.error(f"{type(node)}")
            logging.error(f"{ast.dump(node)}")
            logging.error(f"{node._fields}")
            pprint(self._source_lines)
            return self.generic_parse(node, node._fields)

    def _add_edge(
            self, edges, src: str, dst: str, type, scope: Optional[str] = None,
            position_node=None, var_position_node=None, position=None
    ):
        new_edge = self._make_edge(src=src, dst=dst, type=type, scope=scope)
        new_edge.assign_positions(self._get_positions_from_node(position_node))
        new_edge.assign_positions(self._get_positions_from_node(var_position_node), prefix="var")
        if position is not None:
            assert position_node is None, "position conflict"
            new_edge.assign_positions(position)

        if (
                (
                    self._node_pool[new_edge.src].type.name == "instance" and
                    self._node_pool[new_edge.dst].type.name not in {
                        "FunctionDef", "AsyncFunctionDef", "Global", "Nonlocal", "ImportFrom", "Import", "alias",
                    }  # no position information
                ) or (
                    self._node_pool[new_edge.src].type.name == "mention" and
                    self._node_pool[new_edge.dst].type.name not in {
                        "instance", "FunctionDef", "AsyncFunctionDef", "Global", "Nonlocal", "ImportFrom", "Import",
                        "alias",
                    }
                )
        ):
            if new_edge.offset_start is None:
                assert False, "no offset for crucial node"

        edges.append(new_edge)

        if self._add_reverse_edges is True:
            reverse = new_edge.make_reverse()
            if reverse is not None:
                edges.append(reverse)

    def parse_body(self, nodes):
        edges = []
        last_node = None
        for node in nodes:
            s = self._parse(node)
            if isinstance(s, tuple):
                if self._node_pool[s[1]].type in (self._node_types["Constant"], self._node_types["mention"], self._node_types["instance"]):
                    # this happens when processing docstring, as a result a lot of nodes are connected to the node
                    continue  # in general, constant node has no affect as a body expression, can skip
                # some parsers return edges and names?
                edges.extend(s[0])

                if last_node is not None:
                    self._add_edge(edges, src=last_node, dst=s[1], type=self._edge_types["next"],
                                   scope=self.latest_scope)

                last_node = s[1]

                for cond_name, cond_stat in zip(self._current_condition[-1:], self._condition_status[-1:]):
                    self._add_edge(edges, src=last_node, dst=cond_name, type=cond_stat,
                                   scope=self.latest_scope)  # "defined_in_" +
            else:
                edges.extend(s)
        return edges

    def parse_in_context(self, cond_name, cond_stat, edges, body):
        if not isinstance(cond_name, list):
            cond_name = [cond_name]
            cond_stat = [cond_stat]

        for cn, cs in zip(cond_name, cond_stat):
            self._current_condition.append(cn)
            self._condition_status.append(cs)

        edges.extend(self.parse_body(body))

        for i in range(len(cond_name)):
            self._current_condition.pop(-1)
            self._condition_status.pop(-1)

    def parse_as_mention(self, name, ctx=None):
        mention_name = self._get_node(name=name + "@" + self._latest_scope_name, type=self._node_types["mention"])
        name_ = self._get_node(name=name, type=self._node_types["Name"])

        edges = []
        self._add_edge(edges, src=name_, dst=mention_name, type=self._edge_types["local_mention"],
                       scope=self.latest_scope)

        if self._add_mention_instances:
            mention_instance = self._get_node(
                name="instance", type=self._node_types["instance"], add_random_identifier=True, node_string=name
            )
            self._node_pool[mention_instance].string = name
            self._add_edge(
                edges, src=mention_name, dst=mention_instance, type=self._edge_types["instance"],
                scope=self.latest_scope
            )
            mention_name = mention_instance

            if ctx is not None:
                ctx_node = self._parse(ctx)
                self._add_edge(
                    edges, src=ctx_node, dst=mention_instance, type=self._edge_types["ctx"],
                    scope=self.latest_scope
                )
        return edges, mention_name

    def parse_operand(self, node):
        # need to make sure upper level name is correct; handle @keyword; type placeholder for sourcetrail nodes?
        edges = []
        if isinstance(node, str):
            # fall here when parsing attributes, they are given as strings; should attributes be parsed into subwords?
            if "@" in node:
                node_name, node_type = node.split("@")
                node = self._get_node(name=node_name, type=self._node_types[node_type])
            else:
                node = ast.Name(node)
                edges_, node = self._parse(node)
                edges.extend(edges_)
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = self._get_node(name=str(node), type=self._node_types["astliteral"])
        else:
            iter_e = self._parse(node)
            if type(iter_e) == str:
                iter_ = iter_e
            elif isinstance(iter_e, int):
                iter_ = iter_e
            elif type(iter_e) == tuple:
                ext_edges, name = iter_e
                assert isinstance(name, str) and name in self._node_pool
                edges.extend(ext_edges)
                iter_ = name
            else:
                # unexpected scenario
                print(node)
                print(ast.dump(node))
                print(iter_e)
                print(type(iter_e))
                pprint(self._source_lines)
                print(self._source_lines[node.lineno - 1].strip())
                raise Exception()

        return iter_, edges

    def parse_and_add_operand(self, node_name, operand, type, edges):

        operand_name, ext_edges = self.parse_operand(operand)
        edges.extend(ext_edges)

        if not isinstance(type, self._edge_types):
            type = self._edge_types[type]

        self._add_edge(edges, src=operand_name, dst=node_name, type=type, scope=self.latest_scope,
                       position_node=operand)

    def generic_parse(self, node, operands, with_name=None, ensure_iterables=False):

        edges = []

        if with_name is None:
            node_name = self._get_node(node=node)
        else:
            node_name = with_name

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"]:
                logging.warning(f"Not clear which node is processed here {ast.dump(node)}")
                self.parse_in_context(node_name, operand, edges, node.__getattribute__(operand))
            else:
                operand_ = node.__getattribute__(operand)
                if operand_ is not None:
                    if isinstance(operand_, Iterable) and not isinstance(operand_, str):
                        # TODO:
                        #  appears as leaf node if the iterable is empty. suggest adding an "EMPTY" element
                        for oper_ in operand_:
                            self.parse_and_add_operand(node_name, oper_, operand, edges)
                    else:
                        self.parse_and_add_operand(node_name, operand_, operand, edges)

        # TODO
        # need to identify the benefit of this node
        # maybe it is better to use node types in the graph
        # edges.append({"scope": copy(self._scope[-1]), "src": node.__class__.__name__, "dst": node_name, "type": "node_type"})

        return edges, node_name

    # def parse_type_node(self, node):
    #     if node.lineno == node.end_lineno:
    #         type_str = self._source_lines[node.lineno][node.col_offset - 1: node.end_col_offset]
    #     else:
    #         type_str = ""
    #         for ln in range(node.lineno - 1, node.end_lineno):
    #             if ln == node.lineno - 1:
    #                 type_str += self._source_lines[ln][node.col_offset - 1:].strip()
    #             elif ln == node.end_lineno - 1:
    #                 type_str += self._source_lines[ln][:node.end_col_offset].strip()
    #             else:
    #                 type_str += self._source_lines[ln].strip()
    #     return type_str

    def parse_Module(self, node):
        edges, module_name = self.generic_parse(node, [])
        self._scope.append(module_name)
        self.parse_in_context(module_name, self._edge_types["defined_in_module"], edges, node.body)
        self._scope.pop(-1)
        return edges, module_name

    def parse_FunctionDef(self, node):
        # need to create function name before generic_parse so that the scope is set up correctly
        # scope is used to create local mentions of variable and function names
        fdef_node = self._get_node(node=node)
        self._scope.append(fdef_node)

        to_parse = []
        if (
                len(node.args.posonlyargs) > 0 or
                len(node.args.args) > 0 or
                len(node.args.kwonlyargs) > 0 or
                node.args.vararg is not None or
                node.args.kwarg is not None
        ):
            to_parse.append("args")
        if len("decorator_list") > 0:
            to_parse.append("decorator_list")

        edges, f_name = self.generic_parse(node, to_parse, with_name=fdef_node)

        if node.returns is not None:
            # returns stores return type annotation
            # can contain quotes
            # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
            # https://www.python.org/dev/peps/pep-0484/#forward-references
            annotation_position = self._get_positions_from_node(node.returns, full=True)
            annotation_string = self._get_source_from_range(*annotation_position)
            annotation = self._get_node(
                name=annotation_string, type=self._node_types["type_annotation"]
            )
            self._add_edge(edges, src=annotation, dst=f_name, type=self._edge_types["returned_by"],
                           scope=self.latest_scope, position_node=node.returns)

        self.parse_in_context(f_name, self._edge_types["defined_in_function"], edges, node.body)

        self._scope.pop(-1)

        ext_edges, func_name = self.parse_as_mention(name=node.name)
        edges.extend(ext_edges)

        assert isinstance(node.name, str)
        self._add_edge(edges, src=func_name, dst=f_name, type=self._edge_types["function_name"],
                       scope=self.latest_scope)

        return edges, f_name

    def parse_AsyncFunctionDef(self, node):
        return self.parse_FunctionDef(node)

    def parse_ClassDef(self, node):

        edges, class_node_name = self.generic_parse(node, [])

        self._scope.append(class_node_name)
        self.parse_in_context(class_node_name, self._edge_types["defined_in_class"], edges, node.body)
        self._scope.pop(-1)

        ext_edges, cls_name = self.parse_as_mention(name=node.name)
        edges.extend(ext_edges)
        self._add_edge(edges, src=class_node_name, dst=cls_name, type=self._edge_types["class_name"],
                       scope=self.latest_scope)

        return edges, class_node_name

    def parse_With(self, node):
        edges, with_name = self.generic_parse(node, self._graph_definitions.overridden_node_type_edges["With"])

        self.parse_in_context(with_name, self._edge_types["executed_inside_with"], edges, node.body)

        return edges, with_name

    def parse_AsyncWith(self, node):
        return self.parse_With(node)

    def parse_arg(self, node, default_value=None):
        name = self._get_node(node=node)
        edges, mention_name = self.parse_as_mention(node.arg)
        self._add_edge(
            edges, src=mention_name, dst=name, type=self._edge_types["arg"], scope=self.latest_scope,
            position_node=node
        )

        if node.annotation is not None:
            # can contain quotes
            # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
            # https://www.python.org/dev/peps/pep-0484/#forward-references
            positions = self._get_positions_from_node(node.annotation, full=True)
            annotation_string = self._get_source_from_range(*positions)
            annotation = self._get_node(name=annotation_string, type=self._node_types["type_annotation"])
            mention_name = self._get_node(
                name=node.arg + "@" + self._latest_scope_name, type=self._node_types["mention"],
                scope=self.latest_scope
            )
            self._add_edge(edges, src=annotation, dst=mention_name, type=self._edge_types["annotation_for"],
                           scope=self.latest_scope, position_node=node.annotation, var_position_node=node)

        if default_value is not None:
            deflt_ = self._parse(default_value)
            if isinstance(deflt_, tuple):
                edges.extend(deflt_[0])
                default_val = deflt_[1]
            else:
                default_val = deflt_
            self._add_edge(edges, default_val, name, type=self._edge_types["default"], position_node=default_value,
                          scope=self.latest_scope)
        return edges, name

    def parse_AnnAssign(self, node):
        # can contain quotes
        # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
        # https://www.python.org/dev/peps/pep-0484/#forward-references
        positions = self._get_positions_from_node(node.annotation, full=True)
        annotation_string = self._get_source_from_range(*positions)
        annotation = self._get_node(
            name=annotation_string, type=self._node_types["type_annotation"]
        )
        edges, name = self.generic_parse(node, ["target", "value"])
        try:
            mention_name = self._get_node(
                name=node.target.id + "@" + self._latest_scope_name, type=self._node_types["mention"],
                scope=self.latest_scope
            )
        except Exception as e:
            mention_name = name

        self._add_edge(edges, src=annotation, dst=mention_name, type=self._edge_types["annotation_for"],
                       scope=self.latest_scope, position_node=node.annotation, var_position_node=node)
        return edges, name

    def parse_Lambda(self, node):
        # this is too ambiguous
        edges, lmb_name = self.generic_parse(node, [])
        self.parse_and_add_operand(lmb_name, node.body, self._edge_types["lambda"], edges)

        return edges, lmb_name

    def parse_IfExp(self, node):
        edges, ifexp_name = self.generic_parse(node, ["test"])
        self.parse_and_add_operand(ifexp_name, node.body, self._edge_types["if_true"], edges)
        self.parse_and_add_operand(ifexp_name, node.orelse, self._edge_types["if_false"], edges)
        return edges, ifexp_name

    def parse_ExceptHandler(self, node):
        # have missing fields. example:
        # not parsing "name" field
        # except handler is unique for every function
        return self.generic_parse(node, ["type"])

    def parse_keyword(self, node):
        if isinstance(node.arg, str):
            # change arg name so that it does not mix with variable names
            node.arg += "@#keyword#"
            return self.generic_parse(node, self._graph_definitions.overridden_node_type_edges["keyword"])
        else:
            return self.generic_parse(node, ["value"])

    def parse_name(self, node):
        if type(node) == ast.Name:
            return self.parse_as_mention(str(node.id), ctx=node.ctx if hasattr(node, "ctx") else None)
        elif type(node) == ast.NameConstant:
            return self._get_node(name=str(node.value), type=self._node_types["NameConstant"])

    def parse_Attribute(self, node):
        if node.attr is not None:
            # change attr name so that it does not mix with variable names
            node.attr += "@#attr#"
        return self.generic_parse(node, self._graph_definitions.overridden_node_type_edges["Attribute"])

    def parse_Constant(self, node):
        if self._parse_constants:
            name_ = str(node.value)
        else:
            value_type = type(node.value).__name__
            name_ = f"Constant_{value_type}"
        name = self._get_node(name=name_, type=self._node_types["Constant"])
        return name

    def parse_op_name(self, node):
        return self._get_node(name=node.__class__.__name__, type=self._node_types["Op"])

    def parse_Num(self, node):
        return str(node.n)

    def parse_Str(self, node):
        return self.generic_parse(node, [])

    def parse_Bytes(self, node):
        return repr(node.s)

    def parse_If(self, node):
        edges, if_name = self.generic_parse(node, ["test"])

        self.parse_in_context(if_name, self._edge_types["executed_if_true"], edges, node.body)
        self.parse_in_context(if_name, self._edge_types["executed_if_false"], edges, node.orelse)

        return edges, if_name

    def parse_For(self, node):

        edges, for_name = self.generic_parse(node, ["target", "iter"])

        self.parse_in_context(for_name, self._edge_types["executed_in_for"], edges, node.body)
        self.parse_in_context(for_name, self._edge_types["executed_in_for_orelse"], edges, node.orelse)

        return edges, for_name

    def parse_AsyncFor(self, node):
        return self.parse_For(node)

    def parse_Try(self, node):

        edges, try_name = self.generic_parse(node, [])

        self.parse_in_context(try_name, self._edge_types["executed_in_try"], edges, node.body)

        for h in node.handlers:
            handler_name, ext_edges = self.parse_operand(h)
            edges.extend(ext_edges)
            self.parse_in_context(
                [handler_name],  # [try_name, handler_name],
                [self._edge_types["executed_with_try_handler"]],
                # [self._edge_types["executed_in_try_except"], self._edge_types["executed_with_try_handler"]],
                edges, h.body
            )
            self._add_edge(edges, src=handler_name, dst=try_name, type=self._edge_types["executed_in_try_except"])

        self.parse_in_context(try_name, self._edge_types["executed_in_try_final"], edges, node.finalbody)
        self.parse_in_context(try_name, self._edge_types["executed_in_try_else"], edges, node.orelse)

        return edges, try_name

    def parse_While(self, node):

        edges, while_name = self.generic_parse(node, ["test"])

        # cond_name, ext_edges = self.parse_operand(node.test)
        # edges.extend(ext_edges)

        self.parse_in_context(
            [while_name],  # [while_name, cond_name],
            [self._edge_types["executed_in_while"]],
            edges, node.body
        )

        return edges, while_name

    def parse_Expr(self, node):
        edges = []
        expr_name, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)

        return edges, expr_name

    def parse_control_flow(self, node):
        edges = []
        ctrlflow_name = self._get_node(
            name="ctrl_flow", type=self._node_types["CtlFlowInstance"], node=node, add_random_identifier=True
        )
        self._add_edge(edges, src=self._get_node(name=node.__class__.__name__, type=self._node_types["CtlFlow"]),
                       dst=ctrlflow_name, type=self._edge_types["control_flow"], scope=self.latest_scope,
                       position_node=node)

        return edges, ctrlflow_name

    def parse_ctx(self, node):
        ctx_name = self._get_node(
            name=node.__class__.__name__, type=self._node_types["ctx"], node=node, scope=None
        )
        return ctx_name

    def parse_iterable(self, node):
        return self.generic_parse(node, ["elts", "ctx"], ensure_iterables=True)

    def parse_Dict(self, node):
        return self.generic_parse(node, ["keys", "values"], ensure_iterables=True)

    def parse_JoinedStr(self, node):
        joinedstr_name = self._get_node(
            name="JoinedStr_", type=self._node_types["JoinedStr"], node=node
        )
        return [], joinedstr_name

    def parse_FormattedValue(self, node):
        # have missing fields. example:
        # FormattedValue(value=Subscript(value=Name(id='args', ctx=Load()), slice=Index(value=Num(n=0)), ctx=Load()),conversion=-1, format_spec=None)
        # 'conversion', 'format_spec' not parsed
        return self.generic_parse(node, ["value"])

    def parse_arguments(self, node):
        edges, arguments = self.generic_parse(node, [])

        if node.vararg is not None:
            ext_edges_, vararg = self.parse_arg(node.vararg)
            edges.extend(ext_edges_)
            self._add_edge(edges, vararg, arguments, type=self._edge_types["vararg"],  # position_node=node.vararg,
                          scope=self.latest_scope)

        for i in range(len(node.posonlyargs)):
            ext_edges_, posarg = self.parse_arg(node.posonlyargs[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, posarg, arguments, type=self._edge_types["posonlyarg"],  # position_node=node.posonlyargs[i],
                          scope=self.latest_scope)

        without_default = len(node.args) - len(node.defaults)
        for i in range(without_default):
            ext_edges_, just_arg = self.parse_arg(node.args[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, just_arg, arguments, type=self._edge_types["arg"],  # position_node=node.args[i],
                          scope=self.latest_scope)

        for ind, i in enumerate(range(without_default, len(node.args))):
            ext_edges_, just_arg = self.parse_arg(node.args[i], default_value=node.defaults[ind])
            edges.extend(ext_edges_)
            self._add_edge(edges, just_arg, arguments, type=self._edge_types["arg"],  # position_node=node.args[i],
                          scope=self.latest_scope)

        for i in range(len(node.kwonlyargs)):
            ext_edges_, kw_arg = self.parse_arg(node.kwonlyargs[i], default_value=node.kw_defaults[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, kw_arg, arguments, type=self._edge_types["kwonlyarg"],  # position_node=node.kwonlyargs[i],
                          scope=self.latest_scope)

        if node.kwarg is not None:
            ext_edges_, kwarg = self.parse_arg(node.kwarg)
            edges.extend(ext_edges_)
            self._add_edge(edges, kwarg, arguments, type=self._edge_types["kwarg"],  # position_node=node.kwarg,
                           scope=self.latest_scope)

        return edges, arguments

        # vararg constains type annotations
        # return self.generic_parse(node, ["args", "vararg", "kwarg", "kwonlyargs", "posonlyargs"])

    def parse_comprehension(self, node):
        edges = []

        cph_name = self._get_node(
            name="comprehension", type=self._node_types["comprehension"], add_random_identifier=True
        )

        target, ext_edges = self.parse_operand(node.target)
        edges.extend(ext_edges)

        self._add_edge(edges, src=target, dst=cph_name, type=self._edge_types["target"], scope=self.latest_scope,
                       position_node=node.target)

        iter_, ext_edges = self.parse_operand(node.iter)
        edges.extend(ext_edges)

        self._add_edge(edges, src=iter_, dst=cph_name, type=self._edge_types["iter"], scope=self.latest_scope,
                       position_node=node.iter)

        for if_ in node.ifs:
            if_n, ext_edges = self.parse_operand(if_)
            edges.extend(ext_edges)
            self._add_edge(edges, src=if_n, dst=cph_name, type=self._edge_types["ifs"], scope=self.latest_scope,
                           position_node=if_)

        return edges, cph_name

    def postprocess(self):
        pass
        # if self._parse_ctx is False:
        #     ctx_edge_type = self._edge_types["ctx"]
        #     ctx_node_type = self._node_types["ctx"]
        #     self._edges = [edge for edge in self._edges if edge.type != ctx_edge_type]
        #     nodes_to_remove = [node.hash_id for node in self._node_pool.values() if node.type != ctx_node_type]
        #     for node_id in nodes_to_remove:
        #         self._node_pool.pop(node_id)

    def _get_offsets(self, edges):
        offsets = edges[["src", "offset_start", "offset_end", "scope"]] \
            .dropna() \
            .rename({
                "src": "node_id" #, "offset_start": "start", "offset_end": "end" #, "scope": "mentioned_in"
            }, axis=1)

        # assert len(offsets) == offsets["node_id"].nunique()  # there can be several offsets for constants

        return edges, offsets

    def _assign_node_strings(self, nodes, offsets):
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
            if id_ in start_map:
                node_string = self._original_source[start_map[id_]: end_map[id_]]
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

    def to_df(self, make_table=True):
        self.postprocess()
        nodes, edges, offsets = nodes_edges_to_df(self._node_pool.values(), self._edges, make_table=make_table)
        # edges, offsets = self._get_offsets(edges)  # TODO should include offsets from table with nodes?
        # nodes = self._assign_node_strings(nodes, offsets)
        return nodes, edges, offsets


def make_python_ast_graph(
        source_code, add_reverse_edges=False, save_node_strings=False, add_mention_instances=False,
        graph_builder_class=None, node_edge_definition_class=None, make_table=True, **kwargs
):
    if graph_builder_class is None:
        graph_builder_class = GraphParserV3
    if node_edge_definition_class is None:
        node_edge_definition_class = PythonNodeEdgeDefinitions

    g = graph_builder_class(
        source_code, node_edge_definition_class, add_reverse_edges=add_reverse_edges,
        save_node_strings=save_node_strings, add_mention_instances=add_mention_instances,  **kwargs
    )
    return g.to_df(make_table)


if __name__ == "__main__":
    parser = GraphParserV3(PythonNodeEdgeDefinitions)
    for example, code in PythonCodeExamplesForNodes.examples.items():
        parser.parse(code)