import ast
import logging
from copy import copy
from pprint import pprint
from collections.abc import Iterable
from typing import Union

import astunparse

from nid.ast.graph_builder.common.identifiers import IdentifierPool
from nid.ast.graph_builder.v2.definitions import PythonNodeEdgeDefinitions, PythonSharedNodes
from nid.ast.graph_builder.v2.primitives import GNode
from nid.ast.string_tools import get_cum_lens, get_byte_to_char_map, to_offsets


class GraphParserV2(object):
    def __init__(self, source, add_reverse_edges=True, add_mention_instances=False, **kwargs):
        self.source = source.split("\n")  # lines of the source code
        self.root = ast.parse(source)
        self.current_condition = []
        self.condition_status = []
        self.scope = []
        self._add_reverse_edges = add_reverse_edges
        self._add_mention_instances = add_mention_instances

        self._identifier_pool = IdentifierPool()
        self._prepare_offset_structures()

    def _prepare_offset_structures(self):
        self._body = "\n".join(self.source)
        self._cum_lens = get_cum_lens(self._body, as_bytes=True)
        self._byte2char = get_byte_to_char_map(self._body)

    def _range_to_offset(self, line, end_line, col_offset, end_col_offset):
        if line is None:
            return None
        return to_offsets(
            self._body, [(line, end_line, col_offset, end_col_offset, None)], cum_lens=self._cum_lens,
                   b2c=self._byte2char, as_bytes=True
        )[-1][:2]

    def _is_span_exception(self, node):
        return isinstance(node, ast.ExceptHandler) or isinstance(node, ast.Try)

    def _find_comparator(self, str_):
        comparators = ["==", "!=", ">", "<", ">=", "<=", " in ", " not in ", " is ", " not is "]
        for cmp in comparators:
            if cmp in str_:
                start = str_.index(cmp)
                break
        else:
            raise Exception("Comparator not found")

        len_ = len(cmp)
        if cmp in {" in ", " not in ", " is ", " not is "}:
            start += 1
            len_ -= 1
        return start, len_

    def _handle_span_exceptions(self, node, line, end_line, col_offset, end_col_offset):
        exception_handled = False
        expected_string = None
        skip_check = False
        if isinstance(node, ast.ExceptHandler):
            end_line = line
            end_col_offset = col_offset + 6
            exception_handled = True
            expected_string = "except"
        elif isinstance(node, ast.Try):
            end_line = line
            end_col_offset = col_offset + 3
            exception_handled = True
            expected_string = "try"
        elif isinstance(node, ast.If):
            end_line = line
            if self.source[line][col_offset] == "i":
                end_col_offset = col_offset + 2
                expected_string = "if"
            elif self.source[line][col_offset] == "e":
                end_col_offset = col_offset + 4
                expected_string = "elif"
            else:
                assert False
            exception_handled = True
        elif isinstance(node, ast.For):
            end_line = line
            end_col_offset = col_offset + 3
            exception_handled = True
            expected_string = "for"
        elif isinstance(node, ast.AsyncFor):
            end_line = line
            end_col_offset = col_offset + 9
            exception_handled = True
            handling_async = True
            expected_string = "async for"
        elif isinstance(node, ast.While):
            end_line = line
            end_col_offset = col_offset + 5
            exception_handled = True
            expected_string = "while"
        elif isinstance(node, ast.With):
            end_line = line
            end_col_offset = col_offset + 4
            exception_handled = True
            expected_string = "with"
        elif isinstance(node, ast.AsyncWith):
            end_line = line
            end_col_offset = col_offset + 10
            exception_handled = True
            handling_async = True
            expected_string = "async with"
        elif isinstance(node, ast.FunctionDef):
            # assert "(" in self.source[line] and self.source[line].count("def ") == 1
            end_line = line
            end_col_offset = col_offset + 3
            # end_col_offset = col_offset + 4 + len(self.source[line].split("def ")[1].split("(")[0])
            exception_handled = True
            expected_string = "def"
        elif isinstance(node, ast.AsyncFunctionDef):
            # assert "(" in self.source[line] and self.source[line].count("async def ") == 1
            end_line = line
            end_col_offset = col_offset + 9
            # end_col_offset = col_offset + 10 + len(self.source[line].split("async def ")[1].split("(")[0])
            exception_handled = True
            handling_async = True
            expected_string = "async def"
        elif isinstance(node, ast.ClassDef):
            # assert (":" in self.source[line] or ":" in self.source[line]) and self.source[line].count("class ") == 1
            # if "(" in self.source[line]:
            #     def_len = len(self.source[line].split("class ")[1].split("(")[0])
            # else:
            #     def_len = len(self.source[line].split("class ")[1].split(":")[0])
            end_line = line
            # end_col_offset = col_offset + 6 + def_len
            end_col_offset = col_offset + 5
            exception_handled = True
            expected_string = "class"
        elif isinstance(node, ast.Import):
            end_line = line
            end_col_offset = col_offset + 6
            exception_handled = True
            handling_async = True
            expected_string = "import"
        elif isinstance(node, ast.Delete):
            end_line = line
            end_col_offset = col_offset + 3
            exception_handled = True
            expected_string = "del"
        elif isinstance(node, ast.ImportFrom):
            end_line = line
            end_col_offset = col_offset + 4
            exception_handled = True
            expected_string = "from"
        elif isinstance(node, ast.List):
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "["
        elif isinstance(node, ast.Dict):
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "{"
        elif isinstance(node, ast.Set):
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "{"
        elif isinstance(node, ast.Tuple):
            pass
            # possible variants
            # - (1,2)
            # 1,2
            # end_line = line
            # end_col_offset = col_offset + 1
            # exception_handled = True
        elif isinstance(node, ast.GeneratorExp):
            # cannot if passed as argument ot function
            pass
            # end_line = line
            # end_col_offset = col_offset + 1
            # exception_handled = True
        elif isinstance(node, ast.ListComp):
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "["
        elif isinstance(node, ast.DictComp):
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "{"
        elif isinstance(node, ast.SetComp):
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "{"
        elif isinstance(node, ast.Starred):
            assert "*" in self.source[line]
            end_line = line
            end_col_offset = col_offset + 1
            exception_handled = True
            expected_string = "*"
        elif isinstance(node, ast.Return):
            end_line = line
            end_col_offset = col_offset + 6
            exception_handled = True
            expected_string = "return"
        elif isinstance(node, ast.Global):
            end_line = line
            end_col_offset = col_offset + 6
            exception_handled = True
            expected_string = "global"
        elif isinstance(node, ast.Nonlocal):
            end_line = line
            end_col_offset = col_offset + 8
            exception_handled = True
            expected_string = "nonlocal"
        elif isinstance(node, ast.Assert):
            end_line = line
            end_col_offset = col_offset + 6
            exception_handled = True
            expected_string = "assert"
        elif isinstance(node, ast.Lambda):
            end_line = line
            end_col_offset = col_offset + 6
            exception_handled = True
            expected_string = "lambda"
        elif isinstance(node, ast.Raise):
            end_line = line
            end_col_offset = col_offset + 5
            exception_handled = True
            expected_string = "raise"
        elif isinstance(node, ast.Await):
            end_line = line
            end_col_offset = col_offset + 5
            exception_handled = True
            expected_string = "await"
        elif isinstance(node, ast.Yield):
            end_line = line
            end_col_offset = col_offset + 5
            exception_handled = True
            expected_string = "yield"
        elif isinstance(node, ast.YieldFrom):
            end_line = line
            end_col_offset = col_offset + 10
            exception_handled = True
            handling_async = True
            expected_string = "yield from"
        elif isinstance(node, ast.arg):
            # do not bother
            skip_check = True
        elif type(node) in {
            ast.Name,
            ast.Constant,
            # ast.arg,
            ast.JoinedStr,
            ast.Expr
        }:  # do not bother
            pass
        elif type(node) in {
            ast.Compare,  # can use comparator operator
            ast.BoolOp,  # could be multiline
            ast.BinOp,  # could be multiline
            ast.Assign,  # could be multiline
            ast.AnnAssign,  # could be multiline
            ast.AugAssign,  # could be multiline
            ast.Subscript,  # could be multiline
            ast.Attribute,  # could be multiline
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

        char_offset = self._range_to_offset(line, end_line, col_offset, end_col_offset)

        if skip_check is False:
            if exception_handled is False:
                try:
                    ast.parse(self._body[char_offset[0]: char_offset[1]])
                except SyntaxError:
                    try:
                        ast.parse("(" + self._body[char_offset[0]: char_offset[1]] + ")")
                    except:
                        raise Exception("Range parsed incorrectly")
            else:
                assert expected_string == self._body[char_offset[0]: char_offset[1]], f"{expected_string} != {self._body[char_offset[0]: char_offset[1]]}"

        return line, end_line, col_offset, end_col_offset, char_offset

    def _get_positions(self, node, full=False):
        if node is not None and hasattr(node, "lineno"):
            line = node.lineno - 1
            end_line = node.end_lineno - 1
            col_offset = node.col_offset
            end_col_offset = node.end_col_offset
            char_offset = None
            if full is False:
                line, end_line, col_offset, end_col_offset, char_offset = self._handle_span_exceptions(
                    node, line,
                    end_line,
                    col_offset,
                    end_col_offset
                )
        else:
            line = end_line = col_offset = end_col_offset = char_offset = None
        return line, end_line, col_offset, end_col_offset, char_offset

    def _get_source_from_ast_range(self, node, strip=True):
        try:
            source = astunparse.unparse(node).strip()
        except:
            start_line, end_line, start_col, end_col, char_offset = self._get_positions(node, full=True)

            source = ""
            num_lines = end_line - start_line + 1
            if start_line == end_line:
                section = self.source[start_line].encode("utf8")[start_col: end_col].decode(
                    "utf8")
                source += section.strip() if strip else section + "\n"
            else:
                for ind, lineno in enumerate(range(start_line, end_line + 1)):
                    if ind == 0:
                        section = self.source[lineno].encode("utf8")[start_col:].decode(
                            "utf8")
                        source += section.strip() if strip else section + "\n"
                    elif ind == num_lines - 1:
                        section = self.source[lineno].encode("utf8")[:end_col].decode(
                            "utf8")
                        source += section.strip() if strip else section + "\n"
                    else:
                        section = self.source[lineno]
                        source += section.strip() if strip else section + "\n"

        return source.rstrip()

    def _get_name(self, *, node=None, name=None, type=None, add_random_identifier=False):

        random_identifier = self._identifier_pool.get_new_identifier()

        if node is not None:
            name = f"{node.__class__.__name__}_{random_identifier}"
            type = node.__class__.__name__
        else:
            if add_random_identifier:
                name = f"{name}_{random_identifier}"

        if hasattr(node, "lineno"):
            node_string = self._get_source_from_ast_range(node, strip=False)
        else:
            node_string = None

        if len(self.scope) > 0:
            return GNode(name=name, type=type, scope=copy(self.scope[-1]), string=node_string)
        else:
            return GNode(name=name, type=type, string=node_string)

    def _parse_node(self, node):
        n_type = type(node).__name__
        if n_type in PythonNodeEdgeDefinitions.ast_node_type_edges:
            return self._generic_parse(node, PythonNodeEdgeDefinitions.ast_node_type_edges[n_type])
        elif n_type in PythonNodeEdgeDefinitions.overridden_node_type_edges:
            method_name = "parse_" + n_type
            return self.__getattribute__(method_name)(node)
        elif n_type in PythonNodeEdgeDefinitions.iterable_nodes:
            return self.parse_iterable(node)
        elif n_type in PythonNodeEdgeDefinitions.named_nodes:
            return self.parse_name(node)
        elif n_type in PythonNodeEdgeDefinitions.constant_nodes:
            return self.parse_Constant(node)
        elif n_type in PythonNodeEdgeDefinitions.operand_nodes:
            return self.parse_op_name(node)
        elif n_type in PythonNodeEdgeDefinitions.control_flow_nodes:
            return self.parse_control_flow(node)
        else:
            return self._generic_parse(node, node._fields)

    def _add_edge(
            self, edges, src, dst, type, scope=None,
            position_node=None, var_position_node=None
    ):
        edges.append({
            "src": src, "dst": dst, "type": type, "scope": scope,
        })

        line, end_line, col_offset, end_col_offset, _ = self._get_positions(position_node)
        if self._is_span_exception(position_node):
            line = end_line = col_offset = end_col_offset = None

        if line is not None:
            edges[-1].update({
                "line": line, "end_line": end_line, "col_offset": col_offset, "end_col_offset": end_col_offset
            })

        var_line, var_end_line, var_col_offset, var_end_col_offset, _ = self._get_positions(var_position_node)

        if var_line is not None:
            edges[-1].update({
                "var_line": var_line, "var_end_line": var_end_line, "var_col_offset": var_col_offset, "var_end_col_offset": var_end_col_offset
            })

        reverse_type = PythonNodeEdgeDefinitions.reverse_edge_exceptions.get(type, type + "_rev")
        if self._add_reverse_edges is True and reverse_type is not None and \
                not PythonSharedNodes.is_shared_name_type(src.name, src.type):
            edges.append({
                "src": dst, "dst": src, "type": reverse_type, "scope": scope
            })

    def _parse_body(self, nodes):
        edges = []
        last_node = None
        last_node_ = None
        for node in nodes:
            s = self._parse_node(node)
            if isinstance(s, tuple):
                # it appears that the rule below will remove all the constants form the function body
                # we actually do not need edges next and prev pointing to constants
                if s[1].type == "Constant":  # this happens when processing docstring, as a result a lot of nodes are connected to the node Constant_
                    continue                    # in general, constant node has no affect as a body expression, can skip
                # some parsers return edges and names?
                edges.extend(s[0])

                if last_node is not None:
                    self._add_edge(edges, src=last_node, dst=s[1], type="next", scope=self.scope[-1])

                last_node = s[1]
                last_node_ = node if not isinstance(node, ast.Expr) else node.value

                for cond_name, cond_stat in zip(self.current_condition[-1:], self.condition_status[-1:]):
                    self._add_edge(edges, src=last_node, dst=cond_name, type=cond_stat, scope=self.scope[-1],
                                   position_node=last_node_)  # "defined_in_" +
            else:
                edges.extend(s)
        return edges

    def _parse_in_context(self, cond_name, cond_stat, edges, body):
        if isinstance(cond_name, str):
            cond_name = [cond_name]
            cond_stat = [cond_stat]
        elif isinstance(cond_name, GNode):
            cond_name = [cond_name]
            cond_stat = [cond_stat]

        for cn, cs in zip(cond_name, cond_stat):
            self.current_condition.append(cn)
            self.condition_status.append(cs)

        edges.extend(self._parse_body(body))

        for i in range(len(cond_name)):
            self.current_condition.pop(-1)
            self.condition_status.pop(-1)

    def _parse_as_mention(self, name):
        mention_name = GNode(name=name + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
        name_ = GNode(name=name, type="Name")
        # mention_name = (name + "@" + self.scope[-1], "mention")

        # edge from name to mention in a function
        edges = []
        self._add_edge(edges, src=name_, dst=mention_name, type="local_mention", scope=self.scope[-1])

        if self._add_mention_instances:
            mention_instance = self._get_name(name="instance", type="instance", add_random_identifier=True)
            mention_instance.string = name
            self._add_edge(edges, src=mention_name, dst=mention_instance, type="instance", scope=self.scope[-1])
            mention_name = mention_instance

        return edges, mention_name

    def _parse_operand(self, node):
        # need to make sure upper level name is correct; handle @keyword; type placeholder for sourcetrail nodes?
        edges = []
        if isinstance(node, str):
            # fall here when parsing attributes, they are given as strings; should attributes be parsed into subwords?
            if "@" in node:
                parts = node.split("@")
                node = GNode(name=parts[0], type=parts[1])
            else:
                # node = GNode(name=node, type="Name")
                node = ast.Name(node)
                edges_, node = self._parse_node(node)
                edges.extend(edges_)
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = GNode(name=str(node), type="ast_Literal")
            # iter_ = str(node)
        elif isinstance(node, GNode):
            iter_ = node
        else:
            iter_e = self._parse_node(node)
            if type(iter_e) == str:
                iter_ = iter_e
            elif isinstance(iter_e, GNode):
                iter_ = iter_e
            elif type(iter_e) == tuple:
                ext_edges, name = iter_e
                assert isinstance(name, GNode)
                edges.extend(ext_edges)
                iter_ = name
            else:
                # unexpected scenario
                print(node)
                print(ast.dump(node))
                print(iter_e)
                print(type(iter_e))
                pprint(self.source)
                print(self.source[node.lineno - 1].strip())
                raise Exception()

        return iter_, edges

    def _parse_and_add_operand(self, node_name, operand, type, edges):
        operand_name, ext_edges = self._parse_operand(operand)
        edges.extend(ext_edges)
        self._add_edge(edges, src=operand_name, dst=node_name, type=type, scope=self.scope[-1], position_node=operand)

    def _generic_parse(self, node, operands, with_name=None, ensure_iterables=False):

        edges = []

        if with_name is None:
            node_name = self._get_name(node=node)
        else:
            node_name = with_name

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"]:
                logging.warning(f"Not clear which node is processed here {ast.dump(node)}")
                self._parse_in_context(node_name, operand, edges, node.__getattribute__(operand))
            else:
                operand_ = node.__getattribute__(operand)
                if operand_ is not None:
                    if isinstance(operand_, Iterable) and not isinstance(operand_, str):
                        # TODO:
                        #  appears as leaf node if the iterable is empty. suggest adding an "EMPTY" element
                        for oper_ in operand_:
                            self._parse_and_add_operand(node_name, oper_, operand, edges)
                    else:
                        self._parse_and_add_operand(node_name, operand_, operand, edges)

        return edges, node_name

    def parse_Module(self, node):
        edges, module_name = self._generic_parse(node, [])
        self.scope.append(module_name)
        self._parse_in_context(module_name, "defined_in_module", edges, node.body)
        self.scope.pop(-1)
        return edges, module_name

    def parse_FunctionDef(self, node):
        # need to create function name before generic_parse so that the scope is set up correctly
        # scope is used to create local mentions of variable and function names
        fdef_node_name = self._get_name(node=node)
        self.scope.append(fdef_node_name)

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

        edges, f_name = self._generic_parse(node, to_parse, with_name=fdef_node_name)

        if node.returns is not None:
            annotation_string = self._get_source_from_ast_range(node.returns)
            annotation = GNode(name=annotation_string,
                               type="type_annotation")
            self._add_edge(edges, src=annotation, dst=f_name, type="returned_by", scope=self.scope[-1],
                           position_node=node.returns)

        self._parse_in_context(f_name, "defined_in_function", edges, node.body)

        self.scope.pop(-1)

        ext_edges, func_name = self._parse_as_mention(name=node.name)
        edges.extend(ext_edges)

        assert isinstance(node.name, str)
        self._add_edge(edges, src=f_name, dst=func_name, type="function_name", scope=self.scope[-1])

        return edges, f_name

    def parse_AsyncFunctionDef(self, node):
        return self.parse_FunctionDef(node)

    def parse_ClassDef(self, node):

        edges, class_node_name = self._generic_parse(node, [])

        self.scope.append(class_node_name)
        self._parse_in_context(class_node_name, "defined_in_class", edges, node.body)
        self.scope.pop(-1)

        ext_edges, cls_name = self._parse_as_mention(name=node.name)
        edges.extend(ext_edges)
        self._add_edge(edges, src=class_node_name, dst=cls_name, type="class_name", scope=self.scope[-1])

        return edges, class_node_name

    def parse_With(self, node):
        edges, with_name = self._generic_parse(node, ["items"])

        self._parse_in_context(with_name, "executed_inside_with", edges, node.body)

        return edges, with_name

    def parse_AsyncWith(self, node):
        return self.parse_With(node)

    def parse_arg(self, node, default_value=None):
        name = self._get_name(node=node)
        edges, mention_name = self._parse_as_mention(node.arg)
        self._add_edge(edges, src=mention_name, dst=name, type="arg", scope=self.scope[-1])

        if node.annotation is not None:
            annotation_string = self._get_source_from_ast_range(node.annotation)
            annotation = GNode(name=annotation_string,
                               type="type_annotation")
            mention_name = GNode(name=node.arg + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
            self._add_edge(edges, src=annotation, dst=mention_name, type="annotation_for", scope=self.scope[-1],
                           position_node=node.annotation, var_position_node=node)

        if default_value is not None:
            deflt_ = self._parse_node(default_value)
            if isinstance(deflt_, tuple):
                edges.extend(deflt_[0])
                default_val = deflt_[1]
            else:
                default_val = deflt_
            self._add_edge(edges, default_val, name, type="default", scope=copy(self.scope[-1]),
                           position_node=default_value)
        return edges, name

    def parse_AnnAssign(self, node):
        annotation_string = self._get_source_from_ast_range(node.annotation)
        annotation = GNode(name=annotation_string,
                           type="type_annotation")
        edges, name = self._generic_parse(node, ["target", "value"])
        try:
            mention_name = GNode(name=node.target.id + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
        except Exception as e:
            mention_name = name

        self._add_edge(edges, src=annotation, dst=mention_name, type="annotation_for", scope=self.scope[-1],
                       position_node=node.annotation, var_position_node=node)
        return edges, name

    def parse_Lambda(self, node):
        # this is too ambiguous
        edges, lmb_name = self._generic_parse(node, [])
        self._parse_and_add_operand(lmb_name, node.body, "lambda", edges)

        return edges, lmb_name

    def parse_IfExp(self, node):
        edges, ifexp_name = self._generic_parse(node, ["test"])
        self._parse_and_add_operand(ifexp_name, node.body, "if_true", edges)
        self._parse_and_add_operand(ifexp_name, node.orelse, "if_false", edges)
        return edges, ifexp_name

    def parse_ExceptHandler(self, node: ast.ExceptHandler):
        # not parsing "name" field
        # except handler is unique for every function
        return self._generic_parse(node, ["type"])

    def parse_keyword(self, node):
        if isinstance(node.arg, str):
            # change arg name so that it does not mix with variable names
            node.arg += "@#keyword#"
            return self._generic_parse(node, ["arg", "value"])
        else:
            return self._generic_parse(node, ["value"])

    def parse_name(self, node: Union[ast.Name, ast.NameConstant]):
        if isinstance(node, ast.Name):
            return self._parse_as_mention(str(node.id))
        elif isinstance(node, ast.NameConstant):
            return GNode(name=str(node.value), type="NameConstant")

    def parse_Attribute(self, node: ast.Attribute):
        if node.attr is not None:
            # change attr name so that it does not mix with variable names
            node.attr += "@#attr#"
        return self._generic_parse(node, ["value", "attr"])

    def parse_Constant(self, node: ast.Constant):
        # TODO
        # decide whether this name should be unique or not
        value_type = type(node.value).__name__
        name = GNode(name=f"Constant_{value_type}", type="Constant")
        # name = "Constant_"
        # if node.kind is not None:
        #     name += ""
        return name

    def parse_op_name(self, node):
        return GNode(name=node.__class__.__name__, type="Op")

    def parse_Num(self, node):
        return str(node.n)

    def parse_Str(self, node):
        return self._generic_parse(node, [])
        # return node.s

    def parse_Bytes(self, node):
        return repr(node.s)

    def parse_If(self, node):

        edges, if_name = self._generic_parse(node, ["test"])

        self._parse_in_context(if_name, "executed_if_true", edges, node.body)
        self._parse_in_context(if_name, "executed_if_false", edges, node.orelse)

        return edges, if_name

    def parse_For(self, node):

        edges, for_name = self._generic_parse(node, ["target", "iter"])

        self._parse_in_context(for_name, "executed_in_for", edges, node.body)
        self._parse_in_context(for_name, "executed_in_for_orelse", edges, node.orelse)
        
        return edges, for_name

    def parse_AsyncFor(self, node):
        return self.parse_For(node)
        
    def parse_Try(self, node):

        edges, try_name = self._generic_parse(node, [])

        self._parse_in_context(try_name, "executed_in_try", edges, node.body)
        
        for h in node.handlers:
            handler_name, ext_edges = self._parse_operand(h)
            edges.extend(ext_edges)
            # self.parse_in_context([try_name, handler_name], ["executed_in_try_except", "executed_with_try_handler"], edges, h.body)
            self._parse_in_context([handler_name], ["executed_with_try_handler"], edges, h.body)
            self._add_edge(edges, src=handler_name, dst=try_name, type="executed_in_try_except", scope=self.scope[-1])

        self._parse_in_context(try_name, "executed_in_try_final", edges, node.finalbody)
        self._parse_in_context(try_name, "executed_in_try_else", edges, node.orelse)
        
        return edges, try_name
        
    def parse_While(self, node: ast.While):
        edges, while_name = self._generic_parse(node, ["test"])
        self._parse_in_context([while_name], ["executed_in_while"], edges, node.body)
        return edges, while_name

    def parse_Expr(self, node: ast.Expr):
        edges = []
        expr_name, ext_edges = self._parse_operand(node.value)
        edges.extend(ext_edges)
        
        return edges, expr_name

    def parse_control_flow(self, node):
        edges = []
        ctrlflow_name = self._get_name(name="ctrl_flow", type="CtlFlowInstance", add_random_identifier=True)
        self._add_edge(edges, src=GNode(name=node.__class__.__name__, type="CtlFlow"), dst=ctrlflow_name,
                       type="control_flow", scope=self.scope[-1])

        return edges, ctrlflow_name

    def parse_iterable(self, node):
        return self._generic_parse(node, ["elts"], ensure_iterables=True)

    def parse_Dict(self, node: ast.Dict):
        return self._generic_parse(node, ["keys", "values"], ensure_iterables=True)

    def parse_JoinedStr(self, node: ast.JoinedStr):
        joinedstr_name = GNode(name="JoinedStr_", type="JoinedStr")
        return [], joinedstr_name

    def parse_FormattedValue(self, node: ast.FormattedValue):
        return self._generic_parse(node, ["value"])

    def parse_arguments(self, node: ast.arguments):
        edges, arguments = self._generic_parse(node, [])

        if node.vararg is not None:
            ext_edges_, vararg = self.parse_arg(node.vararg)
            edges.extend(ext_edges_)
            self._add_edge(edges, vararg, arguments, type="vararg", scope=copy(self.scope[-1]),
                           position_node=node.vararg)

        for i in range(len(node.posonlyargs)):
            ext_edges_, posarg = self.parse_arg(node.posonlyargs[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, posarg, arguments, type="posonlyarg", scope=copy(self.scope[-1]),
                           position_node=node.posonlyargs[i])

        without_default = len(node.args) - len(node.defaults)
        for i in range(without_default):
            ext_edges_, just_arg = self.parse_arg(node.args[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, just_arg, arguments, type="arg", scope=copy(self.scope[-1]),
                           position_node=node.args[i])

        for ind, i in enumerate(range(without_default, len(node.args))):
            ext_edges_, just_arg = self.parse_arg(node.args[i], default_value=node.defaults[ind])
            edges.extend(ext_edges_)
            self._add_edge(edges, just_arg, arguments, type="arg", scope=copy(self.scope[-1]),
                           position_node=node.args[i])

        for i in range(len(node.kwonlyargs)):
            ext_edges_, kw_arg = self.parse_arg(node.kwonlyargs[i], default_value=node.kw_defaults[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, kw_arg, arguments, type="kwonlyarg", scope=copy(self.scope[-1]),
                           position_node=node.kwonlyargs[i])

        if node.kwarg is not None:
            ext_edges_, kwarg = self.parse_arg(node.kwarg)
            edges.extend(ext_edges_)
            self._add_edge(edges, kwarg, arguments, type="kwarg", scope=copy(self.scope[-1]), position_node=node.kwarg)

        return edges, arguments
        # return self.generic_parse(node, ["args", "vararg", "kwarg", "kwonlyargs", "posonlyargs"])

    def parse_comprehension(self, node: ast.comprehension):
        edges = []

        cph_name = self._get_name(name="comprehension", type="comprehension", add_random_identifier=True)

        target, ext_edges = self._parse_operand(node.target)
        edges.extend(ext_edges)

        self._add_edge(edges, src=target, dst=cph_name, type="target", scope=self.scope[-1], position_node=node.target)

        iter_, ext_edges = self._parse_operand(node.iter)
        edges.extend(ext_edges)

        self._add_edge(edges, src=iter_, dst=cph_name, type="iter", scope=self.scope[-1], position_node=node.iter)

        for if_ in node.ifs:
            if_n, ext_edges = self._parse_operand(if_)
            edges.extend(ext_edges)
            self._add_edge(edges, src=if_n, dst=cph_name, type="ifs", scope=self.scope[-1])

        return edges, cph_name
