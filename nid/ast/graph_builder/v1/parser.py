import ast
from copy import copy
from pprint import pprint
from collections.abc import Iterable

from nid.ast.graph_builder.common.identifiers import IdentifierPool
from nid.ast.graph_builder.v1.primitives import GNode


class GraphParserV1:
    _source = None  # lines of the source code
    _root = None

    def __init__(self, *args, **kwargs):
        self.current_condition = []
        self.condition_status = []
        self.scope = []

        self._identifier_pool = IdentifierPool()

    def _get_source_from_ast_range(self, start_line, end_line, start_col, end_col, strip=True):
        assert self._source is not None, "Source code not initialized"
        source = ""
        num_lines = end_line - start_line + 1
        if start_line == end_line:
            section = self._source[start_line - 1].encode("utf8")[start_col:end_col].decode(
                "utf8")
            source += section.strip() if strip else section + "\n"
        else:
            for ind, lineno in enumerate(range(start_line - 1, end_line)):
                if ind == 0:
                    section = self._source[lineno].encode("utf8")[start_col:].decode(
                        "utf8")
                    source += section.strip() if strip else section + "\n"
                elif ind == num_lines - 1:
                    section = self._source[lineno].encode("utf8")[:end_col].decode(
                        "utf8")
                    source += section.strip() if strip else section + "\n"
                else:
                    section = self._source[lineno]
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
            node_string = self._get_source_from_ast_range(
                node.lineno, node.end_lineno, node.col_offset,  # type: ignore
                node.end_col_offset, strip=False  # type: ignore
            )
        else:
            node_string = None

        if len(self.scope) > 0:
            return GNode(name=name, type=type, scope=copy(self.scope[-1]), string=node_string)
        else:
            return GNode(name=name, type=type, string=node_string)

    def _parse_node(self, node):
        n_type = type(node)
        method_name = "parse_" + n_type.__name__
        if hasattr(self, method_name):
            return self.__getattribute__(method_name)(node)
        else:
            return self._generic_parse(node, node._fields)

    def _parse_body(self, nodes):
        edges = []
        last_node = None
        for node in nodes:
            s = self._parse_node(node)
            if isinstance(s, tuple):
                if s[1].type == "Constant":
                    # this happens when processing docstring, as a result a lot of nodes are connected to the
                    # node Constant_. in general, constant node has no affect as a body expression, can skip
                    continue
                edges.extend(s[0])

                if last_node is not None:
                    edges.append({"dst": s[1], "src": last_node, "type": "next", "scope": copy(self.scope[-1])})
                    edges.append({"dst": last_node, "src": s[1], "type": "prev", "scope": copy(self.scope[-1])})

                last_node = s[1]

                for cond_name, cond_stat in zip(self.current_condition[-1:], self.condition_status[-1:]):
                    edges.append({
                        "scope": copy(self.scope[-1]), "src": last_node, "dst": cond_name,
                        "type": "defined_in_" + cond_stat
                    })
                    edges.append({
                        "scope": copy(self.scope[-1]), "src": cond_name, "dst": last_node,
                        "type": "defined_in_" + cond_stat + "_rev"
                    })
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
        name = GNode(name=name, type="Name")

        edges = [
            {"scope": copy(self.scope[-1]), "src": name, "dst": mention_name, "type": "local_mention"},
        ]
        return edges, mention_name

    def _parse_operand(self, node):
        assert self._source is not None, "Source code not initialized"

        edges = []
        if isinstance(node, str):
            if "@" in node:
                parts = node.split("@")
                node = GNode(name=parts[0], type=parts[1])
            else:
                node = GNode(name=node, type="Name")
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = GNode(name=str(node), type="ast_Literal")
        elif isinstance(node, GNode):
            iter_ = node
        else:
            iter_e = self._parse_node(node)
            if isinstance(iter_e, str):
                iter_ = iter_e
            elif isinstance(iter_e, GNode):
                iter_ = iter_e
            elif isinstance(iter_e, tuple):
                ext_edges, name = iter_e
                assert isinstance(name, GNode)
                edges.extend(ext_edges)
                iter_ = name
            else:
                # unexpected scenario
                print(node)  # TODO format
                print(ast.dump(node))
                print(iter_e)
                print(type(iter_e))
                pprint(self._source)
                print(self._source[node.lineno - 1].strip())
                raise Exception()

        return iter_, edges

    def _parse_and_add_operand(self, node_name, operand, type, edges):
        operand_name, ext_edges = self._parse_operand(operand)
        edges.extend(ext_edges)

        if hasattr(operand, "lineno"):
            edges.append({"scope": copy(self.scope[-1]), "src": operand_name, "dst": node_name, "type": type,
                          "line": operand.lineno - 1, "end_line": operand.end_lineno - 1,
                          "col_offset": operand.col_offset, "end_col_offset": operand.end_col_offset})
        else:
            edges.append({"scope": copy(self.scope[-1]), "src": operand_name, "dst": node_name, "type": type})

        edges.append({"scope": copy(self.scope[-1]), "src": node_name, "dst": operand_name, "type": type + "_rev"})

    def _generic_parse(self, node, operands, with_name=None, ensure_iterables=False):

        edges = []

        if with_name is None:
            node_name = self._get_name(node=node)
        else:
            node_name = with_name

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"]:
                self._parse_in_context(node_name, "operand", edges, node.__getattribute__(operand))
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

    def _parse_type_node(self, node):
        assert self._source is not None, "Source code not initialized"
        
        if node.lineno == node.end_lineno:
            type_str = self._source[node.lineno][node.col_offset - 1: node.end_col_offset]
        else:
            type_str = ""
            for ln in range(node.lineno - 1, node.end_lineno):
                if ln == node.lineno - 1:
                    type_str += self._source[ln][node.col_offset - 1:].strip()
                elif ln == node.end_lineno - 1:
                    type_str += self._source[ln][:node.end_col_offset].strip()
                else:
                    type_str += self._source[ln].strip()
        return type_str

    def _parse_Module(self, node):
        edges, module_name = self._generic_parse(node, [])
        self.scope.append(module_name)
        self._parse_in_context(module_name, "module", edges, node.body)
        self.scope.pop(-1)
        return edges, module_name

    def _parse_FunctionDef(self, node):
        # need to create function name before generic_parse so that the scope is set up correctly
        # scope is used to create local mentions of variable and function names
        fdef_node_name = self._get_name(node=node)
        self.scope.append(fdef_node_name)

        to_parse = []
        if len(node.args.args) > 0 or node.args.vararg is not None:
            to_parse.append("args")
        if len("decorator_list") > 0:
            to_parse.append("decorator_list")

        edges, f_name = self._generic_parse(node, to_parse, with_name=fdef_node_name)

        if node.returns is not None:
            annotation_string = self._get_source_from_ast_range(node.returns.lineno, node.returns.end_lineno,
                                                                node.returns.col_offset, node.returns.end_col_offset)
            annotation = GNode(name=annotation_string,
                               type="type_annotation")
            edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": f_name, "type": "returned_by",
                          "line": node.returns.lineno - 1, "end_line": node.returns.end_lineno - 1,
                          "col_offset": node.returns.col_offset, "end_col_offset": node.returns.end_col_offset})

        self._parse_in_context(f_name, "function", edges, node.body)

        self.scope.pop(-1)

        ext_edges, func_name = self._parse_as_mention(name=node.name)
        edges.extend(ext_edges)

        assert isinstance(node.name, str)
        edges.append({"scope": copy(self.scope[-1]), "src": f_name, "dst": func_name, "type": "function_name"})
        edges.append({"scope": copy(self.scope[-1]), "src": func_name, "dst": f_name, "type": "function_name_rev"})

        return edges, f_name

    def _parse_AsyncFunctionDef(self, node):
        return self._parse_FunctionDef(node)

    def _parse_Assign(self, node):
        edges, assign_name = self._generic_parse(node, ["value", "targets"])
        return edges, assign_name

    def _parse_AugAssign(self, node):
        return self._generic_parse(node, ["target", "op", "value"])

    def _parse_ClassDef(self, node):
        edges, class_node_name = self._generic_parse(node, [])
        self.scope.append(class_node_name)

        self._parse_in_context(class_node_name, "class", edges, node.body)
        self.scope.pop(-1)

        ext_edges, cls_name = self._parse_as_mention(name=node.name)
        edges.extend(ext_edges)
        edges.append({"scope": copy(self.scope[-1]), "src": class_node_name, "dst": cls_name, "type": "class_name"})
        edges.append({"scope": copy(self.scope[-1]), "src": cls_name, "dst": class_node_name, "type": "class_name_rev"})

        return edges, class_node_name

    def _parse_ImportFrom(self, node):
        if node.module is not None:
            node.module = ast.Name(node.module)
        return self._generic_parse(node, ["module", "names"])

    def _parse_Import(self, node):
        return self._generic_parse(node, ["names"])

    def _parse_Delete(self, node):
        return self._generic_parse(node, ["targets"])

    def _parse_Global(self, node):
        return self._generic_parse(node, ["names"])

    def _parse_Nonlocal(self, node):
        return self._generic_parse(node, ["names"])

    def _parse_With(self, node):
        edges, with_name = self._generic_parse(node, ["items"])
        self._parse_in_context(with_name, "with", edges, node.body)
        return edges, with_name

    def _parse_AsyncWith(self, node):
        return self._parse_With(node)

    def _parse_withitem(self, node):
        return self._generic_parse(node, ['context_expr', 'optional_vars'])

    def _parse_alias(self, node):
        # TODO
        # aliases should be handled by sourcetrail. here i am trying to assign alias to a
        # local mention of the module. maybe I should simply ignore aliases altogether
        if node.name is not None:
            node.name = ast.Name(node.name)
        if node.asname is not None:
            node.asname = ast.Name(node.asname)
        return self._generic_parse(node, ["name", "asname"])

    def _parse_arg(self, node):
        name = self._get_name(node=node)
        edges, mention_name = self._parse_as_mention(node.arg)
        edges.append({"scope": copy(self.scope[-1]), "src": mention_name, "dst": name, "type": 'arg'})
        edges.append({"scope": copy(self.scope[-1]), "src": name, "dst": mention_name, "type": 'arg_rev'})
        if node.annotation is not None:
            annotation_string = self._get_source_from_ast_range(node.annotation.lineno, node.annotation.end_lineno,
                                                                node.annotation.col_offset,
                                                                node.annotation.end_col_offset)
            annotation = GNode(name=annotation_string,
                               type="type_annotation")
            mention_name = GNode(name=node.arg + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
            edges.append(
                {
                    "scope": copy(self.scope[-1]),
                    "src": annotation,
                    "dst": mention_name,
                    "type": 'annotation_for',
                    "line": node.annotation.lineno - 1,
                    "end_line": node.annotation.end_lineno - 1,
                    "col_offset": node.annotation.col_offset,
                    "end_col_offset": node.annotation.end_col_offset,
                    "var_line": node.lineno - 1,
                    "var_end_line": node.end_lineno - 1,
                    "var_col_offset": node.col_offset,
                    "var_end_col_offset": node.end_col_offset
                }
            )
        return edges, name

    def _parse_AnnAssign(self, node):
        annotation_string = self._get_source_from_ast_range(node.annotation.lineno, node.annotation.end_lineno,
                                                            node.annotation.col_offset, node.annotation.end_col_offset)
        annotation = GNode(name=annotation_string,
                           type="type_annotation")
        edges, name = self._generic_parse(node, ["target"])
        try:
            mention_name = GNode(name=node.target.id + "@" + self.scope[-1].name, type="mention",
                                 scope=copy(self.scope[-1]))
            edges.append({
                "scope": copy(self.scope[-1]),
                "src": annotation,
                "dst": mention_name,
                "type": 'annotation_for',
                "line": node.annotation.lineno - 1,
                "end_line": node.annotation.end_lineno - 1,
                "col_offset": node.annotation.col_offset,
                "end_col_offset": node.annotation.end_col_offset,
                "var_line": node.lineno - 1,
                "var_end_line": node.end_lineno - 1,
                "var_col_offset": node.col_offset,
                "var_end_col_offset": node.end_col_offset
            })
        except Exception as e:
            edges.append({
                "scope": copy(self.scope[-1]),
                "src": annotation,
                "dst": name,
                "type": 'annotation_for',
                "line": node.annotation.lineno - 1,
                "end_line": node.annotation.end_lineno - 1,
                "col_offset": node.annotation.col_offset,
                "end_col_offset": node.annotation.end_col_offset,
                "var_line": node.lineno - 1,
                "var_end_line": node.end_lineno - 1,
                "var_col_offset": node.col_offset,
                "var_end_col_offset": node.end_col_offset
            })
        return edges, name

    def _parse_Subscript(self, node):
        return self._generic_parse(node, ["value", "slice"])

    def _parse_Slice(self, node):
        return self._generic_parse(node, ["lower", "upper", "step"])

    def _parse_ExtSlice(self, node):
        return self._generic_parse(node, ["dims"])

    def _parse_Index(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_Lambda(self, node):
        # this is too ambiguous
        edges, lmb_name = self._generic_parse(node, [])
        self._parse_and_add_operand(lmb_name, node.body, "lambda", edges)

        return edges, lmb_name

    def _parse_Starred(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_Yield(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_IfExp(self, node):
        edges, ifexp_name = self._generic_parse(node, ["test"])
        self._parse_and_add_operand(ifexp_name, node.body, "body", edges)
        self._parse_and_add_operand(ifexp_name, node.orelse, "orelse", edges)
        return edges, ifexp_name

    def _parse_ExceptHandler(self, node):
        return self._generic_parse(node, ["type"])

    def _parse_Call(self, node):
        return self._generic_parse(node, ["func", "args", "keywords"])

    def _parse_keyword(self, node):
        # change arg name so that it does not mix with variable names
        if isinstance(node.arg, str):
            node.arg += "@#keyword#"
            return self._generic_parse(node, ["arg", "value"])
        else:
            return self._generic_parse(node, ["value"])

    def _parse_name(self, node):
        if isinstance(node, ast.Name):
            return self._parse_as_mention(str(node.id))
        elif isinstance(node, ast.NameConstant):
            return GNode(name=str(node.value), type="NameConstant")

    def _parse_Attribute(self, node):
        if node.attr is not None:
            node.attr += "@#attr#"
        return self._generic_parse(node, ["value", "attr"])

    def _parse_Name(self, node):
        return self._parse_name(node)

    def _parse_NameConstant(self, node):
        return self._parse_name(node)

    def _parse_Constant(self, node):
        name = GNode(name="Constant_", type="Constant")
        return name

    def _parse_op_name(self, node):
        return GNode(name=node.__class__.__name__, type="Op")

    def _parse_And(self, node):
        return self._parse_op_name(node)

    def _parse_Or(self, node):
        return self._parse_op_name(node)

    def _parse_Not(self, node):
        return self._parse_op_name(node)

    def _parse_Is(self, node):
        return self._parse_op_name(node)

    def _parse_Gt(self, node):
        return self._parse_op_name(node)

    def _parse_Lt(self, node):
        return self._parse_op_name(node)

    def _parse_GtE(self, node):
        return self._parse_op_name(node)

    def _parse_LtE(self, node):
        return self._parse_op_name(node)

    def _parse_Add(self, node):
        return self._parse_op_name(node)

    def _parse_Mod(self, node):
        return self._parse_op_name(node)

    def _parse_Sub(self, node):
        return self._parse_op_name(node)

    def _parse_UAdd(self, node):
        return self._parse_op_name(node)

    def _parse_USub(self, node):
        return self._parse_op_name(node)

    def _parse_Div(self, node):
        return self._parse_op_name(node)

    def _parse_Mult(self, node):
        return self._parse_op_name(node)

    def _parse_MatMult(self, node):
        return self._parse_op_name(node)

    def _parse_Pow(self, node):
        return self._parse_op_name(node)

    def _parse_FloorDiv(self, node):
        return self._parse_op_name(node)

    def _parse_RShift(self, node):
        return self._parse_op_name(node)

    def _parse_LShift(self, node):
        return self._parse_op_name(node)

    def _parse_BitXor(self, node):
        return self._parse_op_name(node)

    def _parse_BitAnd(self, node):
        return self._parse_op_name(node)

    def _parse_BitOr(self, node):
        return self._parse_op_name(node)

    def _parse_IsNot(self, node):
        return self._parse_op_name(node)

    def _parse_NotIn(self, node):
        return self._parse_op_name(node)

    def _parse_In(self, node):
        return self._parse_op_name(node)

    def _parse_Invert(self, node):
        return self._parse_op_name(node)

    def _parse_Eq(self, node):
        return self._parse_op_name(node)

    def _parse_NotEq(self, node):
        return self._parse_op_name(node)

    def _parse_Ellipsis(self, node):
        return self._parse_op_name(node)

    def _parse_Num(self, node):
        return str(node.n)

    def _parse_Str(self, node):
        return self._generic_parse(node, [])

    def _parse_Bytes(self, node):
        return repr(node.s)

    def _parse_If(self, node):
        edges, if_name = self._generic_parse(node, ["test"])
        self._parse_in_context(if_name, "if_true", edges, node.body)
        self._parse_in_context(if_name, "if_false", edges, node.orelse)
        return edges, if_name

    def _parse_For(self, node):
        edges, for_name = self._generic_parse(node, ["target", "iter"])
        self._parse_in_context(for_name, "for", edges, node.body)
        self._parse_in_context(for_name, "for_orelse", edges, node.orelse)
        return edges, for_name

    def _parse_AsyncFor(self, node):
        return self._parse_For(node)

    def _parse_Try(self, node):
        edges, try_name = self._generic_parse(node, [])
        self._parse_in_context(try_name, "try", edges, node.body)

        for h in node.handlers:
            handler_name, ext_edges = self._parse_operand(h)
            edges.extend(ext_edges)
            self._parse_in_context([handler_name], ["try_handler"], edges, h.body)
            edges.append({
                "scope": copy(self.scope[-1]), "src": handler_name, "dst": try_name, "type": 'try_except'
            })

        self._parse_in_context(try_name, "try_final", edges, node.finalbody)
        self._parse_in_context(try_name, "try_else", edges, node.orelse)

        return edges, try_name

    def _parse_While(self, node):

        edges, while_name = self._generic_parse(node, ["test"])

        self._parse_in_context([while_name], ["while"], edges, node.body)

        return edges, while_name

    def _parse_Compare(self, node):
        return self._generic_parse(node, ["left", "ops", "comparators"])

    def _parse_BoolOp(self, node):
        return self._generic_parse(node, ["values", "op"])

    def _parse_Expr(self, node):
        edges = []
        expr_name, ext_edges = self._parse_operand(node.value)
        edges.extend(ext_edges)
        return edges, expr_name

    def _parse_control_flow(self, node):
        edges = []
        ctrlflow_name = self._get_name(name="ctrl_flow", type="CtlFlowInstance", add_random_identifier=True)
        edges.append({
            "scope": copy(self.scope[-1]),
            "src": GNode(name=node.__class__.__name__, type="CtlFlow"),
            "dst": ctrlflow_name, "type": "control_flow"
        })
        return edges, ctrlflow_name

    def _parse_Continue(self, node):
        return self._parse_control_flow(node)

    def _parse_Break(self, node):
        return self._parse_control_flow(node)

    def _parse_Pass(self, node):
        return self._parse_control_flow(node)

    def _parse_Assert(self, node):
        return self._generic_parse(node, ["test", "msg"])

    def _parse_List(self, node):
        return self._generic_parse(node, ["elts"], ensure_iterables=True)

    def _parse_Tuple(self, node):
        return self._generic_parse(node, ["elts"], ensure_iterables=True)

    def _parse_Set(self, node):
        return self._generic_parse(node, ["elts"], ensure_iterables=True)

    def _parse_Dict(self, node):
        return self._generic_parse(node, ["keys", "values"], ensure_iterables=True)

    def _parse_UnaryOp(self, node):
        return self._generic_parse(node, ["operand", "op"])

    def _parse_BinOp(self, node):
        return self._generic_parse(node, ["left", "right", "op"])

    def _parse_Await(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_JoinedStr(self, node):
        joinedstr_name = GNode(name="JoinedStr_", type="JoinedStr")
        return [], joinedstr_name

    def _parse_FormattedValue(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_GeneratorExp(self, node):
        return self._generic_parse(node, ["elt", "generators"])

    def _parse_ListComp(self, node):
        return self._generic_parse(node, ["elt", "generators"])

    def _parse_DictComp(self, node):
        return self._generic_parse(node, ["key", "value", "generators"])

    def _parse_SetComp(self, node):
        return self._generic_parse(node, ["elt", "generators"])

    def _parse_Return(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_Raise(self, node):
        return self._generic_parse(node, ["exc", "cause"])

    def _parse_YieldFrom(self, node):
        return self._generic_parse(node, ["value"])

    def _parse_arguments(self, node):
        return self._generic_parse(node, ["args", "vararg"])  # kwarg, kwonlyargs, posonlyargs???

    def _parse_comprehension(self, node):
        edges = []

        cph_name = self._get_name(name="comprehension", type="comprehension", add_random_identifier=True)

        target, ext_edges = self._parse_operand(node.target)
        edges.extend(ext_edges)
        if hasattr(node.target, "lineno"):
            edges.append({
                "scope": copy(self.scope[-1]),
                "src": target,
                "dst": cph_name,
                "type": "target",
                "line": node.target.lineno - 1,
                "end_line": node.target.end_lineno - 1,
                "col_offset": node.target.col_offset,
                "end_col_offset": node.target.end_col_offset
            })
        else:
            edges.append({
                "scope": copy(self.scope[-1]),
                "src": target,
                "dst": cph_name,
                "type": "target"
            })
        edges.append({
            "scope": copy(self.scope[-1]), "src": cph_name, "dst": target, "type": "target_for"})

        iter_, ext_edges = self._parse_operand(node.iter)
        edges.extend(ext_edges)
        if hasattr(node.iter, "lineno"):
            edges.append({
                "scope": copy(self.scope[-1]),
                "src": iter_,
                "dst": cph_name,
                "type": "iter",
                "line": node.iter.lineno - 1,
                "end_line": node.iter.end_lineno - 1,
                "col_offset": node.iter.col_offset,
                "end_col_offset": node.iter.end_col_offset
            })
        else:
            edges.append({"scope": copy(self.scope[-1]), "src": iter_, "dst": cph_name, "type": "iter"})
        edges.append({"scope": copy(self.scope[-1]), "src": cph_name, "dst": iter_, "type": "iter_for"})

        for if_ in node.ifs:
            if_n, ext_edges = self._parse_operand(if_)
            edges.extend(ext_edges)
            edges.append({"scope": copy(self.scope[-1]), "src": if_n, "dst": cph_name, "type": "ifs"})
            edges.append({"scope": copy(self.scope[-1]), "src": cph_name, "dst": if_n, "type": "ifs_rev"})

        return edges, cph_name
