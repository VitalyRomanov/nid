from enum import Enum
from itertools import chain
from typing import Type

from nid.ast.graph_builder.common.definitions import PythonNodeEdgeDefinitions as PythonNodeEdgeDefinitionsCommon


class PythonNodeEdgeDefinitionsV3(PythonNodeEdgeDefinitionsCommon):
    _shared_node_types_initialized = False

    shared_node_types = None

    ast_node_type_edges = {
        "Assign": ["value", "targets"],
        "AugAssign": ["target", "op", "value"],
        "Import": ["names"],
        "alias": ["name", "asname"],
        "ImportFrom": ["module", "names"],
        "Delete": ["targets"],
        "Global": ["names"],
        "Nonlocal": ["names"],
        "withitem": ["context_expr", "optional_vars"],
        "Subscript": ["value", "slice", "ctx"],
        "Slice": ["lower", "upper", "step"],
        "ExtSlice": ["dims"],
        "Index": ["value"],
        "Starred": ["value", "ctx"],
        "Yield": ["value"],
        "ExceptHandler": ["type"],
        "Call": ["func", "args", "keywords"],
        "Compare": ["left", "ops", "comparators"],
        "BoolOp": ["values", "op"],
        "Assert": ["test", "msg"],
        "List": ["elts", "ctx"],
        "Tuple": ["elts", "ctx"],
        "Set": ["elts"],
        "UnaryOp": ["operand", "op"],
        "BinOp": ["left", "right", "op"],
        "Await": ["value"],
        "GeneratorExp": ["elt", "generators"],
        "ListComp": ["elt", "generators"],
        "SetComp": ["elt", "generators"],
        "DictComp": ["key", "value", "generators"],
        "Return": ["value"],
        "Raise": ["exc", "cause"],
        "YieldFrom": ["value"],
        "NamedExpr": ["target", "value"]
    }

    overridden_node_type_edges = {
        "Module": [],  # overridden
        # overridden, `function_name` replaces `name`, `returned_by` replaces `returns`
        "FunctionDef": ["function_name", "args", "decorator_list", "returned_by"],
        # overridden, `function_name` replaces `name`, `returned_by` replaces `returns`
        "AsyncFunctionDef": ["function_name", "args", "decorator_list", "returned_by"],
        "ClassDef": ["class_name"],  # overridden, `class_name` replaces `name`
        "AnnAssign": ["target", "value", "annotation_for"],  # overridden, `annotation_for` replaces `annotation`
        "With": ["items"],  # overridden
        "AsyncWith": ["items"],  # overridden
        "arg": ["arg", "annotation_for", "default"],  # overridden, `annotation_for` is custom
        "Lambda": ["lambda"],  # overridden
        # overridden, `if_true` renamed from `body`, `if_false` renamed from `orelse`
        "IfExp": ["test", "if_true", "if_false"],
        "keyword": ["arg", "value"],  # overridden
        "Attribute": ["value", "attr", "ctx"],  # overridden
        "Num": [],  # overridden
        "Str": [],  # overridden
        "Bytes": [],  # overridden
        "If": ["test"],  # overridden
        "For": ["target", "iter"],  # overridden
        "AsyncFor": ["target", "iter"],  # overridden
        "Try": [],  # overridden
        "While": [],  # overridden
        "Expr": ["value"],  # overridden
        "Dict": ["keys", "values"],  # overridden
        "JoinedStr": [],  # overridden
        "FormattedValue": ["value"],  # overridden
        # ["args", "vararg", "kwarg", "kwonlyargs", "posonlyargs"],  # overridden
        "arguments": ["vararg", "posonlyarg", "arg", "kwonlyarg", "kwarg"],
        # overridden, `target_for` is custom, `iter_for` is custom, `ifs_rev` is custom
        "comprehension": ["target", "iter", "ifs"],
    }

    extra_node_type_edges = {
        "mention": ["local_mention"]
    }

    context_edge_names = {
        "Module": ["defined_in_module"],
        "FunctionDef": ["defined_in_function"],
        "ClassDef": ["defined_in_class"],
        "With": ["executed_inside_with"],
        "AsyncWith": ["executed_inside_with"],
        "If": ["executed_if_true", "executed_if_false"],
        "For": ["executed_in_for", "executed_in_for_orelse"],
        "AsyncFor": ["executed_in_for", "executed_in_for_orelse"],
        "While": ["executed_in_while", "executed_while_true"],
        "Try": ["executed_in_try", "executed_in_try_final", "executed_in_try_else", "executed_in_try_except",
                "executed_with_try_handler"],
    }

    extra_edge_types = {
        "control_flow", "next", "instance", "inside"
    }

    # exceptions needed when we do not want to filter some edge types using a simple rule `_rev`
    reverse_edge_exceptions = {
        "next": "prev",
        "local_mention": None,  # from name to variable mention
        "returned_by": None,  # for type annotations
        "annotation_for": None,  # for type annotations
        "control_flow": None,  # for control flow
        "op": None,  # for operations
        "attr": None,  # for attributes
        "ctx": None,  # for context
        "default": None,  # for default value for arg
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

    ctx_nodes = {  # parse_ctx
        "Load", "Store", "Del"
    }

    # extra node types exist for keywords and attributes to prevent them from
    # getting mixed with local variable mentions
    extra_node_types = {
        "mention",
        "#keyword#",
        "#attr#",
        "astliteral",
        "type_annotation",
        "Op",
        "CtlFlow", "CtlFlowInstance", "instance", "ctx"
    }

    # @classmethod
    # def regular_node_types(cls):
    #     return set(cls.ast_node_type_edges.keys())
    #
    # @classmethod
    # def overridden_node_types(cls):
    #     return set(cls.overridden_node_type_edges.keys())

    # @classmethod
    # def node_types(cls):
    #     return list(
    #         cls.regular_node_types() |
    #         cls.overridden_node_types() |
    #         cls.iterable_nodes | cls.named_nodes | cls.constant_nodes |
    #         cls.operand_nodes | cls.control_flow_nodes | cls.extra_node_types
    #     )

    # @classmethod
    # def scope_edges(cls):
    #     return set(map(lambda x: x, chain(*cls.context_edge_names.values())))  # "defined_in_" +

    # @classmethod
    # def auxiliary_edges(cls):
    #     direct_edges = cls.scope_edges() | cls.extra_edge_types
    #     reverse_edges = cls.compute_reverse_edges(direct_edges)
    #     return direct_edges | reverse_edges

    # @classmethod
    # def compute_reverse_edges(cls, direct_edges):
    #     reverse_edges = set()
    #     for edge in direct_edges:
    #         if edge in cls.reverse_edge_exceptions:
    #             reverse = cls.reverse_edge_exceptions[edge]
    #             if reverse is not None:
    #                 reverse_edges.add(reverse)
    #         else:
    #             reverse_edges.add(edge + "_rev")
    #     return reverse_edges

    @classmethod
    def edge_types(cls):
        direct_edges = list(
            set(chain(*cls.ast_node_type_edges.values())) |
            set(chain(*cls.overridden_node_type_edges.values())) |
            set(chain(*cls.extra_node_type_edges.values())) |
            cls.scope_edges() | cls.extra_edge_types
        )

        reverse_edges = list(cls.compute_reverse_edges(direct_edges))
        return direct_edges + reverse_edges

    # @classmethod
    # def make_node_type_enum(cls) -> Type[Enum]:
    #     if not cls._node_type_enum_initialized:
    #         cls._node_type_enum = Enum("NodeTypes", " ".join(cls.node_types()))  # type: ignore
    #         cls._node_type_enum_initialized = True
    #     return cls._node_type_enum  # type: ignore

    # @classmethod
    # def make_edge_type_enum(cls) -> Type[Enum]:
    #     if not cls._edge_type_enum_initialized:
    #         cls._edge_type_enum = Enum("EdgeTypes", " ".join(cls.edge_types()))  # type: ignore
    #         cls._edge_type_enum_initialized = True
    #     return cls._edge_type_enum  # type: ignore

    @classmethod
    def _initialize_shared_nodes(cls):
        node_types_enum = cls.make_node_type_enum()
        ctx = {node_types_enum["ctx"]}
        annotation_types = {node_types_enum["type_annotation"]}
        tokenizable_types = {node_types_enum["Name"], node_types_enum["#attr#"], node_types_enum["#keyword#"]}
        python_token_types = {
            node_types_enum["Op"], node_types_enum["Constant"], node_types_enum["JoinedStr"],
            node_types_enum["CtlFlow"], node_types_enum["astliteral"]
        }

        cls.shared_node_types = annotation_types | tokenizable_types | python_token_types | ctx

        cls._shared_node_types_initialized = True

    @classmethod
    def get_shared_node_types(cls):
        if not cls._shared_node_types_initialized:
            cls._initialize_shared_nodes()
        return cls.shared_node_types

    @classmethod
    def is_shared_name_type(cls, name, type):
        if not cls._shared_node_types_initialized:
            cls._initialize_shared_nodes()

        if type in cls.shared_node_types:
            return True
        return False

    @classmethod
    def get_reverse_type(cls, type: str):
        if type.endswith("_rev"):
            return None

        if not cls._edge_type_enum_initialized:
            cls.make_edge_type_enum()

        reverse_type = cls.reverse_edge_exceptions.get(type, type + "_rev")
        if reverse_type is not None:
            assert cls._edge_type_enum is not None
            return cls._edge_type_enum[reverse_type]
        return None
