from enum import Enum
from itertools import chain

from nid.ast.graph_builder import PythonNodeEdgeDefinitions


class PythonNodeEdgeDefinitionsV2(PythonNodeEdgeDefinitions):
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
        "Subscript": ["value", "slice"],
        "Slice": ["lower", "upper", "step"],
        "ExtSlice": ["dims"],
        "Index": ["value"],
        "Starred": ["value"],
        "Yield": ["value"],
        "ExceptHandler": ["type"],
        "Call": ["func", "args", "keywords"],
        "Compare": ["left", "ops", "comparators"],
        "BoolOp": ["values", "op"],
        "Assert": ["test", "msg"],
        "List": ["elts"],
        "Tuple": ["elts"],
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
        "Lambda": [],  # overridden
        # overridden, `if_true` renamed from `body`, `if_false` renamed from `orelse`
        "IfExp": ["test", "if_true", "if_false"],
        "keyword": ["arg", "value"],  # overridden
        "Attribute": ["value", "attr"],  # overridden
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
        # overridden, `target_for` is custom, `iter_for` is custom `ifs_rev` is custom
        "comprehension": ["target", "iter", "ifs"],
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
        "control_flow", "next", "local_mention",
    }

    # exceptions needed when we do not want to filter some edge types using a simple rule `_rev`
    reverse_edge_exceptions = {
        # "target": "target_for",
        # "iter": "iter_for",  # mainly used in comprehension
        # "ifs": "ifs_for",  # mainly used in comprehension
        "next": "prev",
        "local_mention": None,  # from name to variable mention
        "returned_by": None,  # for type annotations
        "annotation_for": None,  # for type annotations
        "control_flow": None,  # for control flow
        "op": None,  # for operations
        "attr": None,  # for attributes
        # "arg": None  # for keywords ???
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

    # extra node types exist for keywords and attributes to prevent them from
    # getting mixed with local variable mentions
    extra_node_types = {
        "#keyword#",
        "#attr#"
    }

    @classmethod
    def edge_types(cls):
        direct_edges = list(
            set(chain(*cls.ast_node_type_edges.values())) |
            set(chain(*cls.overridden_node_type_edges.values())) |
            cls.scope_edges() |
            cls.extra_edge_types | cls.named_nodes | cls.constant_nodes |
            cls.operand_nodes | cls.control_flow_nodes | cls.extra_node_types
        )

        reverse_edges = list(cls.compute_reverse_edges(direct_edges))
        return direct_edges + reverse_edges


# class PythonSharedNodes:
#     annotation_types = {"type_annotation", "returned_by"}
#     tokenizable_types = {"Name", "#attr#", "#keyword#"}
#     python_token_types = {"Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal"}
#     subword_types = {'subword'}
#
#     subword_leaf_types = annotation_types | subword_types | python_token_types
#     named_leaf_types = annotation_types | tokenizable_types | python_token_types
#     tokenizable_types_and_annotations = annotation_types | tokenizable_types
#
#     shared_node_types = annotation_types | subword_types | tokenizable_types | python_token_types
#
#     @classmethod
#     def is_shared(cls, node):
#         # nodes that are of stared type
#         # nodes that are subwords of keyword arguments
#         return cls.is_shared_name_type(node.name, node.type)
#
#     @classmethod
#     def is_shared_name_type(cls, name, type):
#         if type in cls.shared_node_types or \
#                 (type == "subword_instance" and "0x" not in name):
#             return True
#         return False
