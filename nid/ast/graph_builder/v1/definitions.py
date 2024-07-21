# TODO consider deprecating
from enum import Enum


class PythonSyntheticNodeTypes(Enum):  # TODO NOT USED
    NAME = 1  # "Name"
    MENTION = 2  # "mention"
    AST_LITERAL = 3  # "ast_Literal"
    TYPE_ANNOTATION = 4  # "type_annotation"
    NAME_CONSTANT = 5  # "NameConstant"
    CONSTANT = 6  # "Constant"
    OP = 7  # "Op"
    CTL_FLOW_INSTANCE = 8  # "CtlFlowInstance"
    CTL_FLOW = 9  # "CtlFlow"
    JOINED_STR = 10  # "JoinedStr"
    COMPREHENSION = 11  # "comprehension"
    KEYWORD_PROP = 12  # "#keyword#"
    ATTR_PROP = 13  # "#attr#"


class PythonSyntheticEdgeTypes:
    subword_instance = "subword_instance"
    next = "next"
    prev = "prev"
    # depends_on_ = "depends_on_"
    execute_when_ = "execute_when_"
    local_mention = "local_mention"
    mention_scope = "mention_scope"
    returned_by = "returned_by"
    # TODO
    #  make sure every place uses function_name and not fname
    fname = "function_name"
    # fname_for = "fname_for"
    annotation_for = "annotation_for"
    control_flow = "control_flow"


class PythonSharedNodes:
    annotation_types = {"type_annotation", "returned_by"}
    tokenizable_types = {"Name", "#attr#", "#keyword#"}
    python_token_types = {"Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal"}
    subword_types = {'subword'}

    subword_leaf_types = annotation_types | subword_types | python_token_types
    named_leaf_types = annotation_types | tokenizable_types | python_token_types
    tokenizable_types_and_annotations = annotation_types | tokenizable_types

    shared_node_types = annotation_types | subword_types | tokenizable_types | python_token_types

    # leaf_types = {'subword', "Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal", "Name", "type_annotation", "returned_by"}
    # shared_node_types = {'subword', "Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal", "Name", "type_annotation", "returned_by", "#attr#", "#keyword#"}

    @classmethod
    def is_shared(cls, node):
        # nodes that are of stared type
        # nodes that are subwords of keyword arguments
        return cls.is_shared_name_type(node.name, node.type)

    @classmethod
    def is_shared_name_type(cls, name, type):
        if type in cls.shared_node_types or \
                (type == "subword_instance" and "0x" not in name):
            return True
        return False
