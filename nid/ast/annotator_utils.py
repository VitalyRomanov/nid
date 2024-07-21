from copy import copy
from typing import Iterable, List, Tuple


def adjust_offsets(offsets: Iterable[Tuple[int, int, str]], amount: int) -> List[Tuple[int, int, str]]:
    """
    Adjust offset by subtracting certain amount from the start and end positions
    :param offsets: iterable with offsets
    :param amount: adjustment amount
    :return: list of adjusted offsets
    """
    if amount == 0:
        return list(offsets)
    return [(offset[0] - amount, offset[1] - amount, offset[2]) for offset in offsets]


def adjust_offsets2(offsets: Iterable[Tuple[int, int, str]], amount: int) -> List[Tuple[int, int, str]]:
    """
    Adjust offset by adding certain amount to the start and end positions
    :param offsets: iterable with offsets
    :param amount: adjustment amount
    :return: list of adjusted offsets
    """
    return [(offset[0] + amount, offset[1] + amount, offset[2]) for offset in offsets]


def overlap(p1: Tuple, p2: Tuple) -> bool:
    """
    Check whether two entities defined by (start_position, end_position) overlap
    :param p1: tuple for the first entity
    :param p2: tuple for the second entity
    :return: boolean flag whether two entities overlap
    """
    if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
        return True
    else:
        return False


def resolve_self_collision(offsets: Iterable[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    no_collisions = []

    for ind_1, offset_1 in enumerate(offsets):
        # keep first
        if any(map(lambda x: overlap(offset_1, x), no_collisions)):
            pass
        else:
            no_collisions.append(offset_1)
        # new = []
        # evict = []
        #
        # for ind_2, offset_2 in enumerate(no_collisions):
        #     if overlap(offset_1, offset_2):
        #         # keep smallest
        #         if (offset_1[1] - offset_1[0]) <= (offset_2[1] - offset_2[0]):
        #             evict.append(ind_2)
        #             new.append(offset_1)
        #         else:
        #             pass
        #     else:
        #         new.append(offset_1)
        #
        # for ind in sorted(evict, reverse=True):
        #     no_collisions.pop(ind)
        #
        # no_collisions.extend(new)

    return no_collisions


def resolve_self_collisions2(offsets: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Resolve self collision in favour of the smallest entity.
    :param offsets:
    :return:
    """
    offsets = copy(offsets)
    no_collisions = []

    while len(offsets) > 0:
        offset_1 = offsets.pop(0)
        evict = []
        new = []

        add = True
        for ind_2, offset_2 in enumerate(no_collisions):
            if overlap(offset_1, offset_2):
                # keep smallest
                if (offset_1[1] - offset_1[0]) <= (offset_2[1] - offset_2[0]):
                    evict.append(ind_2)
                    new.append(offset_1)
                else:
                    pass
                add = False

        if add:
            new.append(offset_1)

        for ind in sorted(evict, reverse=True):
            no_collisions.pop(ind)

        no_collisions.extend(new)

    no_collisions = list(set(no_collisions))

    return no_collisions


# TODO need to revive this function
# def align_tokens_with_graph(doc, spans, tokenizer_name: str):
#     spans = deepcopy(spans)
#     if tokenizer_name == "codebert":
#         backup_tokens = doc
#         doc, adjustment = codebert_to_spacy(doc)
#         spans = adjust_offsets(spans, adjustment)

#     node_tags = biluo_tags_from_offsets(doc, spans, no_localization=False)

#     if tokenizer_name == "codebert":
#         doc = ["<s>"] + [t.text for t in backup_tokens] + ["</s>"]
#     return doc, node_tags


# TODO need to revive this function
# def source_code_graph_alignment(source_codes, node_spans, tokenizer="codebert"):
#     supported_tokenizers = ["spacy", "codebert"]
#     assert tokenizer in supported_tokenizers, f"Only these tokenizers supported for alignment: {supported_tokenizers}"
#     nlp = create_tokenizer(tokenizer)

#     for code, spans in zip(source_codes, node_spans):
#         yield align_tokens_with_graph(nlp(code), resolve_self_collisions2(spans), tokenizer_name=tokenizer)


# TODO consider removing
# def map_offsets(column, id_map):
#     def map_entry(entry):
#         return [(e[0], e[1], id_map[e[2]]) for e in entry]
#     return [map_entry(entry) for entry in column]
