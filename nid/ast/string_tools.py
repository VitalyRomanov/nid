import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple


# TODO state explicitly in the function name that their only words with unicode strings
def get_byte_to_char_map(unicode_string: str) -> Dict[int, int]:
    """
    Generates a dictionary mapping character offsets to byte offsets for unicode_string.
    """
    response: Dict[int, int] = {}
    
    byte_offset = 0
    for char_offset, character in enumerate(unicode_string):
        response[byte_offset] = char_offset
        byte_offset += len(character.encode('utf-8'))
    response[byte_offset] = len(unicode_string)
    return response


# TODO state explicitly in the function name that their only words with unicode strings
def get_byte_to_char_map2(unicode_string: str) -> Dict[int, int]:
    """
    Generates a dictionary mapping character offsets to byte offsets for unicode_string.
    The difference from `get_byte_to_char_map` is that each byte is mapped to the character index.
    """
    response: Dict[int, int] = {}
    byte_offset = 0
    for char_offset, character in enumerate(unicode_string):
        bytes = character.encode('utf-8')
        for byte_ind, _ in enumerate(bytes):
            response[byte_offset + byte_ind] = char_offset
        byte_offset += len(bytes)
    response[byte_offset] = len(unicode_string)
    return response


def get_cum_lens(body: str, as_bytes: bool = False) -> List[int]:
    """
    Calculate the cumulative lengths of each line with respect to the beginning of
    the function's body.
    """
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line if not as_bytes else line.encode('utf8')) + cum_lens[-1] + 1)  # +1 for new line character
    return cum_lens


def to_offsets(
        body: str, entities: Iterable[Tuple[int, int, int, int, Any]], as_bytes: bool = False, 
        cum_lens: Optional[List[int]] = None, b2c: Optional[Dict[int, int]] = None
) -> List[Tuple[int, int, Any]]:
    """
    Transform entity annotation format from (line, end_line, col, end_col) returned by python parser
    to (char_ind, end_char_ind).
    
    :param body: string containing function body
    :param entities: list of tuples containing entity start- and end-offsets in bytes
    :param as_bytes: treat entity offsets as offsets for bytes. this is needed when offsets are given in bytes,
            not in str positions
    :param cum_lens: dictionary mapping from line index to the cumulative number of chars in preceding string
    :param b2c: dictionary mapping from byte index to character index
    :return: list of tuples that represent start- and end-offsets in a string that contains function body
    """
    if cum_lens is None:
        cum_lens = get_cum_lens(body, as_bytes=as_bytes)

    # TODO cleanup
    # # b2c = [get_byte_to_char_map(line) for line in body.split("\n")]
    #
    # # repl = [(cum_lens[line] + b2c[line][start], cum_lens[end_line] + b2c[end_line][end], annotation) for
    # #         ind, (line, end_line, start, end, annotation) in enumerate(entities)]
    # repl = [(cum_lens[line] + start, cum_lens[end_line] + end, annotation) for
    #         ind, (line, end_line, start, end, annotation) in enumerate(entities)]
    #
    # if as_bytes:
    #     if b2c is None:
    #         b2c = get_byte_to_char_map(body)
    #     repl = list(map(lambda x: (b2c[x[0]], b2c[x[1]], x[2]), repl))

    repl: List[Tuple[int, int, str]] = []
    for ind, (line, end_line, start, end, annotation) in enumerate(entities):
        try:
            r_ = (cum_lens[line] + start, cum_lens[end_line] + end, annotation)
            if as_bytes:
                if b2c is None:
                    b2c = get_byte_to_char_map(body)
                r_ = (b2c[r_[0]], b2c[r_[1]], r_[2])
            repl.append(r_)
        except KeyError:
            logging.warning("Skipping offset, does not align with the source code")
            continue

    return repl


def string_hash(str_: str) -> str:
    return hashlib.md5(str_.encode('utf-8')).hexdigest()
