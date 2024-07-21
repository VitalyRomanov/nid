from os import urandom
from time import time_ns


class IdentifierPool:
    """
    Creates identifier that is almost guaranteed to be unique. Beginning of identifier is based on
    current time, and the tail of identifier is randomly generated.
    """
    def __init__(self):
        self._used_identifiers = set()

    @staticmethod
    def _get_candidate() -> str:
        return str(hex(int(time_ns())))[:12] + str(urandom(3).hex())

    def get_new_identifier(self) -> str:
        candidate = self._get_candidate()
        while candidate in self._used_identifiers:
            candidate = self._get_candidate()
        self._used_identifiers.add(candidate)
        return candidate


class IntIdentifierPool(IdentifierPool):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_candidate() -> str:
        candidate = str(int((str(hex(int(time_ns())))[:12] + str(urandom(3).hex())), 16))
        return candidate
