"""
Rule that masks MAC addresses.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_mac


class MacAddressRule(BaseRule):
    """
    Detects MAC addresses (six octets, optional ``:`` or ``-`` separators) and
    replaces them with ``{{MAC_ADDRESS}}``.
    """

    _MAC_REGEX = r"""
        \b
        (?:[0-9A-Fa-f]{2}[:\-]?){5}[0-9A-Fa-f]{2}\b
    """

    def __init__(self):
        super().__init__(
            regex=self._MAC_REGEX,
            placeholder="{{MAC_ADDRESS}}",
            flags=re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        def _replace(m: re.Match) -> str:
            self.placeholder if is_valid_mac(m.group(0)) else m.group(0)

        return re.sub(self.pattern, _replace, text)
