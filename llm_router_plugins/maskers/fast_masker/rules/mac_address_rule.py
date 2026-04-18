"""
Rule that masks MAC addresses.

The rule:

1. Detects MAC addresses consisting of six octets.  Separators may be a colon
   (``:``), a hyphen (``-``) or omitted entirely.
2. Validates the format with :func:`is_valid_mac`.
3. Replaces **valid** MACs with the placeholder ``{{MAC_ADDRESS}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is
   used instead of the static placeholder – consistent with the other masking
   rules.
"""

import re
from typing import Optional, Callable, Match

from .base_rule import BaseRule
from ..utils.validators import is_valid_mac


class MacAddressRule(BaseRule):
    """
    Detects MAC addresses (six octets, optional ``:`` or ``-`` separators)
    and masks them.
    """

    # Six octets, each two hex digits, optional ``:`` or ``-`` separator.
    _REGEX = r"""
        \b
        (?:[0-9A-Fa-f]{2}[:\-]?){5}[0-9A-Fa-f]{2}\b
    """

    _PLACEHOLDER = "{{MAC_ADDRESS}}"

    # Pre‑compile for speed.
    _COMPILED = re.compile(_REGEX, flags=re.VERBOSE)

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.VERBOSE,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
        """
        Replace each *valid* MAC address with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain MAC addresses.
        anonymizer_fn :
            Optional callable ``fn(mac: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{MAC_ADDRESS}}``.
        """

        def _replacer(match: Match) -> str:
            mac = match.group(0)
            if is_valid_mac(mac):
                replacement = (
                    "{" + anonymizer_fn(mac, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid MAC – leave original text unchanged.
            return mac

        return self._COMPILED.sub(_replacer, text)
