"""
Rule that masks valid Polish PESEL numbers.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_pesel


class PeselRule(BaseRule):
    """
    Detects 11‑digit PESEL numbers, validates the checksum and replaces only
    the valid ones with ``{{PESEL}}``.
    """

    REGEX = r"(?<!\w)(?:[_*]+)?(?P<pesel>\d{11})(?:[_*]+)?(?!\w)"

    _MASK_TAG_PLACEHOLDER = "{{PESEL}}"

    _PESEL_REGEX = re.compile(REGEX)

    def __init__(self):
        super().__init__(
            regex=PeselRule.REGEX,
            placeholder=PeselRule._MASK_TAG_PLACEHOLDER,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        """
        Replace each *valid* PESEL occurrence with the placeholder or pseudonym.

        Invalid PESEL strings (wrong checksum) are left untouched.
        """

        def replacer(match: re.Match) -> str:
            pesel = match.group("pesel")
            if is_valid_pesel(pesel):
                replacement = (
                    anonymizer_fn(pesel, self.tag_type)
                    if anonymizer_fn
                    else self.placeholder
                )
                # We replace the numeric part inside the full match (including markdown)
                full_match = match.group(0)
                return full_match.replace(pesel, replacement)
            return match.group(0)

        return self.pattern.sub(replacer, text)
