"""
Rule that masks Polish NRB (bank-account) numbers.

The rule:

1. Detects NRB numbers – 26 digits, optionally separated by spaces in the
   typical "2-4-4-4-4-4-4" grouping.
2. Validates the number with :func:`is_valid_nrb`.
3. Replaces **valid** NRBs with the placeholder ``{{NRB}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the static placeholder – consistent with the other masking rules.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_valid_nrb


class NrbRule(BaseRule):
    """
    Detects Polish NRB numbers, validates the checksum and masks them.
    """

    _REGEX = (
        r"(?<!\w)"
        r"(?:(?:\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4})"
        r"|\d{26})"
        r"(?!\w)"
    )

    _PLACEHOLDER = "{{NRB}}"

    def __init__(self) -> None:
        super().__init__(
            regex=NrbRule._REGEX,
            placeholder=NrbRule._PLACEHOLDER,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[str, List]:
        """
        Replace each *valid* NRB occurrence with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain NRB numbers.
        anonymizer_fn :
            Optional callable ``fn(nrb: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{NRB}}``.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            raw_nrb = match.group(0).replace(" ", "")
            if is_valid_nrb(raw_nrb):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(raw_nrb, self.tag_type)
                    mappings.append({"original": raw_nrb, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append(
                    {"original": raw_nrb, "replacement": self.placeholder}
                )
                return self.placeholder
            # Invalid NRB – keep original text.
            return match.group(0)

        return self.pattern.sub(_replacer, text), mappings
