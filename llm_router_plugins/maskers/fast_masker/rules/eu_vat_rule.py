"""
Rule that masks EU VAT numbers.

The rule:

1. Detects EU VAT identifiers such as ``PL1234567890`` (two‑letter country
   code followed by 8‑12 alphanumeric characters).
2. Performs a light format check via :func:`is_valid_eu_vat`.
3. Replaces **valid** identifiers with the placeholder ``{{EU_VAT}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is
   used instead of the static placeholder – consistent with the other masking
   rules.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_eu_vat


class EuVatRule(BaseRule):
    """
    Detects EU VAT identifiers and masks them.
    """

    # Two‑letter country code followed by 8‑12 alphanumeric characters.
    _REGEX = r"""
        \b
        [A-Z]{2}[A-Z0-9]{8,12}\b
    """

    _PLACEHOLDER = "{{EU_VAT}}"

    # Pre‑compile for performance.
    _COMPILED = re.compile(_REGEX, flags=re.IGNORECASE | re.VERBOSE)

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
        """
        Replace each *valid* EU VAT occurrence with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain EU VAT numbers.
        anonymizer_fn :
            Optional callable ``fn(vat: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{EU_VAT}}``.
        """

        def _replacer(match: re.Match) -> str:
            vat = match.group(0)
            if is_valid_eu_vat(vat):
                replacement = (
                    "{" + anonymizer_fn(vat, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid VAT – keep original text.
            return vat

        return self._COMPILED.sub(_replacer, text)
