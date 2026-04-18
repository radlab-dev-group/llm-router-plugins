"""
Rule that masks credit‑card numbers.

The rule:

1. Detects 13‑ to 19‑digit credit‑card numbers (optionally separated by
   spaces or dashes).
2. Validates the number with the Luhn algorithm via
   :func:`is_valid_credit_card`.
3. Replaces **valid** numbers with the placeholder ``{{CREDIT_CARD}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is
   used instead of the static placeholder – consistent with the other
   masking rules.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_valid_credit_card


class CreditCardRule(BaseRule):
    """
    Detects credit‑card numbers and masks them.
    """

    # Regex captures 13‑19 digits, allowing optional spaces or dashes between groups.
    _REGEX = r"""
        \b
        \d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}   # 4‑4‑4‑(1‑7) grouping
        \b
    """

    _PLACEHOLDER = "{{CREDIT_CARD}}"

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
    ) -> Tuple[str, List]:
        """
        Replace each *valid* credit‑card occurrence with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain credit‑card numbers.
        anonymizer_fn :
            Optional callable ``fn(card_number: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{CREDIT_CARD}}``.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            candidate = match.group(0)
            if is_valid_credit_card(candidate):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(candidate, self.tag_type)
                    mappings.append({"original": candidate, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append(
                    {"original": candidate, "replacement": self.placeholder}
                )
                return self.placeholder
            # Invalid number – leave it unchanged.
            return candidate

        return self._COMPILED.sub(_replacer, text), mappings
