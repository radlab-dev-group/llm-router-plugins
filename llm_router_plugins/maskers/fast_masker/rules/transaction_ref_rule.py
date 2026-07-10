"""
Rule that masks generic transaction reference strings.

The rule:

1. Detects strings that look like transaction IDs, such as
   ``TRX-20231125-001``.
2. Validates the reference with :func:`is_possible_transaction_ref`.
3. Replaces **valid** references with the placeholder
   ``{{TRANSACTION_REF}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the static placeholder, consistent with the other masking rules.
"""

import re
from typing import Optional, Callable, Tuple, List

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule
from llm_router_plugins.maskers.fast_masker.utils.validators import (
    is_possible_transaction_ref,
)


class TransactionRefRule(BaseRule):
    """
    Detects transaction reference strings and masks them.
    """

    # Matches IDs like TRX-20231125-001 or ABCD_20240101_1234.
    # Negative lookahead blocks country-code + 2-digit check + separator patterns (IBAN signature).
    _REGEX = r"""
        (?<![A-Z]{2}-)            # not after "LL-" (IBAN CC + dash — end of IBAN)
        (?<![A-Z]{2}\d)           # not after "LL\d" (compact IBAN like "PL17...")
        \b
        [A-Z]{2,5}                # prefix (e.g. "TRX", "ORD", etc.)
        [-_]
        (?!\d{2}[-_])             # negative lookahead: not 2 check digits + sep (IBAN signature)
        \d{4,8}                   # date-like number
        [-_]
        \d{3,6}                   # reference number
        \b
    """

    _PLACEHOLDER = "{{TRANSACTION_REF}}"

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
        Replace each *valid* transaction reference with the placeholder.

        Parameters
        ----------
        text:
            Input string that may contain transaction references.
        anonymizer_fn:
            Optional callable ``fn(ref: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{TRANSACTION_REF}}``.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            ref = match.group(0)
            if is_possible_transaction_ref(ref):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(ref, self.tag_type)
                    mappings.append({"original": ref, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append({"original": ref, "replacement": self.placeholder})
                return self.placeholder
            # Invalid reference – keep original text.
            return ref

        return self.pattern.sub(_replacer, text), mappings
