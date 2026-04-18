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
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_possible_transaction_ref


class TransactionRefRule(BaseRule):
    """
    Detects transaction reference strings and masks them.
    """

    # Matches IDs like TRX-20231125-001 or ABCD_20240101_1234.
    _REGEX = r"""
        \b
        [A-Z]{2,5}[-_]\d{4,8}[-_]\d{3,6}\b
    """

    _PLACEHOLDER = "{{TRANSACTION_REF}}"

    # Pre-compile for performance.
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

        def _replacer(match: re.Match) -> str:
            ref = match.group(0)
            if is_possible_transaction_ref(ref):
                replacement = (
                    "{" + anonymizer_fn(ref, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid reference – keep original text.
            return ref

        return self._COMPILED.sub(_replacer, text)
