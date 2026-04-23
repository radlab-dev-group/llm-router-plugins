"""
Rule that masks US Social Security Numbers (SSN).

The rule:

1. Detects SSN patterns ``AAA‑GG‑SSSS`` (three digits, a hyphen, two digits,
   a hyphen, four digits).
2. Validates the number with :func:`is_valid_ssn`.
3. Replaces **valid** SSNs with the placeholder ``{{SSN}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the default placeholder – matching the behaviour of the other
   masking rules.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_valid_ssn


class SsnRule(BaseRule):
    """
    Detects US Social Security Numbers (``AAA‑GG‑SSSS``) and masks them with
    ``{{SSN}}`` after validation.
    """

    # Regex that matches the canonical SSN format.
    _REGEX = r"""
        \b
        \d{3}-\d{2}-\d{4}\b
    """

    _PLACEHOLDER = "{{SSN}}"

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.VERBOSE,
        )
        # Pre‑compile for performance.
        self._compiled_regex = re.compile(self._REGEX, flags=re.VERBOSE)

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[str, List]:
        """
        Replace each *valid* SSN occurrence with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain SSNs.
        anonymizer_fn :
            Optional callable ``fn(ssn: str, tag_type: str) -> str``.  If
            supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{SSN}}``.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            ssn = match.group(0)
            if is_valid_ssn(ssn):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(ssn, self.tag_type)
                    mappings.append({"original": ssn, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append({"original": ssn, "replacement": self.placeholder})
                return self.placeholder
            # Invalid SSN – keep original text.
            return ssn

        return self.pattern.sub(_replacer, text), mappings
