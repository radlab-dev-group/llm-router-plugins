"""
Rule that masks PESEL identifiers that appear after the literal ``PESEL:``.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_valid_pesel


class PeselTaggedRule(BaseRule):
    """
    Detects patterns like ``PESEL: 44051401359`` (optional whitespace)
    and replaces the numeric part with ``{{PESEL_TAGGED}}``.
    """

    _PESEL_TAGGED_REGEX = r"""
        \bPESEL[:\s]+
        (?P<pesel>\d{11})\b
    """

    def __init__(self):
        super().__init__(
            regex=self._PESEL_TAGGED_REGEX,
            placeholder="{{PESEL_TAGGED}}",
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, List]:
        mappings = []

        def _replace(m: re.Match) -> str:
            pesel = m.group("pesel")
            if is_valid_pesel(pesel):
                # Keep the leading label, replace only the number
                if anonymizer_fn:
                    pseudo = anonymizer_fn(pesel, self.tag_type)
                    mappings.append({"original": pesel, "replacement": pseudo})
                    replacement = "{" + pseudo + "}"
                else:
                    mappings.append(
                        {"original": pesel, "replacement": self.placeholder}
                    )
                    replacement = self.placeholder
                return m.group(0).replace(pesel, replacement)
            return m.group(0)

        return re.sub(self.pattern, _replace, text), mappings
