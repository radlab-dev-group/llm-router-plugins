"""
Rule that masks Polish domestic phone numbers (9 digits).
"""

import re
from typing import Optional, Callable, Tuple, List

from llm_router_plugins.maskers.fast_masker.rules.base_rule import (
    BaseRule,
    _format_pseudo,
)
from llm_router_plugins.maskers.fast_masker.utils.validators import (
    is_valid_polish_phone,
)


class PhoneRule(BaseRule):
    """
    Detects Polish domestic phone number formats (9 digits) and replaces them
    with ``{{PHONE}}``.

    Supported formats:

    * Plain:          512750525
    * Grouped by 3:   512 750 525, 512-750-525, 512-750 525
    * Arbitrary:      51 27 50 52 5 (any grouping with space/dash separators)

    Only numbers whose first two digits form a valid Polish mobile prefix
    (e.g. ``51``, ``53``, ``60``, ``79``) are masked — this reduces false
    positives from random 9‑digit sequences.
    """

    _PHONE_REGEX = r"""
        (?<!\d)
        (?:
            \b\d{9}\b                            # plain 9 consecutive digits
            |
            (?:\(0?[\s\-]?\d{2,3}\)[\s\-]?)      # area code in parentheses, e.g. (81), (081), (0-81), (22)
            (?:\d{7}|\d{2,4}(?:[\s\-\.]\d{1,4}){1,3})
            |
            # digit groups (min. 3 total, max ~15 chars)
            \d{2,5}(?:[\s\-\.]\d{1,4}){2,4}
        )
        (?!\d)
    """

    _PHONE_PLACEHOLDER = "{{PHONE}}"

    def __init__(self):
        super().__init__(
            regex=self._PHONE_REGEX,
            placeholder=self._PHONE_PLACEHOLDER,
            flags=re.VERBOSE,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[str, List]:
        """Replace only valid Polish phone numbers with the placeholder."""
        mappings = []

        def _replacer(match: re.Match) -> str:
            val = match.group(0)
            if is_valid_polish_phone(val):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(val, self.tag_type)
                    mappings.append({"original": val, "replacement": pseudo})
                    return _format_pseudo(pseudo)
                mappings.append({"original": val, "replacement": self.placeholder})
                return self.placeholder
            # Invalid phone number — keep original text.
            return val

        return self.pattern.sub(_replacer, text), mappings
