"""Time-of-day masking rule for the fast masker."""

import re

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


class TimeRule(BaseRule):
    """
    Detects time-of-day in ``HH:MM`` and ``HH.MM`` format and masks them.

    Examples of matched patterns:
        06:35, 14:30, 09.15

    The rule is context-aware — it avoids matching inside longer numeric
    strings or email addresses.
    """

    # Matches HH:MM or HH.MM (24-hour format)
    _REGEX = r"""
        \b                    # word boundary
        (?<!\d)               # not preceded by another digit
        (?!\+48\s)?           # not part of international phone prefix (+48 ...)
        ([01]\d|2[0-3])       # hour (00–23)
        [:.]                  # separator (: or .)
        [0-5]\d               # minute (00–59)
        (?!\d)                # not followed by another digit
        \b                    # word boundary
    """

    _PLACEHOLDER = "{{TIME}}"

    def __init__(self):
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )
