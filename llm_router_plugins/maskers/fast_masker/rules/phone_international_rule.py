"""
Rule that masks international phone numbers (with country code prefix +).
"""

import re

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


class PhoneInternationalRule(BaseRule):
    """
    Detects international phone numbers starting with ``+`` followed by a
    1-3 digit country code and subscriber digits (e.g. ``+48512750525``,
    ``+48 512 750 525``) and masks them with ``{{PHONE_INTERNATIONAL}}``.

    Supports flexible digit groupings:
        - Continuous:     +48512750525
        - Spaced groups:  +48 512 750 525, +48 314 2343
        - Mixed dashes:   +48 512-750-525
    """

    _PHONE_INTL_REGEX = r"""
        (?<!\S)\+             # leading + (not preceded by non-whitespace)
        \d{1,3}              # country code (1-3 digits)
        (?:[\s\-]?\d{1,4}){2,5}  # 2-5 groups of subscriber digits with optional separator
        (?!\d)               # not followed by another digit (allows trailing punctuation like ) , . etc.)
    """

    def __init__(self):
        super().__init__(
            regex=self._PHONE_INTL_REGEX,
            placeholder="{{PHONE_INTERNATIONAL}}",
            flags=re.VERBOSE,
        )
