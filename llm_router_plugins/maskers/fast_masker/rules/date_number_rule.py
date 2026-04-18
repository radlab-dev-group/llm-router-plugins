"""
Rule that masks dates written in numeric form.

Supported formats (separator may be ``.``, ``-``, ``/`` or whitespace‑wrapped
variants):

* ``YYYY.MM.DD``   – year‑month‑day
* ``DD.MM.YYYY``   – day‑month‑year

All matches are replaced with ``{{DATE}}``.  If an ``anonymizer_fn`` is
provided, its result (wrapped in ``{}``) is used instead of the static
placeholder, matching the behaviour of the other masking rules.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule


class DateNumberRule(BaseRule):
    """
    Detects numeric dates and masks them.
    """

    # Regex with two alternatives:
    #   1) YYYY‑MM‑DD
    #   2) DD‑MM‑YYYY
    # Separators: dot, dash, slash, optionally surrounded by whitespace.
    _REGEX = (
        r"(?<!\d)"  # not preceded by a digit
        r"(?:"
        # 1) YYYY‑MM‑DD
        r"(?P<year>\d{4})\s*[-./]\s*"
        r"(?P<month>0[1-9]|1[0-2])\s*[-./]\s*"
        r"(?P<day>0[1-9]|[12]\d|3[01])"
        r"|"
        # 2) DD‑MM‑YYYY
        r"(?P<day2>0[1-9]|[12]\d|3[01])\s*[-./]\s*"
        r"(?P<month2>0[1-9]|1[0-2])\s*[-./]\s*"
        r"(?P<year2>\d{4})"
        r")"
        r"(?!\d)"  # not followed by a digit
    )

    _PLACEHOLDER = "{{DATE}}"

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )
