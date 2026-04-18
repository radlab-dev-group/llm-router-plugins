"""
Rule that masks dates written in words (Polish & English).

The rule:

1. Detects textual dates such as:
   * Polish – ``DD miesiąc YYYY`` or ``YYYY miesiąc DD`` (genitive month
     forms and common abbreviations).
   * English – ``Month DD, YYYY`` or ``DD Month YYYY`` (full names,
     three‑letter abbreviations, optional ordinal suffixes and commas).
2. Replaces **any** recognised textual date with the placeholder
   ``{{DATE}}``.
3. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is
   used instead of the default placeholder – mirroring the behaviour of
   the other masking rules.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule


class DateWordRule(BaseRule):
    """
    Detects dates written in words (Polish & English) and masks them.
    """

    # ----- month name alternatives (Polish & English) -------------------------
    _PL_MONTHS = (
        "styczeń|stycznia|sty|"
        "luty|lutego|lut|"
        "marzec|marca|mar|"
        "kwiecień|kwietnia|kwi|"
        "maj|maja|"
        "czerwiec|czerwca|cze|"
        "lipiec|lipca|lip|"
        "sierpień|sierpnia|sie|"
        "wrzesień|września|wrz|"
        "październik|października|paź|"
        "listopad|listopada|lis|"
        "grudzień|grudnia|gru"
    )
    _EN_MONTHS = (
        "January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|"
        "August|Aug|September|Sep|October|Oct|November|Nov|December|Dec"
    )

    # ----- regular expression -------------------------------------------------
    # Two main alternatives: Polish and English.  Each side supports both
    # “day‑month‑year” and “year‑month‑day” order.
    REGEX = (
        r"(?<!\w)"  # left word boundary
        r"(?:"
        # ---------- Polish ----------------------------------------------------
        r"(?:"  # 1a) DD month YYYY
        r"(?P<pl_day>\d{1,2})\s+"
        r"(?P<pl_month>" + _PL_MONTHS + r")\s+"
        r"(?P<pl_year>\d{4})"
        r")|(?:"
        r"(?P<pl_year2>\d{4})\s+"
        r"(?P<pl_month2>" + _PL_MONTHS + r")\s+"
        r"(?P<pl_day2>\d{1,2})"
        r")"
        r"|"
        # ---------- English --------------------------------------------------
        r"(?:"  # 2a) Month DD, YYYY
        r"(?P<en_month>" + _EN_MONTHS + r")\s+"
        r"(?P<en_day>\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,\s*|\s+)"  # optional comma
        r"(?P<en_year>\d{4})"
        r")|(?:"
        r"(?P<en_day2>\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(?P<en_month2>" + _EN_MONTHS + r")\s+"
        r"(?P<en_year2>\d{4})"
        r")"
        r")"
        r"(?!\w)"  # right word boundary
    )

    _PLACEHOLDER = "{{DATE}}"

    # Compile once for speed.
    _COMPILED = re.compile(REGEX, flags=re.IGNORECASE | re.VERBOSE)

    def __init__(self) -> None:
        super().__init__(
            regex=self.REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
        """
        Replace any recognised textual date with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain textual dates.
        anonymizer_fn :
            Optional callable ``fn(date_str: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead
            of ``{{DATE}}``.
        """

        def _replacer(match: re.Match) -> str:
            original = match.group(0)
            if anonymizer_fn:
                # Custom anonymisation – wrap the result in curly braces.
                return "{" + anonymizer_fn(original, self.tag_type) + "}"
            # Default behaviour – static placeholder.
            return self.placeholder

        return self._COMPILED.sub(_replacer, text)
