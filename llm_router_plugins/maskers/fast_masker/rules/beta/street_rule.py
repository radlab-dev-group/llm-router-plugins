"""
Rule that detects street names in Polish text.

Typical forms that are recognised (case‑insensitive, diacritics supported):

    ul. Mickiewicza 12
    ulica Marszałkowska 1A/2
    aleja Jana Pawła II 5
    plac Grunwaldzki 1
    przy Skwerze 3

The placeholder used for masking is ``{{STREET}}``.
"""

import re
from typing import Match, Tuple, List, Optional, Callable

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


class StreetNameRule(BaseRule):
    """
    Detects Polish street names, allowing optional house numbers and common
    abbreviations.
    """

    # Common street type prefixes (full word or abbreviation)
    _TYPE = r"""
        (?:ul\.?|ulica|al\.?|aleja|pl\.?|plac|skwer|
            os\.?|osiedle|rondo|droga|dr\.?|trakt|t\.?|ścieżka|ś\.?)
    """

    # Street name – 1 to 5 words max, each may contain Polish diacritics
    # Limit to prevent overly greedy matching
    _NAME = r"""
        (?:\s+[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż][A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż-]*){1,5}
    """

    # House number (e.g. 12, 12A, 12/3) - now REQUIRED when present
    _NUMBER = r"""
        (?:\s+\d+[A-Za-z]?(?:\/\d+)?)?
    """

    # Full pattern with word boundaries
    _FULL_PATTERN = rf"""
        \b                      # start of word
        {_TYPE}                 # street type (required)
        {_NAME}                 # street name (1-5 words)
        {_NUMBER}               # house number (optional but if present, must be digit)
        \b                      # end of word
    """

    _REGEX = _FULL_PATTERN
    _PLACEHOLDER = "{{STREET}}"

    def __init__(self):
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        # Pre‑compile for fast reuse.
        self._compiled_regex = re.compile(
            self._REGEX, flags=re.IGNORECASE | re.VERBOSE
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, List]:
        """
        Replace each detected street name (with optional house number) with the
        ``{{STREET}}`` placeholder.
        """
        mappings = []

        def _replacer(match: Match) -> str:
            val = match.group(0)
            if anonymizer_fn:
                pseudo = anonymizer_fn(val, self.tag_type)
                mappings.append({"original": val, "replacement": pseudo})
                return "{" + pseudo + "}"
            mappings.append({"original": val, "replacement": self.placeholder})
            return self.placeholder

        return self._compiled_regex.sub(_replacer, text), mappings
