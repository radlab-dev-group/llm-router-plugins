"""
Rule that masks Polish postal codes.

Polish postal codes have the form ``dd-ddd`` (two digits, a hyphen,
three digits).  In informal text the hyphen is sometimes omitted
(``ddddd``).  The rule also recognises the code when it is wrapped in
markdown emphasis characters (`_` or `*`).

Valid matches are replaced with the placeholder ``{{POSTAL_CODE}}``.
"""

import re
from typing import Optional, Callable, Match, Tuple, List

from .base_rule import BaseRule


class PostalCodeRule(BaseRule):
    """
    Detects Polish postal codes in the following variants:

    * ``12-345``
    * ``12345`` (without the hyphen)
    * ``_12-345_`` or ``*12-345*`` – markdown emphasis

    The rule does **not** perform any checksum validation – the format is
    sufficient for a postal code.
    """

    # The regex captures the numeric part (with optional hyphen) in a named group
    # ``code``.  Optional leading/trailing markdown markers (`_` or `*`) are
    # allowed, but they are not part of the captured group.
    _POSTAL_REGEX = r"""
        (?<!\d)                     # not preceded by another digit
        (?:[_*]+)?                  # optional leading markdown markers
        (?P<code>
            \d{2}-\d{3}            # two digits, optional hyphen, three digits
        )
        (?:[_*]+)?                  # optional trailing markdown markers
        (?!\d)                      # not followed by another digit
    """

    _PLACEHOLDER = "{{POSTAL_CODE}}"

    def __init__(self):
        super().__init__(
            regex=self._POSTAL_REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        # Pre‑compile for speed
        self._compiled_regex = re.compile(
            self._POSTAL_REGEX, flags=re.IGNORECASE | re.VERBOSE
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, List]:
        """
        Replace each detected postal code with the placeholder.
        """
        mappings = []

        def _replacer(match: Match) -> str:
            val = match.group(0)
            if anonymizer_fn:
                pseudo = anonymizer_fn(val, self.tag_type)
                mappings.append({"original": val, "replacement": pseudo})
                return "{" + pseudo + "}"
            mappings.append({"original": val, "replacement": self._PLACEHOLDER})
            return self._PLACEHOLDER

        return self._compiled_regex.sub(_replacer, text), mappings
