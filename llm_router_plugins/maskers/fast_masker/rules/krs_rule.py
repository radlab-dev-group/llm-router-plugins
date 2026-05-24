"""
Rule that masks valid Polish KRS numbers.

Supported formats:
* Plain ``1234567890`` (10 digits)
* Formatted with optional hyphens **or** spaces, e.g.
  ``123-456-78-90`` or ``123 456 78 90`` or mixed ``123-456 78-90``

The rule validates the checksum via :func:`is_valid_krs` and replaces
**valid** numbers with ``{{KRS}}``.  If an ``anonymizer_fn`` is supplied,
its result (wrapped in ``{}``) is used instead of the static placeholder.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_valid_krs


class KrsRule(BaseRule):
    """
    Detects Polish KRS numbers, validates the checksum and masks them.
    """

    _REGEX = (
        r"(?<!\w)"
        r"(?:(?:\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2})"
        r"|\d{10})"
        r"(?!\w)"
    )

    _PLACEHOLDER = "{{KRS}}"

    def __init__(self) -> None:
        super().__init__(
            regex=KrsRule._REGEX,
            placeholder=KrsRule._PLACEHOLDER,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[str, List]:
        """
        Replace each *valid* KRS occurrence with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain KRS numbers.
        anonymizer_fn :
            Optional callable ``fn(krs: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{KRS}}``.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            raw_krs = match.group(0).replace(" ", "").replace("-", "")
            if is_valid_krs(raw_krs):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(raw_krs, self.tag_type)
                    mappings.append({"original": raw_krs, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append(
                    {"original": raw_krs, "replacement": self.placeholder}
                )
                return self.placeholder
            # Invalid KRS – keep original text.
            return match.group(0)

        return self.pattern.sub(_replacer, text), mappings
