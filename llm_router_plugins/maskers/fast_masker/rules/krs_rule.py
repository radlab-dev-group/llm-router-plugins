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
from typing import Optional, Callable, Match

from .base_rule import BaseRule
from ..utils.validators import is_valid_krs


class KrsRule(BaseRule):
    """
    Detects Polish KRS numbers, validates the checksum and masks them.
    """

    # Named group ``krs`` captures the whole match (including optional
    # hyphens or spaces).  The separator may be ``-`` or a single space;
    # it is optional between each block.
    _REGEX = r"""
        \b
        (?P<krs>
            (?:\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2})   # formatted with hyphens/spaces
            |
            \d{10}                                        # plain 10‑digit string
        )
        \b
    """

    _PLACEHOLDER = "{{KRS}}"

    # Pre‑compile for performance.
    _COMPILED = re.compile(_REGEX, flags=re.IGNORECASE | re.VERBOSE)

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
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

        def _replacer(match: Match) -> str:
            raw_krs = match.group("krs")
            if is_valid_krs(raw_krs):
                replacement = (
                    "{" + anonymizer_fn(raw_krs, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid KRS – leave original text untouched.
            return raw_krs

        return self._COMPILED.sub(_replacer, text)
