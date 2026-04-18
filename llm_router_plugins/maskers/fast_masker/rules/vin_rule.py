"""
Rule that masks Vehicle Identification Numbers (VIN).

The rule:

1. Detects a 17‑character VIN (letters A‑H, J‑N, P‑R, Z and digits).
2. Validates the checksum (position 9) using :func:`is_valid_vin`.
3. Replaces **valid** VINs with the placeholder ``{{VIN}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is
   used instead of the default placeholder – this mirrors the behaviour of
   the ``RegonRule`` implementation.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_vin


class VinRule(BaseRule):
    """
    Detects 17‑character VINs and masks them with ``{{VIN}}`` **only if the
    checksum (position 9) is correct**.  When an ``anonymizer_fn`` is provided,
    the function is called with the original VIN and the rule’s ``tag_type``,
    and its return value is wrapped in curly braces.
    """

    # Regex that matches a 17‑character VIN surrounded by word boundaries.
    _REGEX = r"""
        \b
        [A-HJ-NPR-Z0-9]{17}\b
    """

    _PLACEHOLDER = "{{VIN}}"

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        # Compile once for fast reuse in ``apply``.
        self._compiled_regex = re.compile(
            self._REGEX, flags=re.IGNORECASE | re.VERBOSE
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
        """
        Replace each *valid* VIN occurrence with the placeholder.

        Parameters
        ----------
        text:
            The input string that may contain VINs.
        anonymizer_fn:
            Optional callable ``fn(vin: str, tag_type: str) -> str``.  If
            provided, its return value is used (wrapped in ``{}``) instead of
            ``{{VIN}}``.  This enables custom anonymisation strategies.

        Returns
        -------
        str
            The text with all valid VINs replaced.
        """

        def _replacer(match: re.Match) -> str:
            vin = match.group(0)
            if is_valid_vin(vin):
                # Use the custom anonymiser if supplied; otherwise, the default placeholder.
                replacement = (
                    "{" + anonymizer_fn(vin, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid VIN – leave it untouched.
            return vin

        return self.pattern.sub(_replacer, text)
