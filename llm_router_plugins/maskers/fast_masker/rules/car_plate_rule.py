"""
Rule that masks Polish car registration plates.

The rule:

1. Detects Polish car plates (e.g. ``ABC 12345`` or ``AB 123 CD``) using a
   permissive regular expression.
2. Validates the plate with :func:`is_valid_car_plate`.
3. Replaces **valid** plates with the placeholder ``{{CAR_PLATE}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the default placeholder – mirroring the behaviour of the VIN
   and REGON rules.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_car_plate


class CarPlateRule(BaseRule):
    """
    Detects Polish vehicle plates (e.g. ``ABC 12345``) and masks them with
    ``{{CAR_PLATE}}`` after a permissive validation.
    """

    # Regex that matches typical Polish plates:
    #   - 2‑3 letters
    #   - optional space
    #   - 2‑5 digits
    #   - optional trailing 0‑2 letters
    _REGEX = r"""
        \b
        [A-Z]{2,3}\s?\d{2,5}[A-Z]{0,2}\b
    """

    _PLACEHOLDER = "{{CAR_PLATE}}"

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        # Pre‑compile for performance.
        self._compiled_regex = re.compile(
            self._REGEX, flags=re.IGNORECASE | re.VERBOSE
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
        """
        Replace each *valid* car‑plate occurrence with the placeholder.

        Parameters
        ----------
        text :
            Input string that may contain car plates.
        anonymizer_fn :
            Optional callable ``fn(plate: str, tag_type: str) -> str``.  If
            supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{CAR_PLATE}}``.
        """

        def _replacer(match: re.Match) -> str:
            plate = match.group(0)
            if is_valid_car_plate(plate):
                replacement = (
                    "{" + anonymizer_fn(plate, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid plate – keep original text.
            return plate

        return self.pattern.sub(_replacer, text)
