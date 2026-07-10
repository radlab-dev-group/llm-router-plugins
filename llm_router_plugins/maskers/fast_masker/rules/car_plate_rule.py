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
from typing import Optional, Callable, Tuple, List

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule
from llm_router_plugins.maskers.fast_masker.utils.validators import (
    is_valid_car_plate,
)


class CarPlateRule(BaseRule):
    """
    Detects Polish vehicle plates (e.g. ``ABC 12345``) and masks them with
    ``{{CAR_PLATE}}`` after a permissive validation.
    """

    # Regex that matches typical Polish plates (loose pattern; strict validation
    # is delegated to :func:`is_valid_car_plate` in validators.py).
    _REGEX = r"""
        (?<![A-Z]{2})       # not preceded by two uppercase letters (blocks IBAN CC like "PL" in compact)
        (?<![A-Z]{2}\s)     # not after "LL " (spaced IBAN like "PL 17...")
        (?<![A-Z]{2}-)      # not after "LL-" (dashed IBAN like "PL-17...")
        \b
        [A-Z]{2,3}          # leading letters
        (?:                 # followed by plate content: digits and optional trailing letters
            (?![\s\d]*[A-Za-z0-9])  # not followed by whitespace+digit then alnum — blocks spaced IBAN country codes
            \d{2,5}[A-Z]{0,2}\b    # standard car plate: digits + optional trailing letters
        )
    """

    _PLACEHOLDER = "{{CAR_PLATE}}"

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
    ) -> Tuple[str, List]:
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
        mappings = []

        def _replacer(match: re.Match) -> str:
            plate = match.group(0)
            if is_valid_car_plate(plate):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(plate, self.tag_type)
                    mappings.append({"original": plate, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append({"original": plate, "replacement": self.placeholder})
                return self.placeholder
            # Invalid plate – keep original text.
            return plate

        return self.pattern.sub(_replacer, text), mappings
