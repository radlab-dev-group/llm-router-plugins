"""
Rule that masks Polish car registration plates.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import (
    is_valid_car_plate,
)


class CarPlateRule(BaseRule):
    """
    Detects Polish vehicle plates (e.g. ``ABC 12345``) and masks them with
    ``{{CAR_PLATE}}`` after a permissive validation.
    """

    _CAR_PLATE_REGEX = r"""
        \b
        [A-Z]{2,3}\s?\d{2,5}[A-Z]{0,2}\b
    """

    def __init__(self):
        super().__init__(
            regex=self._CAR_PLATE_REGEX,
            placeholder="{{CAR_PLATE}}",
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        def _replace(m: re.Match) -> str:
            self.placeholder if is_valid_car_plate(m.group(0)) else m.group(0)

        return re.sub(self.pattern, _replace, text)
