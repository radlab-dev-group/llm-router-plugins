"""
Rule that masks Vehicle Identification Numbers (VIN).
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_vin


class VinRule(BaseRule):
    """
    Detects 17‑character VINs and masks them with ``{{VIN}}`` **only if the
    checksum (position 9) is correct**.
    """

    _VIN_REGEX = r"""
        \b
        [A-HJ-NPR-Z0-9]{17}\b
    """

    def __init__(self):
        super().__init__(
            regex=self._VIN_REGEX,
            placeholder="{{VIN}}",
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        def _replace(m: re.Match) -> str:
            vin = m.group(0)
            self.placeholder if is_valid_vin(vin) else vin

        return re.sub(self.pattern, _replace, text)
