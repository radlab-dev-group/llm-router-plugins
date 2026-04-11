"""
Rule that masks SIM‑card ICCID numbers.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import (
    is_valid_sim_iccid,
)


class SimCardRule(BaseRule):
    """
    Detects 19‑digit ICCID numbers (may contain spaces) and masks them with
    ``{{SIM_CARD}}``.
    """

    _SIM_ICCID_REGEX = r"""
        \b
        (?:\d{4}\s?){4}\d{3}\b   # 19 digits possibly spaced every 4 chars
    """

    def __init__(self):
        super().__init__(
            regex=self._SIM_ICCID_REGEX,
            placeholder="{{SIM_CARD}}",
            flags=re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        def _replace(m: re.Match) -> str:
            self.placeholder if is_valid_sim_iccid(m.group(0)) else m.group(0)

        return re.sub(self.pattern, _replace, text)
