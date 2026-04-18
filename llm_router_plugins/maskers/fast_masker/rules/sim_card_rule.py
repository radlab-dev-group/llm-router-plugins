"""
Rule that masks SIM-card ICCID numbers.

The rule:

1. Detects 19-digit ICCID numbers, optionally separated by spaces.
2. Validates the number with :func:`is_valid_sim_iccid`.
3. Replaces **valid** ICCIDs with the placeholder ``{{SIM_CARD}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the static placeholder, consistent with the other masking rules.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_valid_sim_iccid


class SimCardRule(BaseRule):
    """
    Detects 19-digit ICCID numbers and masks them.
    """

    # Matches 19 digits, optionally spaced every 4 chars:
    # e.g. 1234 5678 9012 3456 789
    _REGEX = r"""
        \b
        (?:\d{4}\s?){4}\d{3}\b
    """

    _PLACEHOLDER = "{{SIM_CARD}}"

    # Pre-compile for performance.
    _COMPILED = re.compile(_REGEX, flags=re.VERBOSE)

    def __init__(self) -> None:
        super().__init__(
            regex=self._REGEX,
            placeholder=self._PLACEHOLDER,
            flags=re.VERBOSE,
        )

    def apply(
        self,
        text: str,
        anonymizer_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[str, List]:
        """
        Replace each *valid* ICCID occurrence with the placeholder.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            iccid = match.group(0)
            if is_valid_sim_iccid(iccid):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(iccid, self.tag_type)
                    mappings.append({"original": iccid, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append({"original": iccid, "replacement": self.placeholder})
                return self.placeholder
            return iccid

        return self._COMPILED.sub(_replacer, text), mappings
