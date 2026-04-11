"""
Rule that masks JSON Web Tokens (JWT).
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_possible_jwt


class JwtRule(BaseRule):
    """
    Detects strings that look like JWTs (three Base64URL parts separated by dots)
    and masks them with ``{{JWT}}``.
    """

    _JWT_REGEX = r"""
        \b
        [A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b
    """

    def __init__(self):
        super().__init__(
            regex=self._JWT_REGEX,
            placeholder="{{JWT}}",
            flags=re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        def _replace(m: re.Match) -> str:
            self.placeholder if is_possible_jwt(m.group(0)) else m.group(0)

        return re.sub(self.pattern, _replace, text)
