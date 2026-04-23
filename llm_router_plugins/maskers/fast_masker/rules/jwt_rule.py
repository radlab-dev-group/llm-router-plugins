"""
Rule that masks JSON Web Tokens (JWT).

The rule:

1. Detects strings that look like JWTs — three Base64URL parts separated by dots.
2. Validates the token structure with :func:`is_possible_jwt`.
3. Replaces **valid** JWTs with the placeholder ``{{JWT}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the static placeholder, consistent with the other masking rules.
"""

import re
from typing import Optional, Callable, Tuple, List

from .base_rule import BaseRule
from ..utils.validators import is_possible_jwt


class JwtRule(BaseRule):
    """
    Detects strings that look like JWTs and masks them.
    """

    # Three Base64URL parts separated by dots.
    _REGEX = r"""
        \b
        [A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b
    """

    _PLACEHOLDER = "{{JWT}}"

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
        Replace each *valid* JWT occurrence with the placeholder.

        Parameters
        ----------
        text:
            Input string that may contain JWTs.
        anonymizer_fn:
            Optional callable ``fn(jwt: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{JWT}}``.
        """
        mappings = []

        def _replacer(match: re.Match) -> str:
            token = match.group(0)
            if is_possible_jwt(token):
                if anonymizer_fn:
                    pseudo = anonymizer_fn(token, self.tag_type)
                    mappings.append({"original": token, "replacement": pseudo})
                    return "{" + pseudo + "}"
                mappings.append({"original": token, "replacement": self.placeholder})
                return self.placeholder
            # Not a valid JWT – keep original text.
            return token

        return self._COMPILED.sub(_replacer, text), mappings
