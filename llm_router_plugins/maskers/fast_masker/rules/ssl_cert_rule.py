"""
Rule that masks SSL/TLS certificate serial numbers.

The rule:

1. Detects hex serial numbers in the typical SSL certificate range
   (16–40 hex characters).
2. Validates the value with :func:`is_valid_ssl_serial`.
3. Replaces **valid** serials with the placeholder ``{{SSL_CERT}}``.
4. If an ``anonymizer_fn`` is supplied, its result (wrapped in ``{}``) is used
   instead of the static placeholder, consistent with the other masking rules.
"""

import re
from typing import Optional, Callable

from .base_rule import BaseRule
from ..utils.validators import is_valid_ssl_serial


class SslCertRule(BaseRule):
    """
    Detects SSL/TLS certificate serial numbers and masks them.
    """

    # Match 16-40 hex characters (typical SSL cert serial range).
    _REGEX = r"""
        \b
        [0-9A-Fa-f]{16,40}\b
    """

    _PLACEHOLDER = "{{SSL_CERT}}"

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
    ) -> str:
        """
        Replace each *valid* SSL certificate serial with the placeholder.

        Parameters
        ----------
        text:
            Input string that may contain certificate serial numbers.
        anonymizer_fn:
            Optional callable ``fn(serial: str, tag_type: str) -> str``.
            If supplied, its return value is used (wrapped in ``{}``) instead of
            ``{{SSL_CERT}}``.
        """

        def _replacer(match: re.Match) -> str:
            serial = match.group(0)
            if is_valid_ssl_serial(serial):
                replacement = (
                    "{" + anonymizer_fn(serial, self.tag_type) + "}"
                    if anonymizer_fn
                    else self.placeholder
                )
                return replacement
            # Invalid serial – keep original text.
            return serial

        return self._COMPILED.sub(_replacer, text)
