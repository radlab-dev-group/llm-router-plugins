"""
Optional helper base class for rules that share common behaviour.
"""

import abc
import re
from typing import Pattern, Optional, Callable

from ..core.rule_interface import MaskerRuleI


class BaseRule(MaskerRuleI, abc.ABC):
    """
    Simple base class that stores a compiled regular expression and a
    replacement placeholder. Sub‑classes only need to provide the pattern
    and placeholder.
    """

    pattern: Pattern
    placeholder: str
    tag_type: str

    def __init__(self, regex: str, placeholder: str, flags: int = 0):
        """
        Parameters
        ----------
        regex: str
            Regular expression pattern to search for.
        placeholder: str
            Text that will replace each match.
        flags: int, optional
            Flags passed to :func:`re.compile`. Default is ``0``.
        """
        self.pattern = re.compile(regex, flags)
        self.placeholder = placeholder

        # Extract tag type from placeholder, e.g., "{{PESEL}}" -> "PESEL"
        self.tag_type = placeholder.strip("{}")
        # self.tag_type = placeholder.strip("{}")

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        """
        Replace all occurrences of ``self.pattern`` in *text* with the
        configured placeholder or a dynamic pseudonym.

        Returns
        -------
        str
            The transformed text.
        """

        def replacer(match: re.Match) -> str:
            val = match.group(0)
            if anonymizer_fn:
                return "{" + anonymizer_fn(val, self.tag_type) + "}"
            return self.placeholder

        return self.pattern.sub(replacer, text)
