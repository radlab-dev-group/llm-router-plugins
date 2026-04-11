"""
Definition of the rule interface that every masking rule must implement.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class MaskerRuleI(ABC):
    """
    Abstract base class for all masking rules.

    Sub‑classes must implement the :meth:`apply` method, which receives a
    string and returns a new string where the rule‑specific patterns have been
    replaced with placeholders.
    """

    @abstractmethod
    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> str:
        """
        Apply the rule to *text* and return the transformed string.

        Parameters
        ----------
        text: str
            The input text to be processed.
        anonymizer_fn: Callable[[str, str], str] | None
            A function that takes (original_value, tag_type) and returns a pseudonym.
            If None, the rule should use its default static placeholder.

        Returns
        -------
        str
            The text after the rule has been applied.
        """
        raise NotImplementedError
