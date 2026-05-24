"""
FastMasker module
=================

Provides the :class:`FastMasker` class – a thin orchestration layer that
applies a configurable sequence of
:class:`~llm_router_lib.fast_masker.core.rule_interface.MaskerRuleI`
implementations to arbitrary payloads.

The module imports the concrete rule classes (``PhoneRule``, ``UrlRule``, …) and
exposes a default rule set (``ALL_MASKER_RULES``) that can be overridden
by supplying a custom list to the constructor.
"""

import re
import datetime
import pandas as pd

from typing import List, Tuple, Dict, Optional

from llm_router_plugins.maskers.payload_interface import MaskerPayloadTraveler

from .rule_interface import MaskerRuleI
from ..rules import (
    EmailRule,
    UrlRule,
    IpRule,
    PeselTaggedRule,
    PeselRule,
    NipRule,
    KrsRule,
    RegonRule,
    DateWordRule,
    DateNumberRule,
    MoneyRule,
    VinRule,
    PostalCodeRule,
    NrbRule,
    BankAccountRule,
    PhoneInternationalRule,
    PhoneRule,
    MacAddressRule,
    CreditCardRule,
    PassportRule,
    IdCardRule,
    SsnRule,
    HealthIdRule,
    SslCertRule,
    JwtRule,
    InvoiceNumberRule,
    OrderNumberRule,
    TransactionRefRule,
    SimCardRule,
    SocialIdRule,
    EuVatRule,
)


class FastMasker(MaskerPayloadTraveler):
    """
    Orchestrates the application of a list of simple masking rules.
    """

    _rules_cache: List[MaskerRuleI] | None = None

    @classmethod
    def _get_rules(cls) -> List[MaskerRuleI]:
        if cls._rules_cache is None:

            cls._rules_cache = [
                EmailRule(),
                UrlRule(),
                IpRule(),
                PeselTaggedRule(),
                PeselRule(),
                NipRule(),
                KrsRule(),
                RegonRule(),
                DateWordRule(),
                DateNumberRule(),
                MoneyRule(),
                VinRule(),
                PostalCodeRule(),
                NrbRule(),
                BankAccountRule(),
                PhoneInternationalRule(),
                PhoneRule(),
                MacAddressRule(),
                CreditCardRule(),
                PassportRule(),
                IdCardRule(),
                SsnRule(),
                HealthIdRule(),
                SslCertRule(),
                JwtRule(),
                InvoiceNumberRule(),
                OrderNumberRule(),
                TransactionRefRule(),
                SimCardRule(),
                SocialIdRule(),
                EuVatRule(),
            ]
        return cls._rules_cache

    def __init__(self, rules: Optional[List[MaskerRuleI]] = None):
        self.rules = rules or self._get_rules()
        self.mapping = {}  # original -> pseudo
        self.reverse_mapping = {}  # pseudo -> original

    def _get_pseudo(self, value: str, tag_type: str = None) -> str:
        # We should NOT strip punctuation from the value we are replacing in text
        # unless the rule itself excludes it from the match.
        # But if we want to store it in mapping, we should store exactly what we want to replace.
        val_norm = value
        if val_norm in self.mapping:
            return self.mapping[val_norm]

        # Use the tag_type if provided, otherwise default to "ENTITY"
        base_tag = tag_type or "ENTITY"
        dt_str = int(datetime.datetime.now().timestamp() * 1_000_000)
        pseudo = f"{base_tag}_{dt_str}_{len(self.mapping) + 1}"
        # If it's a simple value (not inside text), we might want a simpler tag
        # but for consistency with the traveler, we use the same.
        self.mapping[val_norm] = pseudo
        self.reverse_mapping[pseudo] = val_norm
        return pseudo

    def mask(self, text: str) -> Tuple[str, Dict]:
        """
        Apply all configured rules to a plain‑text string with de-anonymization support.
        """
        # Ensure we are working with a string
        if not isinstance(text, str):
            return text, {}

        # If the text is already a pseudonym from a previous rule, don't mask it further
        if re.fullmatch(r"[A-Z]+_\d+_\d+", text):
            return text, {}

        result = text
        all_mappings = []
        # List of rules to apply
        for rule in self.rules:
            # We want to avoid masking already masked parts in a larger text.
            # This is tricky with simple regex replacement.
            # For now, let's at least avoid the common case where a rule
            # matches parts of our own pseudonyms (like NIP matching digits in the timestamp).
            [result, mappings] = rule.apply(result, anonymizer_fn=self._get_pseudo)
            all_mappings.extend(mappings)

        mappings = {}
        for m in all_mappings:
            mappings[m["replacement"]] = m["original"]
        return result, mappings

    def _mask_text(self, text: str) -> Tuple[str, Dict]:
        """
        Implements the traveler interface by calling the mask method.
        """
        return self.mask(text)

    def get_mapping_df(self) -> pd.DataFrame:
        data = []
        for orig, pseudo in self.mapping.items():
            data.append(
                {"Oryginalna wartość": orig, "Wygenerowany pseudonim": pseudo}
            )
        return pd.DataFrame(data)

    def save_mapping(self, path: str):
        df = self.get_mapping_df()
        df.to_excel(path, index=False)


class FastDeanonymizer(MaskerPayloadTraveler):
    def __init__(self):
        self.reverse_map = {}
        self.pattern = None

    def load_mapping(self, path: str):
        try:
            df = pd.read_excel(path)
            self.reverse_map = {
                str(row["Wygenerowany pseudonim"]): str(row["Oryginalna wartość"])
                for _, row in df.iterrows()
            }
            if self.reverse_map:
                keys = sorted(self.reverse_map.keys(), key=len, reverse=True)
                self.pattern = re.compile("|".join([re.escape(k) for k in keys]))
            return True
        except Exception:
            return False

    def deanonymize(self, text: str) -> str:
        """
        Implements the de-anonymization logic.
        """
        if not isinstance(text, str) or not self.pattern or not self.reverse_map:
            return text
        return self.pattern.sub(lambda m: self.reverse_map[m.group(0)], text)

    def _mask_text(self, text: str) -> str:
        """
        Implements the traveler interface by calling deanonymize.
        """
        return self.deanonymize(text)
