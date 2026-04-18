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

from typing import List, Tuple, Dict

from ..rules import *
from .rule_interface import MaskerRuleI
from llm_router_plugins.maskers.payload_interface import MaskerPayloadTraveler


class FastMasker(MaskerPayloadTraveler):
    """
    Orchestrates the application of a list of simple masking rules.
    """

    # Rules ordered from most specific/certain to least specific
    # Priority: checksum-validated > structured formats > pattern-based
    __ALL_MASKER_RULES = [
        EmailRule(),  # Email addresses (before URLs!)
        UrlRule(),  # URLs and domains
        IpRule(),  # IP addresses with optional ports
        PeselTaggedRule(),  # PESEL: + PESEL with checksum
        PeselRule(),  # PESEL with checksum
        NipRule(),  # NIP with checksum
        KrsRule(),  # KRS with checksum
        RegonRule(),  # REGON with checksum
        DateWordRule(),  # Textual dates
        DateNumberRule(),  # Numeric dates
        MoneyRule(),  # Amounts with currency
        VinRule(),  # VIN checksum (ISO 3779)
        PhoneInternationalRule(),  # + prefix with country code
        PhoneRule(),  # Local phone (9 digits),
        # CarPlateRule(),  # License plates
        PostalCodeRule(),  # DD-DDD format
        MacAddressRule(),  # MAC address format
        CreditCardRule(),  # Luhn checksum
        NrbRule(),  # 26 digits (bank account)
        BankAccountRule(),  # Polish IBAN
        PassportRule(),  # 2 letters + 7 digits
        IdCardRule(),  # 3 letters + 6 digits
        SsnRule(),  # SSN format AAA-GG-SSSS
        HealthIdRule(),  # NFZ with slash
        SslCertRule(),  # 16-40 hex chars
        JwtRule(),  # JWT tokens (3 parts)
        InvoiceNumberRule(),  # FV/INV patterns
        OrderNumberRule(),  # ORD patterns
        TransactionRefRule(),  # Transaction IDs
        SimCardRule(),  # 19-20 digit ICCID
        SocialIdRule(),  # fbid only
        EuVatRule(),  # fbid only
    ]

    ALL_MASKER_RULES = [cls for cls in __ALL_MASKER_RULES if cls]

    def __init__(self, rules: Optional[List[MaskerRuleI]] = None):
        self.rules = rules or self.ALL_MASKER_RULES
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
