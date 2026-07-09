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

from llm_router_plugins.maskers.fast_masker.core.rule_interface import MaskerRuleI
from llm_router_plugins.maskers.fast_masker.rules import (
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
    Orchestrates the application of a configurable list of :class:`MaskerRuleI`
    implementations to arbitrary payloads.

    :class:`FastMasker` is the core engine of the masking pipeline.  It applies
    each rule in sequence to every string value found in a payload (recursively
    traversing dicts and lists).  Matches are replaced with either a static
    placeholder or a dynamically generated pseudonym (via *anonymizer_fn*).

    Mappings of original → pseudonym values are tracked in :attr:`mapping`
    and can be used later for de‑anonymization (see :class:`FastDeanonymizer`).

    Attributes
    ----------
    rules : List[MaskerRuleI]
        The masking rules to apply.  If *rules* is ``None`` in the
        constructor, the default rule set is loaded via :meth:`_get_rules`.
    mapping : Dict[str, str]
        Mapping of original → pseudonym values generated during masking.
    reverse_mapping : Dict[str, str]
        Mapping of pseudonym → original values (mirror of *mapping*).

    Parameters
    ----------
    rules : List[MaskerRuleI], optional
        Custom list of masking rules.  If ``None``, the default rule set
        is used (email, URL, IP, PESEL, NIP, phone, credit card, etc.).

    Returns
    -------
    None
    """

    _rules_cache: List[MaskerRuleI] | None = None

    @classmethod
    def _get_rules(cls) -> List[MaskerRuleI]:
        """
        Load the default set of masking rules (cached across calls).

        The default rules cover: email, URL, IP, PESEL, NIP, KRS, REGON,
        dates (word + numeric), money, VIN, postal code, NRB (bank account),
        phone (PL + international), MAC, credit card, passport, ID card,
        SSN, health ID, SSL cert, JWT, invoice/order/transaction numbers,
        SIM card ICCID, social ID, and EU VAT.

        Returns
        -------
        List[MaskerRuleI]
            A list of the default masking rules.

        Raises
        ------
        None
        """
        if cls._rules_cache is None:
            cls._rules_cache = [
                # Structural identifiers first (validate before generic patterns)
                EmailRule(),
                UrlRule(),
                IpRule(),
                PeselTaggedRule(),
                PeselRule(),
                NipRule(),
                RegonRule(),
                DateWordRule(),
                DateNumberRule(),
                MoneyRule(),
                VinRule(),
                PostalCodeRule(),
                NrbRule(),
                BankAccountRule(),
                KrsRule(),  # structural — before Phone to avoid false positives
                PassportRule(),
                IdCardRule(),
                SsnRule(),
                HealthIdRule(),
                # Generic patterns (lower specificity, run last)
                PhoneInternationalRule(),
                PhoneRule(),
                MacAddressRule(),
                CreditCardRule(),
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
        """
        Initialise the FastMasker with a set of masking rules.

        Parameters
        ----------
        rules : List[MaskerRuleI], optional
            Custom list of masking rules.  If ``None``, the default rule set
            is loaded via :meth:`_get_rules`.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.rules = rules or self._get_rules()
        self.mapping = {}  # original -> pseudo
        self.reverse_mapping = {}  # pseudo -> original

    def _get_pseudo(self, value: str, tag_type: str = None) -> str:
        """
        Generate or return a pseudonym for *value*.

        Pseudonyms are cached in :attr:`mapping` so the same original value
        always maps to the same pseudonym within a single FastMasker instance.

        Parameters
        ----------
        value : str
            The original text to pseudonymise.
        tag_type : str, optional
            Prefix for the pseudonym (e.g. ``"PESEL"``).  Defaults to
            ``"ENTITY"`` when ``None``.

        Returns
        -------
        str
            A unique pseudonym string (e.g. ``"PESEL_1625234567890_1"``).

        Raises
        ------
        None
        """
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
        Apply all configured rules to a plain‑text string with de‑anonymization support.

        Each rule in :attr:`rules` is applied in sequence.  Already‑masked
        fragments (matching ``{TAG_N_M}`` pattern) are skipped to avoid
        double‑masking.

        Parameters
        ----------
        text : str
            The input text to mask.

        Returns
        -------
        Tuple[str, Dict]
            ``(masked_text, mappings)`` where *masked_text* has all sensitive
            data replaced with pseudonyms and *mappings* maps original →
            pseudonym values.

        Raises
        ------
        None
        """
        # Ensure we are working with a string
        if not isinstance(text, str):
            return text, {}

        # If the text is already a pseudonym
        # f.e. from a previous rule, don't mask it further
        if re.fullmatch(r"\{[A-Z_]+_\d+(?:_\d+)?\}", text):
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
        Implements the :class:`MaskerPayloadTraveler` interface by calling
        :meth:`mask`.

        Parameters
        ----------
        text : str
            The input text to mask.

        Returns
        -------
        Tuple[str, Dict]
            The masked text and its mappings (from :meth:`mask`).

        Raises
        ------
        None
        """
        return self.mask(text)

    def get_mapping_df(self) -> pd.DataFrame:
        """
        Build a :class:`pandas.DataFrame` of original → pseudonym mappings.

        Returns
        -------
        pd.DataFrame
            A two‑column DataFrame with columns ``"Oryginalna wartość"``
            and ``"Wygenerowany pseudonim"``.

        Raises
        ------
        None
        """
        data = []
        for orig, pseudo in self.mapping.items():
            data.append(
                {"Oryginalna wartość": orig, "Wygenerowany pseudonim": pseudo}
            )
        return pd.DataFrame(data)

    def save_mapping(self, path: str):
        """
        Save the current mapping to an Excel file at *path*.

        Parameters
        ----------
        path : str
            File path where the Excel file will be written.

        Returns
        -------
        None

        Raises
        ------
        pandas.errors.EmptyDataError
            If there are no mappings to save.
        """
        df = self.get_mapping_df()
        df.to_excel(path, index=False)


class FastDeanonymizer(MaskerPayloadTraveler):
    """
    Restores original text from pseudonymized payloads using a saved mapping.

    Loads pseudonym → original mappings from an Excel file (produced by
    :meth:`FastMasker.save_mapping`) and replaces every pseudonym in the
    input text with its original value.

    Attributes
    ----------
    reverse_map : Dict[str, str]
        Mapping of pseudonym → original value loaded from the Excel file.
    pattern : re.Pattern, optional
        Compiled regex that matches any pseudonym in *reverse_map*.
    """

    def __init__(self):
        """
        Initialise an empty FastDeanonymizer with no loaded mapping.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.reverse_map = {}
        self.pattern = None

    def load_mapping(self, path: str):
        """
        Load a pseudonym → original mapping from an Excel file.

        Parameters
        ----------
        path : str
            Path to the Excel file produced by :meth:`FastMasker.save_mapping`.

        Returns
        -------
        bool
            ``True`` if the mapping was loaded successfully, ``False`` on
            any error (e.g. file not found, malformed Excel).

        Raises
        ------
        None
        """
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
        Replace every pseudonym in *text* with its original value.

        Parameters
        ----------
        text : str
            The pseudonymized text to restore.

        Returns
        -------
        str
            The de‑anonymized text.  If no mapping is loaded, *text* is
            returned unchanged.

        Raises
        ------
        None
        """
        if not isinstance(text, str) or not self.pattern or not self.reverse_map:
            return text
        return self.pattern.sub(lambda m: self.reverse_map[m.group(0)], text)

    def _mask_text(self, text: str) -> str:
        """
        Implements the :class:`MaskerPayloadTraveler` interface by calling
        :meth:`deanonymize`.

        Parameters
        ----------
        text : str
            The pseudonymized text to restore.

        Returns
        -------
        str
            The de‑anonymized text.

        Raises
        ------
        None
        """
        return self.deanonymize(text)
