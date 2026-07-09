"""
Rule that mask Polish bank account numbers (IBAN format).

Polish IBAN (28 characters, without spaces):
    PL58105012981000009062923173

Polish IBAN with spaces:
    PL58 1050 1298 1000 0090 6292 3173

The rule also recognises **partially masked** accounts where any group of
four characters may be replaced with the literal character ``X`` (or a
mixture of ``X`` and digits), e.g.:

    PL58 10XX 1298 1XXX XXXX 6292 31X3
    XX 10XX 1298 1XXX XXXX 6292 31X3

Only strings that have the exact length of a Polish IBAN (28 alphanumeric
characters, ignoring whitespace) are matched – short numbers such as
``64001000152`` are **not** treated as bank accounts.  The placeholder used
for masking is ``{{BANK_ACCOUNT}}``.
"""

import re
from typing import Optional, Callable, Match, Tuple, List

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


def _iban_mod97(iban: str) -> int:
    """Calculate IBAN modulo 97 checksum per ISO 13616.

    Rearranges the IBAN (first 4 characters to end), converts letters to
    digits (A=10, B=11, …, Z=35), and returns ``result % 97``.
    A valid IBAN always yields 1.
    """
    rearranged = iban[4:] + iban[:4]
    converted = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(converted) % 97


class BankAccountRule(BaseRule):
    """
    Detects Polish IBAN numbers (full 28‑character length), allowing optional
    masking with ``X`` characters and optional whitespace between groups.

    Country codes are validated against the list of known IBAN-adopting countries
    from :attr:`_IBAN_COUNTRIES`. Unknown two-letter prefixes are rejected to
    reduce false positives (e.g. "XX1234..." no longer matches).

    Full IBANs (no ``X``) are further validated with the ISO 13616 modulo-97
    checksum.  Partially masked accounts skip the checksum but must still
    match the structural pattern.
    """

    # ISO 3166-1 alpha‑2 country codes that have adopted IBAN.  Sources:
    #   https://www.iban.com/structure
    _IBAN_COUNTRIES = frozenset(
        {
            "AL",
            "AD",
            "AE",
            "AT",
            "BA",
            "BH",
            "BY",
            "BE",
            "BG",
            "BR",
            "CH",
            "CR",
            "CY",
            "CZ",
            "DE",
            "DK",
            "DO",
            "EE",
            "EG",
            "ES",
            "FI",
            "FO",
            "FR",
            "GB",
            "GE",
            "GI",
            "GL",
            "GR",
            "GT",
            "HR",
            "HU",
            "IE",
            "IQ",
            "IR",
            "IS",
            "IT",
            "JO",
            "KW",
            "KZ",
            "LB",
            "LC",
            "LI",
            "LT",
            "LU",
            "LV",
            "MC",
            "MD",
            "ME",
            "MK",
            "MR",
            "MT",
            "MU",
            "NL",
            "NO",
            "PK",
            "PL",
            "PS",
            "PT",
            "QA",
            "RO",
            "RS",
            "SA",
            "SC",
            "SE",
            "SI",
            "SK",
            "SM",
            "ST",
            "TL",
            "TN",
            "TR",
            "UA",
            "UK",
            "VA",
            "VG",
        }
    )

    # Pre‑compiled alternation of all valid IBAN country codes (max 2 chars each).
    _IBAN_CC = r"(?:AL|AD|AE|AT|BA|BH|BY|BE|BG|BR|CH|CR|CY|CZ|DE|DK|DO|EE|EG|ES|FI|FO|FR|GB|GE|GI|GL|GR|GT|HR|HU|IE|IQ|IR|IS|IT|JO|KW|KZ|LB|LC|LI|LT|LU|LV|MC|MD|ME|MK|MR|MT|MU|NL|NO|PK|PL|PS|PT|QA|RO|RS|SA|SC|SE|SI|SK|SM|ST|TL|TN|TR|UA|UK|VA|VG)"

    # Two check digits – must be two numeric digits.
    _CHECK = r"\d{2}"

    # One group of four characters – digits, X or any mixture (e.g. X1X2)
    _GROUP = r"(?:[0-9X]{4})"

    # Full pattern:
    #   - mandatory valid IBAN country code (from the list above)
    #   - optional whitespace between country code and check digits (handles "PL 12...")
    #   - mandatory check digits (two numeric digits)
    #   - exactly six groups of four characters, each optionally preceded by
    #     whitespace (including the possibility of no whitespace at all)
    #   - word boundaries on both sides to avoid partial matches
    _FULL_PATTERN = rf"""
                \b                      # start of word
                {_IBAN_CC}              # mandatory valid IBAN country code
                \s*                     # optional whitespace between CC and check digits
                {_CHECK}                # mandatory check digits (2 digits)
                (?:\s*{_GROUP}){{6}}   # six groups of 4 chars, whitespace optional
                \b                      # end of word
            """

    _REGEX = _FULL_PATTERN

    def __init__(self):
        super().__init__(
            regex=self._REGEX,
            placeholder="{{BANK_ACCOUNT}}",
            flags=re.IGNORECASE | re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, List]:
        """
        Replace each detected (possibly masked) bank account number with the
        ``{{BANK_ACCOUNT}}`` placeholder.

        Full IBANs (no ``X`` characters) are validated with the ISO 13616
        modulo‑97 checksum.  Masked accounts skip the checksum.
        """
        mappings = []

        def _replacer(match: Match) -> str:
            val = match.group(0)
            # Clean up whitespace for validation
            cleaned = re.sub(r"\s+", "", val)

            has_x = "X" in cleaned or "x" in cleaned

            if not has_x and _iban_mod97(cleaned.upper()) != 1:
                return val  # invalid IBAN checksum

            if anonymizer_fn:
                pseudo = anonymizer_fn(val, self.tag_type)
                mappings.append({"original": val, "replacement": pseudo})
                return "{" + pseudo + "}"
            mappings.append({"original": val, "replacement": self.placeholder})
            return self.placeholder

        return self.pattern.sub(_replacer, text), mappings
