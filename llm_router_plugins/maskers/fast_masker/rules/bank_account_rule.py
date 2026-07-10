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

Full IBANs (no ``X``) are validated with the ISO 13616 modulo-97 checksum.
Partially masked accounts skip the checksum but must still match the
structural pattern and pass country-specific length validation.

The rule supports three separator styles: compact (no separators), spaced
(space between groups), and dashed (hyphens between groups).  Different IBAN
countries have different total lengths; all are handled via a per-country
length table in ``_IBAN_LENGTHS``.
"""

import re
from typing import Optional, Callable, Match, Tuple, List

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule
from llm_router_plugins.maskers.fast_masker.utils.validators import is_valid_iban


# ── Country-code alternation (sorted to ensure longest match first) ───────

_IbanCountryAlt = "|".join(
    sorted(
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
)

# Anchor pattern: finds any valid country code followed by an optional separator
# and exactly 2 digits. This is used in apply() to locate potential IBAN starts;
# the BBAN is then greedily consumed by the regex below.
_IbanAnchorPattern = re.compile(
    rf"\b(?:{_IbanCountryAlt})[ \t\n\r\v\f-]?\d\d",
    re.IGNORECASE,
)

# Combined anchor: real country codes only.  Two branches handle all formats:
# 1. Compact/separated: CC + optional sep + exactly 2 digits. E.g. "PL58" or "GB 29".
#    The optional separator allows spaces/hyphens between country code and check digits.
# 2. Partial (X-masked): CC + optional sep + 2 alnum where 2nd char is followed by
#    a non-alpha boundary or end-of-string.  Catches "PLXX", "PL-58", "PL 73" but
#    rejects "tran" (no alpha-boundary after 'an' → mid-word).
_IbanAnchorCombined = re.compile(
    rf"\b(?:{_IbanCountryAlt})"
    rf"(?:[ \t\n\r\v\f-]?\d\d|[ \t\n\r\v\f-]?[A-Za-z0-9]"
    rf"[ \t\n\r\v\f-]?[A-Za-z0-9](?=[^a-zA-Z]|$))",
    re.IGNORECASE,
)


class BankAccountRule(BaseRule):
    """
    Detects IBAN bank accounts (any supported country), allowing optional
    masking with ``X`` characters and optional space/hyphen separators.

    Country codes are validated against the ISO 3166-1 alpha‑2 list in
    :attr:`_IBAN_COUNTRIES`. Unknown two-letter prefixes are rejected to
    reduce false positives (e.g. "XX1234..." no longer matches).

    Full IBANs (no ``X``) are further validated with the ISO 13616 modulo-97
    checksum **and** country-specific length validation.  Partially masked
    accounts skip the checksum but must still match the structural pattern
    and pass length validation.
    """

    # ── Country-to-length mapping (ISO standard total IBAN lengths) ───────

    _IBAN_LENGTHS: dict[str, int] = {
        "AL": 28,
        "AD": 24,
        "AE": 23,
        "AT": 20,
        "BA": 20,
        "BH": 18,
        "BY": 29,
        "BE": 16,
        "BG": 22,
        "BR": 29,
        "CH": 21,
        "CR": 21,
        "CY": 28,
        "CZ": 24,
        "DE": 22,
        "DK": 18,
        "DO": 28,
        "EE": 20,
        "EG": 29,
        "ES": 24,
        "FI": 18,
        "FO": 18,
        "FR": 27,
        "GB": 22,
        "GE": 22,
        "GI": 23,
        "GL": 18,
        "GR": 27,
        "GT": 28,
        "HR": 21,
        "HU": 28,
        "IE": 22,
        "IQ": 23,
        "IR": 26,
        "IS": 26,
        "IT": 27,
        "JO": 30,
        "KW": 30,
        "KZ": 20,
        "LB": 28,
        "LC": 32,
        "LI": 21,
        "LT": 20,
        "LU": 20,
        "LV": 21,
        "MC": 27,
        "MD": 24,
        "ME": 22,
        "MK": 19,
        "MR": 27,
        "MT": 31,
        "MU": 30,
        "NL": 18,
        "NO": 15,
        "PK": 24,
        "PL": 28,
        "PS": 29,
        "PT": 25,
        "QA": 29,
        "RO": 24,
        "RS": 22,
        "SA": 30,
        "SC": 31,
        "SE": 24,
        "SI": 19,
        "SK": 24,
        "SM": 27,
        "ST": 25,
        "TL": 23,
        "TN": 24,
        "TR": 26,
        "UA": 29,
        "UK": 22,
        "VA": 22,
        "VG": 24,
    }

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

    def __init__(self) -> None:
        super().__init__(
            regex=r"\b[A-Z]{2}[ \t\n\r\v\f-]?[A-Za-z0-9]{2}",
            placeholder="{{BANK_ACCOUNT}}",
            flags=re.IGNORECASE,
        )

    def _validate_iban(self, iban: str) -> bool:
        """Validate an IBAN candidate. Returns True if valid."""
        cleaned = re.sub(r"[\s\-]+", "", iban).upper()
        cc = cleaned[:2]

        if cc not in self._IBAN_COUNTRIES:
            return False  # unknown country code

        expected_len = self._IBAN_LENGTHS.get(cc)
        if expected_len is not None and len(cleaned) != expected_len:
            return False  # wrong length for this country

        has_x = "X" in iban or "x" in iban
        if not has_x and is_valid_iban(cleaned) != 1:
            return False  # invalid checksum

        return True

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, List]:
        """
        Replace each detected (possibly masked) bank account number with the
        ``{{BANK_ACCOUNT}}`` placeholder.

        Full IBANs (no ``X`` characters) are validated with the ISO 13616
        modulo‑97 checksum **and** country-specific length validation.
        Partially masked accounts skip the checksum but must still pass
        length validation.
        """
        mappings: List[dict] = []
        result: List[str] = []
        last_end = 0

        for match in _IbanAnchorCombined.finditer(text):
            start = match.start()

            # Skip if already covered by a previous replacement.
            if start < last_end:
                continue

            anchor_start = start
            pos = match.end()  # right after CC + check digits (all formats)

            anchor_text = text[anchor_start:pos]
            # Strip spaces/hyphens to get pure CC (always 2 chars for valid country codes)
            cc = re.sub(r"[\s\-]+", "", anchor_text).upper()[:2]

            if cc not in self._IBAN_COUNTRIES:
                continue  # unknown country code — skip

            expected_len = self._IBAN_LENGTHS.get(cc)

            if expected_len is not None:
                # Count alnum/X chars consumed by the anchor.
                anchor_alnum = sum(1 for c in anchor_text if c.isalnum())
                need_after_anchor = expected_len - anchor_alnum

                end_pos = pos  # start scanning BBAN content right after the anchor
                alnum_count = 0
                for c in text[pos:]:
                    if alnum_count >= need_after_anchor:
                        break  # done consuming BBAN chars
                    if c == " " or c in "\t\r\v\f-":
                        end_pos += 1  # skip separators, don't count them
                    elif c.isalnum():
                        end_pos += 1  # consume one alnum/X char
                        alnum_count += 1
                    else:
                        break  # stop at non-separator, non-alnum chars

            else:
                # Unknown country — fall back to greedy BBAN consumption.
                bban_match = re.match(
                    r"[ \t\n\r\v\f-]*[A-Za-z0-9Xx]+(?:[ \t\-][A-Za-z0-9Xx]+)*",
                    text[pos:],
                )
                if bban_match:
                    end_pos = pos + bban_match.end()
                else:
                    end_pos = pos  # nothing after check digits

            candidate = text[anchor_start:end_pos]
            cleaned = re.sub(r"[\s\-]+", "", candidate).upper()

            if len(cleaned) != expected_len:
                continue  # wrong length — skip (IBAN checksum won't matter)

            has_x = "X" in candidate or "x" in candidate
            if not has_x and is_valid_iban(cleaned) != 1:
                continue  # invalid checksum — let other rules handle it

            # Valid IBAN — replace.
            result.append(text[last_end:anchor_start])
            if anonymizer_fn:
                pseudo = anonymizer_fn(candidate, self.tag_type)
                mappings.append({"original": candidate, "replacement": pseudo})
                result.append("{" + pseudo + "}")
            else:
                mappings.append(
                    {"original": candidate, "replacement": self.placeholder}
                )
                result.append(self.placeholder)
            last_end = end_pos

        result.append(text[last_end:])
        return "".join(result), mappings
