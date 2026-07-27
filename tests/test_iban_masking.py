"""Quick smoke test for IBAN masking fixes."""

import re

# ── Test BankAccountRule regex fix ──────────────────────────────────────
# The key fix: [\s -] matches actual whitespace, vs [\\s -] which matches
# literal backslash + 's' + space + hyphen.

print("=== Testing regex character class fix ===")

# Original (broken) pattern
broken_pattern = r"[\\s -]*"
# Fixed pattern
fixed_pattern = r"[\s -]*"

test_separators = [" ", "\t", "-", "\n"]
for sep in test_separators:
    broken_matches = re.fullmatch(broken_pattern + "x", sep + "x") is not None
    fixed_matches = re.fullmatch(fixed_pattern + "x", sep + "x") is not None
    status = (
        "OK"
        if fixed_matches and not broken_matches
        else ("SAME" if broken_matches == fixed_matches else "FAIL")
    )
    print(
        f"  separator '{sep!r}': "
        f"broken={broken_matches}, fixed={fixed_matches} [{status}]"
    )

# ── Test spaced/dashed IBAN matching with BankAccountRule ───────────────
print("\n=== Testing BankAccountRule spaced/dashed matching ===")

_CC = (
    "(?:(?:AL|AD|AE|AT|BA|BH|BY|BE|BG|BR|CH|CR|CY|CZ|DE|DK|"
    "DO|EE|EG|ES|FI|FO|FR|GB|GE|GI|GL|GR|GT|HR|HU|IE|IQ|IR|IS|"
    "JO|KW|KZ|LB|LC|LI|LT|LU|LV|MC|MD|ME|MK|MR|MT|MU|NL|NO|"
    "PK|PL|PS|PT|QA|RO|RS|SA|SC|SE|SI|SK|SM|ST|TL|TN|TR|UA|"
    "UK|VA|VG))"
)

# Fixed spaced_dashed pattern
spaced_dashed_fixed = (
    rf"\b{_CC}"
    + r"(?:[ \t\n\r\v\f-])"
    + r"\d\d"
    + r"([A-Za-z0-9]"
    + r"[\s -]*[A-Za-z0-9]?"
    + r"){1,9})"
)

# Test IBANs
test_cases = [
    ("GB 29NW BK60 1613 3192 6819", "GB", 22),
    ("NL 91AB NA04 1716 4300", "NL", 18),
    ("FR 7630 0060 0001 1234 5678 9018 9", "FR", 27),
    ("PL 17 102000 105799 000000 00000000", "PL", 28),
]

pattern = re.compile(spaced_dashed_fixed, re.IGNORECASE)

for iban, cc, expected_len in test_cases:
    # The regex needs to match the full IBAN including spaces/hyphens.
    # But \b at start means word boundary, and the pattern consumes all chars after CC.
    # For spaced/dashed: we need the BBAN to consume separators correctly.

    # Count non-separator chars after CC+check digits for this country
    expected_bban_chars = expected_len - 4  # minus CC(2) + check_digits(2)

    # Find where BBAN starts (after "CC DD")
    # The spaced_dashed pattern should match starting at position of CC

    m = pattern.search(iban)
    if m:
        matched_text = m.group(0)
        cleaned = re.sub(r"[\s\-]+", "", matched_text).upper()
        actual_len = len(cleaned)
        print(
            f"  {iban[:35]:35s} -> "
            f"match='{matched_text}', len={actual_len}, "
            f"expected={expected_len} "
            f"{'OK' if actual_len == expected_len else 'MISMATCH'}"
        )
    else:
        # Try compact alternative
        compact = rf"\b{_CC}\d\d[A-Za-z0-9]+"
        cm = re.search(compact, iban, re.IGNORECASE)
        print(f"  {iban[:35]:35s} -> NO MATCH (spaced_dashed)")

# ── Test with actual FastMasker ─────────────────────────────────────────
print("\n=== Testing with FastMasker ===")

import sys  # noqa: E402

sys.path.insert(0, "/mnt/data2/dev/develop/llm-router-plugins")  # noqa: E402

from llm_router_plugins.maskers.fast_masker.core.masker import (
    FastMasker,
)  # noqa: E402

# Clear the rule cache to pick up changes
FastMasker._rules_cache = None

masker = FastMasker()

test_ibans = [
    # Spaced
    "PL 17 102000 105799 000000 00000000",
    "DE 4450 0105 1704 4567 9095",
    "GB 29NW BK60 1613 3192 6819",
    "FR 7630 0060 0001 1234 5678 9018 9",
    "ES 9121 0004 1845 0200 0513 32",
    "NL 91AB NA04 1716 4300",
    # Compact
    "PL171020001057990000000000000000",
    "DE44500105170445679095",
    "GB29NWBK60161331926819",
    "FR7630006000011234567890189",
    # Dashed
    "PL-1710-2000-1057-9900-0000-0000-0000",
]

print(f"\n{'Input':45s} | {'Masked':45s} | Status")
print("-" * 110)

for iban in test_ibans:
    masked, mappings = masker.mask(iban)

    # Check what rule caught it
    if len(mappings) > 0:
        key = list(mappings.keys())[0]
        rule_name = ""
        for r in masker.rules:
            if hasattr(r, "placeholder") and "{{" in r.placeholder:
                tag = r.placeholder.strip("{}")
                if tag in key:
                    rule_name = tag
                    break
    else:
        rule_name = "NO MATCH"

    is_masked = masked != iban
    status = "MASKED" if is_masked else "UNMASKED"
    print(f"{iban:45s} | {masked:45s} | {status} (by {rule_name})")

# ── Test with text containing IBANs ─────────────────────────────────────
print("\n=== Testing with mixed text ===")

test_text = """
Moje konta bankowe:
PL 17 102000 105799 000000 00000000 (Polska)
DE44500105170445679095 (Niemcy)
GB 29NW BK60 1613 3192 6819 (Wielka Brytania)

Natomiast to jest adres: ul. Nowa 15, Warszawa
"""

FastMasker._rules_cache = None
masker2 = FastMasker()
masked_text, _ = masker2.mask(test_text)
print("Original:")
print(test_text)
print("\nMasked:")
print(masked_text)

# ── Verify no false positives remain ─────────────────────────────────────
print("\n=== Checking for false positives ===")

# A masked IBAN should use BANK_ACCOUNT placeholder, not others
for iban in test_ibans:
    FastMasker._rules_cache = None
    m2 = FastMasker()
    masked, _ = m2.mask(iban)
    if masked == iban:
        print(f"  WARNING: {iban} was NOT masked!")
    else:
        for tag in ["NRB", "TRANSACTION_REF", "CAR_PLATE", "SSL_CERT", "KRS"]:
            if f"{tag}_" in masked:
                print(f"  FALSE POSITIVE ({tag}): {iban} -> {masked}")

print("\n=== Done ===")
