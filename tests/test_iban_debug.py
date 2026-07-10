"""Quick IBAN matching debug."""
import re

# Replicate BankAccountRule patterns exactly as they are in the file
_IbanCountryAlt = "|".join(sorted({
    "AL", "AD", "AE", "AT", "BA", "BH", "BY", "BE", "BG", "BR", "CH",
    "CR", "CY", "CZ", "DE", "DK", "DO", "EE", "EG", "ES", "FI", "FO",
    "FR", "GB", "GE", "GI", "GL", "GR", "GT", "HR", "HU", "IE", "IQ",
    "IR", "IS", "IT", "JO", "KW", "KZ", "LB", "LC", "LI", "LT", "LU",
    "LV", "MC", "MD", "ME", "MK", "MR", "MT", "MU", "NL", "NO", "PK",
    "PL", "PS", "PT", "QA", "RO", "RS", "SA", "SC", "SE", "SI", "SK",
    "SM", "ST", "TL", "TN", "TR", "UA", "UK", "VA", "VG",
}))
_CC = rf"(?:{_IbanCountryAlt})"

compact = rf"\b{_CC}\d\d[A-Za-z0-9]+"
spaced_dashed = (
    rf"\b{_CC}"
    + r"(?:[ \t\n\r\v\f-])"
    + r"\d\d"
    + r"([A-Za-z0-9]"
    + r"[\s -]*[A-Za-z0-9]?"
    + r"){1,9})"
)

print("Compact pattern:", compact[:80], "...")
print("Spaced/dashed length:", len(spaced_dashed))

# Try compiling spaced_dashed
try:
    sd_pat = re.compile(spaced_dashed, re.IGNORECASE)
    print("Spaced/dashed compiles OK")
except Exception as e:
    print(f"Spaced/dashed compile error: {e}")

compact_pat = re.compile(compact, re.IGNORECASE)
print("\n=== Testing regex matches ===\n")

test_ibans = [
    "PL 17 102000 105799 000000 00000000",
    "PL171020001057990000000000000000",
    "DE 4450 0105 1704 4567 9095",
    "DE44500105170445679095",
    "GB 29NW BK60 1613 3192 6819",
    "GB29NWBK60161331926819",
    "NL 91AB NA04 1716 4300",
    "NL91ABNA0417164300",
]

for iban in test_ibans:
    print(f"IBAN: {iban} (len={len(iban)})")

    # Check compact pattern matches
    for m in compact_pat.finditer(iban):
        matched = m.group(0)
        cleaned = re.sub(r"[\s\-]+", "", matched).upper()
        print(f"  COMPACT match: pos={m.start()} '{matched}' len={len(cleaned)}")

    # Check spaced/dashed matches
    if 'sd_pat' in dir():
        for m in sd_pat.finditer(iban):
            matched = m.group(0)
            cleaned = re.sub(r"[\s\-]+", "", matched).upper()
            print(f"  SPACED/DASHED match: pos={m.start()} '{matched}' len={len(cleaned)}")

    print()
