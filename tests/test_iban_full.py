"""Test actual FastMasker IBAN masking after fixes."""
import sys
sys.path.insert(0, "/mnt/data2/dev/develop/llm-router-plugins")

# Clear cache
from llm_router_plugins.maskers.fast_masker.core.masker import FastMasker
FastMasker._rules_cache = None

masker = FastMasker()

# IBAN test data (all should be masked as BANK_ACCOUNT)
test_cases = {
    "PL spaced":    "PL 17 102000 105799 000000 00000000",
    "PL compact":   "PL171020001057990000000000000000",
    "PL dashed":    "PL-1710-2000-1057-9900-0000-0000-0000",
    "PL variant1":  "PL 17 10 2000 1057 9900 0000 0000 0000",
    "PL variant2":  "PL17 10200010579900000000000000",
    "PL variant3":  "PL 1710 20001057 990000 000000 0000",
    "DE spaced":    "DE 4450 0105 1704 4567 9095",
    "DE compact":   "DE44500105170445679095",
    "DE dashed":    "DE-4450-0105-1704-4567-9095",
    "GB spaced":    "GB 29NW BK60 1613 3192 6819",
    "GB compact":   "GB29NWBK60161331926819",
    "FR spaced":    "FR 7630 0060 0001 1234 5678 9018 9",
    "FR compact":   "FR7630006000011234567890189",
    "FR dashed":    "FR-7630-0060-0001-1234-5678-9018-9",
    "ES spaced":    "ES 9121 0004 1845 0200 0513 32",
    "ES compact":   "ES9121000418450200051332",
    "ES dashed":    "ES-9121-0004-1845-0200-0513-32",
    "NL spaced":    "NL 91AB NA04 1716 4300",
    "NL compact":   "NL91ABNA0417164300",
    "NL dashed":    "NL-91AB-NA04-1716-4300",
}

print(f"{'Test':25s} | {'Masked':35s} | Status")
print("-" * 80)

issues = []
for name, iban in test_cases.items():
    FastMasker._rules_cache = None  # Clear cache for each test
    m = FastMasker()
    masked, mappings = m.mask(iban)

    is_masked = masked != iban

    # Determine which rule matched
    matched_by = "UNKNOWN"
    for tag in ["BANK_ACCOUNT", "NRB", "TRANSACTION_REF", "CAR_PLATE", "SSL_CERT", "KRS"]:
        if f"{tag}_" in masked:
            matched_by = tag
            break

    status = "OK" if is_masked else "FAIL"
    if not is_masked:
        issues.append((name, iban, masked))

    print(f"{name:25s} | {masked[:35]:35s} | {status} ({matched_by})")

print(f"\n=== Results: {len(test_cases) - len(issues)}/{len(test_cases)} OK ===")

if issues:
    print("\nFailed test cases:")
    for name, iban, masked in issues:
        print(f"  {name}: '{iban}' -> '{masked}'")

# Check rule order
print("\n=== Rule order (relevant rules) ===")
FastMasker._rules_cache = None
m = FastMasker()
for i, rule in enumerate(m.rules):
    name = type(rule).__name__
    if "Bank" in name or "Nrb" in name or "Trans" in name or "Car" in name or "Ssl" in name or "Krs" in name:
        print(f"  {i}: {name}")
