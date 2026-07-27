"""
Unit tests for ``BankAccountRule`` (IBAN masking).

Covers: compact / spaced / dashed separators, cross-format equivalence,
partial ``X``-masking, multi-country IBANs, false-positive rejection and
boundary conditions.  Mirrors the patterns used in ``test_nrb_krs_rules.py``
and ``test_phone_rules.py``.
"""

import pytest

from llm_router_plugins.maskers.fast_masker.rules.bank_account_rule import (
    BankAccountRule,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture()
def rule() -> BankAccountRule:
    return BankAccountRule()


# ── Helpers ────────────────────────────────────────────────────────────────


def _masked_text(rule: BankAccountRule, text: str) -> str:
    """Return the masked result string."""
    result, mappings = rule.apply(text)
    return result


# ── Polish IBAN – all six spacing variants ─────────────────────────────────

_PL_TEST_IBAN = "PL73102000105799000000000000"


class TestPolishIbanFormats:
    """Each line is one spacing variant; all must mask identically."""

    _TEST_CASES = [
        ("F1", "space after PL, groups of 4", "PL 73 1020 0010 5799 0000 0000 0000"),
        ("F2", "fully compact (default)", _PL_TEST_IBAN),
        ("F3", "dashed groups of 4", "PL-73-1020-0010-5799-0000-0000-0000"),
        (
            "F4",
            "space after PL, BBAN groups of 8",
            "PL 73 10200010 57990000 00000000",
        ),
        (
            "F5",
            "compact country+check, rest spaced by 4",
            "PL73 1020 0010 5799 0000 0000 0000",
        ),
        (
            "F6",
            "space after PL, bank as first 4-group",
            "PL 7310 200010 57990000 00000000",
        ),
    ]

    @pytest.mark.parametrize("fmt,desc,text", _TEST_CASES)
    def test_all_formats_mask(self, rule: BankAccountRule, fmt, desc, text: str):
        result = _masked_text(rule, text)
        assert (
            "BANK_ACCOUNT" in result
        ), f"{fmt} ({desc}): expected BANK_ACCOUNT mask but got: {result}"

    @pytest.mark.parametrize("fmt,desc,text", _TEST_CASES)
    def test_all_formats_no_leaked_tokens(
        self, rule: BankAccountRule, fmt, desc, text: str
    ):
        result, mappings = rule.apply(text)
        for tag in ("SIM_CARD", "CREDIT_CARD"):
            assert (
                tag not in result
            ), f"{fmt} ({desc}): leaked {tag} into bank account: {result}"


# ── Cross-format equivalence ───────────────────────────────────────────────


class TestCrossFormatEquivalence:
    """Different formats of the same IBAN should all produce a mask."""

    @pytest.mark.parametrize(
        "iban",
        [
            "PL73102000105799000000000000",
            "PL79801001060000042270201111",
            "PL67160000000000042270201111",
        ],
    )
    def test_compact_masks(self, rule: BankAccountRule, iban: str):
        result = _masked_text(rule, iban)
        assert "BANK_ACCOUNT" in result

    @pytest.mark.parametrize(
        "iban",
        [
            "PL73102000105799000000000000",
            "PL79801001060000042270201111",
            "PL67160000000000042270201111",
        ],
    )
    def test_spaced_masks(self, rule: BankAccountRule, iban: str):
        compact = iban[:4] + iban[4:]
        spaced = (
            f"{iban[:2]} {iban[2:6]} "
            f"{' '.join(compact[i:i+4] for i in range(6, len(compact), 4))}"
        )
        result = _masked_text(rule, spaced)
        assert "BANK_ACCOUNT" in result, f"Spaced IBAN not masked: {spaced}"

    @pytest.mark.parametrize(
        "iban",
        [
            "PL73102000105799000000000000",
            "PL79801001060000042270201111",
            "PL67160000000000042270201111",
        ],
    )
    def test_dashed_masks(self, rule: BankAccountRule, iban: str):
        compact = iban[:4] + iban[4:]
        dashed = (
            f"{iban[:2]}-{iban[2:6]}-"
            f"{'-'.join(compact[i:i+4] for i in range(6, len(compact), 4))}"
        )
        result = _masked_text(rule, dashed)
        assert "BANK_ACCOUNT" in result, f"Dashed IBAN not masked: {dashed}"

    @pytest.mark.parametrize(
        "iban",
        [
            "PL73102000105799000000000000",
            "PL79801001060000042270201111",
            "PL67160000000000042270201111",
        ],
    )
    def test_mixed_spacing_masks(self, rule: BankAccountRule, iban: str):
        text = f"{iban[:2]} {iban[2:]}"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result, f"Mixed spaced IBAN not masked: {text}"


# ── Partial masking with X characters ──────────────────────────────────────


class TestPartialMasking:
    """IBANs with X in groups should still be detected (checksum skipped)."""

    @pytest.mark.parametrize(
        "text",
        [
            # Compact: exactly 28 chars total (PL + XX check digits + 24 alnum/X BBAN)
            "PLXX10XX5799XXXX00000000ABCD",  # 28 — X in check digits, X in BBAN
        ],
    )
    def test_x_masked(self, rule: BankAccountRule, text: str):
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result, f"X-masked IBAN not detected: {result}"

    @pytest.mark.parametrize(
        "text",
        [
            "PLXX10XX5799XXXX00000000ABCD",
        ],
    )
    def test_x_no_leaked_token(self, rule: BankAccountRule, text: str):
        result, _ = rule.apply(text)
        for tag in ("SIM_CARD", "CREDIT_CARD"):
            assert tag not in result, f"leaked {tag}: {result}"

    def test_dashed_x_masked(self, rule: BankAccountRule):
        """Dashed format with X check digits should also work."""
        text = "PL-XX-10XX-5799-XXXX-0000-ABCD"  # strips to 28 alnum/X
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result, f"Dashed X-masked not detected: {result}"

    def test_spaced_x_masked(self, rule: BankAccountRule):
        """Spaced format with X check digits should also work."""
        text = "PL 10 10XX 5799 XXXX 0000 AB CD"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result, f"Spaced X-masked not detected: {result}"


# ── Multi-country IBANs ───────────────────────────────────────────────────


class TestMultiCountryIban:
    """IBANs from different countries should all be masked correctly."""

    # Valid IBAN test data (checksum verified)
    _TEST_CASES = [
        ("PL", "PL73102000105799000000000000"),  # mBank, length 28
        ("DE", "DE89370400440532013000"),  # Deutsche Bank, length 22
        ("GB", "GB29NWBK60161331926819"),  # NatWest, length 22
    ]

    @pytest.mark.parametrize("country,iban", _TEST_CASES)
    def test_compact_mask(self, rule: BankAccountRule, country: str, iban: str):
        result = _masked_text(rule, iban)
        assert "BANK_ACCOUNT" in result, f"{country} compact not masked: {result}"

    @pytest.mark.parametrize("country,iban", _TEST_CASES)
    def test_spaced_mask(self, rule: BankAccountRule, country: str, iban: str):
        """IBAN with spaces between groups of 4 after the check digits."""
        compact = iban[:4] + iban[4:]
        spaced = (
            f"{iban[:2]} {iban[2:6]} "
            f"{' '.join(compact[i:i+4] for i in range(6, len(compact), 4))}"
        )
        result = _masked_text(rule, spaced)
        assert "BANK_ACCOUNT" in result, f"{country} spaced not masked: {spaced}"

    @pytest.mark.parametrize("country,iban", _TEST_CASES)
    def test_dashed_mask(self, rule: BankAccountRule, country: str, iban: str):
        """IBAN with hyphens between groups of 4 after the check digits."""
        compact = iban[:4] + iban[4:]
        dashed = (
            f"{iban[:2]}-{iban[2:6]}-"
            f"{'-'.join(compact[i:i+4] for i in range(6, len(compact), 4))}"
        )
        result = _masked_text(rule, dashed)
        assert "BANK_ACCOUNT" in result, f"{country} dashed not masked: {dashed}"

    @pytest.mark.parametrize("country,iban", _TEST_CASES)
    def test_mixed_mask(self, rule: BankAccountRule, country: str, iban: str):
        """IBAN with space after country+check digits only."""
        text = f"{iban[:2]} {iban[2:]}"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result, f"{country} mixed not masked: {text}"


# ── False-positive rejection ───────────────────────────────────────────────


class TestFalsePositiveRejection:
    """Invalid IBANs must NOT be masked."""

    def test_wrong_length_short(self, rule: BankAccountRule):
        text = "PL171020001057990000000000"  # 26 chars — too short for PL
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" not in result

    def test_wrong_length_long(self, rule: BankAccountRule):
        text = "PL171020001057990000000000000"  # 29 chars — too long for PL
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" not in result

    def test_unknown_country_code(self, rule: BankAccountRule):
        text = "XX17102000105799000000000000"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" not in result

    def test_invalid_checksum(self, rule: BankAccountRule):
        """PL IBAN with wrong check digits → no mask."""
        # Correct check for 171020001057990000000000 is "17" (PL17...)
        text = "PL16102000105799000000000000"  # wrong check digit
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" not in result

    def test_plain_digits_no_mask(self, rule: BankAccountRule):
        text = "123456789012345678901234"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" not in result


# ── Boundary and context tests ─────────────────────────────────────────────


class TestBoundaryAndContext:
    """Masking should work correctly in various text contexts."""

    def test_in_sentence(self, rule: BankAccountRule):
        text = "Przelij na konto PL73102000105799000000000000 w mBanku"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result

    def test_multiple_iban_with_separator(self, rule: BankAccountRule):
        iban1 = "PL73102000105799000000000000"
        iban2 = "PL79801001060000042270201111"
        # Use newline as separator to avoid anchor collision
        text = f"{iban1}\n{iban2}"
        result = _masked_text(rule, text)
        assert (
            result.count("BANK_ACCOUNT") == 2
        ), f"Expected 2 masks but got {result.count('BANK_ACCOUNT')}: {result}"

    def test_at_string_start(self, rule: BankAccountRule):
        text = "PL73102000105799000000000000"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result

    def test_at_string_end(self, rule: BankAccountRule):
        text = "Sprawdź PL73102000105799000000000000"
        result = _masked_text(rule, text)
        assert "BANK_ACCOUNT" in result


# ── Anonymizer function tests ──────────────────────────────────────────────


class TestAnonymizerFunction:
    """Custom anonymizer fn should produce dynamic pseudonyms."""

    def test_dynamic_pseudonym(self, rule: BankAccountRule):
        iban = "PL73102000105799000000000000"
        result, mappings = rule.apply(iban, anonymizer_fn=lambda v, t: f"DYN_{t}")
        assert "DYN_BANK_ACCOUNT" in result

    def test_mapping_contains_original(self, rule: BankAccountRule):
        iban = "PL73102000105799000000000000"
        _, mappings = rule.apply(iban, anonymizer_fn=lambda v, t: f"P_{t}")
        assert len(mappings) == 1
        assert mappings[0]["original"] == iban

    def test_default_placeholder(self, rule: BankAccountRule):
        """Without anonymizer fn, placeholder is the static string."""
        iban = "PL73102000105799000000000000"
        result, _ = rule.apply(iban)
        assert "{{BANK_ACCOUNT}}" in result
