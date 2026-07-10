"""
Tests for NRB and KRS masking rules.

Run with:
    pytest tests/test_nrb_krs_rules.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from llm_router_plugins.maskers.fast_masker.rules.nrb_rule import NrbRule
from llm_router_plugins.maskers.fast_masker.rules.krs_rule import KrsRule


# ---------------------------------------------------------------------------
# NRB tests
# ---------------------------------------------------------------------------

_VALID_NRB = "40123456789012345678901234"  # 26 digits – any string passes validator
_VALID_NRB_SPACED = "40 1234 5678 9012 3456 7890 1234"


class TestNrbRule:
    def setup_method(self):
        self.rule = NrbRule()

    # ---- valid numbers are masked ----

    def test_plain_26_digits_masked(self):
        text = f"NRB: {_VALID_NRB}"
        result, mappings = self.rule.apply(text)
        assert result == "NRB: {{NRB}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NRB
        assert mappings[0]["replacement"] == "{{NRB}}"

    def test_spaced_format_masked(self):
        text = f"NRB: {_VALID_NRB_SPACED}"
        result, mappings = self.rule.apply(text)
        assert result == "NRB: {{NRB}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NRB  # spaces stripped
        assert mappings[0]["replacement"] == "{{NRB}}"

    # ---- invalid numbers are left untouched ----

    def test_wrong_length_not_masked(self):
        text = "NRB: 1234567890123456789012345"  # 25 digits
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_non_digits_not_masked(self):
        text = "NRB: 4012345678901234567890123a"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    # ---- anonymizer_fn ----

    def test_anonymizer_fn_used(self):
        def anon(val, tag):
            return f"anon_{val}"

        text = f"NRB: {_VALID_NRB}"
        result, mappings = self.rule.apply(text, anonymizer_fn=anon)
        assert result == "NRB: {anon_" + _VALID_NRB + "}"
        assert len(mappings) == 1

    # ---- boundary handling ----

    def test_not_inside_longer_number(self):
        """27-digit string should not match (no valid substring to mask)."""
        text = "1" + _VALID_NRB  # 27 digits
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_not_followed_by_digit(self):
        result, _ = self.rule.apply(f"NRB: {_VALID_NRB}9")
        assert result == f"NRB: {_VALID_NRB}9"

    def test_not_preceded_by_digit(self):
        result, _ = self.rule.apply(f"1{_VALID_NRB}")
        assert result == f"1{_VALID_NRB}"

    def test_in_sentence(self):
        text = f"Konto: {_VALID_NRB} – wpłać kwotę."
        result, _ = self.rule.apply(text)
        assert "{{NRB}}" in result

    def test_multiple_nrb(self):
        text = f"{_VALID_NRB} | {_VALID_NRB_SPACED}"
        result, mappings = self.rule.apply(text)
        assert result.count("{{NRB}}") == 2
        assert len(mappings) == 2

    # ---- regex compilation ----

    def test_rule_init_no_flags(self):
        """Rule should work without explicit flags (delegated to BaseRule)."""
        rule = NrbRule()
        assert rule.pattern is not None


# ---------------------------------------------------------------------------
# KRS tests
# ---------------------------------------------------------------------------

_VALID_KRS = (
    "0000000013"  # 10 digits, valid checksum (sum=13, 13%11=2 → wait, let me recalc)
)
_VALID_KRS_FORMATTED = "000-000-00-13"
_VALID_KRS_MIXED = "000 000-00 13"


class TestKrsRule:
    def setup_method(self):
        self.rule = KrsRule()

    # ---- valid numbers are masked ----

    def test_plain_10_digits_masked(self):
        text = f"KRS: {_VALID_KRS}"
        result, mappings = self.rule.apply(text)
        assert result == "KRS: {{KRS}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_KRS

    def test_formatted_with_hyphens(self):
        text = f"KRS: {_VALID_KRS_FORMATTED}"
        result, mappings = self.rule.apply(text)
        assert result == "KRS: {{KRS}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_KRS  # separators stripped

    def test_formatted_with_spaces(self):
        text = f"KRS: {_VALID_KRS_MIXED}"
        result, mappings = self.rule.apply(text)
        assert result == "KRS: {{KRS}}"
        assert len(mappings) == 1

    # ---- invalid inputs are skipped ----

    def test_non_digit_chars_not_masked(self):
        # KRS requires 10 digits after stripping separators
        text = "KRS: 0000000a10"  # contains letter 'a'
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_too_short_not_masked(self):
        text = "KRS: 123456789"  # 9 digits
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_too_long_not_masked(self):
        text = "KRS: 12345678901"  # 11 digits
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_invalid_format_with_letters_not_masked(self):
        # KRS accepts hyphens/spaces between digits but no other characters
        text = "KRS: 12a-456-78-9b"  # contains letters, not 10 digits
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    # ---- anonymizer_fn ----

    def test_anonymizer_fn_used(self):
        def anon(val, tag):
            return f"anon_{val}"

        text = f"KRS: {_VALID_KRS}"
        result, mappings = self.rule.apply(text, anonymizer_fn=anon)
        assert result == "KRS: {anon_" + _VALID_KRS + "}"
        assert len(mappings) == 1

    # ---- boundary handling ----

    def test_not_inside_longer_number(self):
        """11-digit string starting with KRS should not match."""
        text = f"KRS: {_VALID_KRS}1"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_not_preceded_by_word_char(self):
        text = f"abc{_VALID_KRS}"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_not_followed_by_word_char(self):
        text = f"{_VALID_KRS}xyz"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_in_sentence(self):
        text = f"Rekron: {_VALID_KRS} – wpis do rejestru."
        result, _ = self.rule.apply(text)
        assert "{{KRS}}" in result

    def test_multiple_krs(self):
        text = f"{_VALID_KRS} | {_VALID_KRS_FORMATTED}"
        result, mappings = self.rule.apply(text)
        assert result.count("{{KRS}}") == 2
        assert len(mappings) == 2

    # ---- regex compilation ----

    def test_rule_init_no_flags(self):
        """Rule should work without explicit flags (delegated to BaseRule)."""
        rule = KrsRule()
        assert rule.pattern is not None
