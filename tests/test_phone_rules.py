"""
Tests for Polish phone number masking rules.

Run with:
    pytest tests/test_phone_rules.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from llm_router_plugins.maskers.fast_masker.rules.phone_rule import PhoneRule
from llm_router_plugins.maskers.fast_masker.rules.phone_international_rule import (
    PhoneInternationalRule,
)


# ---------------------------------------------------------------------------
# Domestic Phone Rule tests
# ---------------------------------------------------------------------------


class TestPhoneDomesticRule:
    def setup_method(self):
        self.rule = PhoneRule()

    # ---- valid formats are masked ----

    def test_plain_9_digits_masked(self):
        """Plain 9-digit phone number."""
        text = "Tel: 599422765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE}}" in result
        assert len(mappings) == 1
        assert mappings[0]["original"] == "599422765"

    def test_grouped_by_3_spaces_masked(self):
        text = "Tel: 592 722 765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE}}" in result
        assert len(mappings) == 1
        assert mappings[0]["original"] == "592 722 765"

    def test_grouped_by_3_dashes_masked(self):
        text = "Tel: 592-722-765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE}}" in result
        assert len(mappings) == 1
        assert mappings[0]["original"] == "592-722-765"

    def test_mixed_separators_masked(self):
        text = "Tel: 592-722 765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE}}" in result
        assert len(mappings) == 1
        assert mappings[0]["original"] == "592-722 765"

    def test_grouped_2x5_format_masked(self):
        """Traditional Polish grouping: XX XX XX XX X."""
        text = "Tel: 59 94 22 76 5"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE}}" in result
        assert len(mappings) == 1

    def test_in_sentence(self):
        text = "Skontaktuj sie pod numerem 599422765 w sprawie oferty."
        result, _ = self.rule.apply(text)
        assert "{{PHONE}}" in result

    # ---- invalid formats are NOT masked ----

    def test_8_digits_not_masked(self):
        text = "Num: 59942276"  # 8 digits
        result, _ = self.rule.apply(text)
        assert result == text

    def test_10_digits_not_masked(self):
        text = "Num: 5994227622"  # 10 digits
        result, _ = self.rule.apply(text)
        assert result == text

    def test_11_digits_not_masked(self):
        text = "Num: 59942276221"  # 11 digits
        result, _ = self.rule.apply(text)
        assert result == text

    # ---- boundary handling ----

    def test_not_inside_longer_number(self):
        """NRB-like string should not be matched by PhoneRule."""
        text = "Num: 15923456789012345678901234"  # 26 digits
        result, _ = self.rule.apply(text)
        assert "{{PHONE}}" not in result

    def test_not_preceded_by_digit(self):
        text = "Num: 3599422765"  # digit before
        result, _ = self.rule.apply(text)
        assert result == text

    def test_not_followed_by_digit(self):
        text = "Num: 5994227659"  # digit after
        result, _ = self.rule.apply(text)
        assert result == text

    def test_at_string_start(self):
        text = "599422765 done"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE}}" in result
        assert len(mappings) == 1

    def test_at_string_end(self):
        text = "Call 599422765"
        result, _ = self.rule.apply(text)
        assert "{{PHONE}}" in result

    # ---- multiple phones ----

    def test_two_phones_same_text(self):
        text = "599422765 i 600123456"
        result, mappings = self.rule.apply(text)
        assert result.count("{{PHONE}}") == 2
        assert len(mappings) == 2

    def test_mixed_formats_in_text(self):
        text = "599422765 | 600 123 456 | 720-987-654"
        result, mappings = self.rule.apply(text)
        assert result.count("{{PHONE}}") == 3
        assert len(mappings) == 3

    # ---- anonymizer_fn ----

    def test_anonymizer_fn_used(self):
        def anon(val, tag):
            return f"anon_{val}"

        text = "Tel: 599422765"
        result, mappings = self.rule.apply(text, anonymizer_fn=anon)
        assert len(mappings) == 1
        assert mappings[0]["original"] == "599422765"

    # ---- regex compilation ----

    def test_rule_init_compiles(self):
        rule = PhoneRule()
        assert rule.pattern is not None


# ---------------------------------------------------------------------------
# International Phone Rule tests
# ---------------------------------------------------------------------------


class TestPhoneInternationalRule:
    def setup_method(self):
        self.rule = PhoneInternationalRule()

    # ---- valid formats are masked ----

    def test_polish_intl_no_space_masked(self):
        """+48 + 9 digits, no spaces."""
        text = "Tel: +48599422765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result
        assert len(mappings) == 1
        assert mappings[0]["original"] == "+48599422765"

    def test_polish_intl_with_spaces_masked(self):
        text = "Tel: +48 592 722 765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result
        assert len(mappings) == 1
        assert mappings[0]["original"] == "+48 592 722 765"

    def test_polish_intl_mixed_seps_masked(self):
        text = "Tel: +48 592-722-765"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result
        assert len(mappings) == 1

    def test_polish_intl_dashes_masked(self):
        text = "Tel: +48-592-722-765"
        result, _ = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result

    def test_german_number_masked(self):
        """+49 (Germany) subscriber number."""
        text = "Tel: +49 170 123456789"
        result, _ = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result

    def test_uk_number_masked(self):
        """+44 (UK) subscriber number."""
        text = "Tel: +44 20 7946 0958"
        result, _ = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result

    def test_at_string_start(self):
        text = "+48599422765 done"
        result, mappings = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result
        assert len(mappings) == 1

    def test_continuous_digits_at_start_of_string(self):
        """Continuous digits without preceding space should still match."""
        text = "+48599422765"
        result, _ = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result

    # ---- invalid formats are NOT masked ----

    def test_no_plus_not_masked(self):
        """Plain number without + should not match international rule."""
        text = "Tel: 48599422765"
        result, _ = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" not in result

    def test_plus_alone_not_masked(self):
        """+48 alone (no subscriber digits)."""
        text = "Tel: +48"
        result, _ = self.rule.apply(text)
        assert result == text

    # ---- boundary handling ----

    def test_at_string_boundary(self):
        text = "+48 592 722 765"
        result, _ = self.rule.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result

    def test_adjacent_to_paren(self):
        """Phone inside parens — paren is non-whitespace so (?<!\\S) prevents match."""
        text = "Call (+48 592 722 765) now"
        result, mappings = self.rule.apply(text)
        # The '(' before + is non-whitespace, so the pattern does not match.
        assert "{{PHONE_INTERNATIONAL}}" not in result

    # ---- anonymizer_fn ----

    def test_anonymizer_fn_used(self):
        def anon(val, tag):
            return f"anon_{val}"

        text = "Tel: +48599422765"
        result, mappings = self.rule.apply(text, anonymizer_fn=anon)
        assert len(mappings) == 1

    # ---- regex compilation ----

    def test_rule_init_compiles(self):
        rule = PhoneInternationalRule()
        assert rule.pattern is not None


# ---------------------------------------------------------------------------
# Integration tests with both rules active
# ---------------------------------------------------------------------------


class TestPhoneRulesIntegration:
    """Verify both rules work correctly together."""

    def setup_method(self):
        self.rule_dom = PhoneRule()
        self.rule_intl = PhoneInternationalRule()

    def test_intl_masks_first_not_consumed_by_domestic(self):
        """International number masked by intl rule only, not domestic."""
        text = "+48 599422765"
        result_intl, _ = self.rule_intl.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" in result_intl
        result_dom, _ = self.rule_dom.apply(result_intl)
        assert result_dom == result_intl  # unchanged

    def test_plain_phone_only_matched_by_domestic(self):
        """Plain 9-digit phone is domestic-only."""
        text = "Tel: 599422765"
        result_intl, _ = self.rule_intl.apply(text)
        assert "{{PHONE_INTERNATIONAL}}" not in result_intl
        result_dom, _ = self.rule_dom.apply(result_intl)
        assert "{{PHONE}}" in result_dom

    def test_both_numbers_in_text(self):
        """Both intl and domestic numbers masked correctly."""
        text = "599422765 i +48 600123456"
        result_intl, mappings_intl = self.rule_intl.apply(text)
        result_dom, mappings_dom = self.rule_dom.apply(result_intl)
        assert "{{PHONE_INTERNATIONAL}}" in result_dom
        assert "{{PHONE}}" in result_dom
        # Check counts
        assert (
            result_dom.count("{{PHONE}}")
            + result_dom.count("{{PHONE_INTERNATIONAL}}")
            == 2
        )
