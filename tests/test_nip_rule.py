"""
Tests for NIP masking rule.

Run with:
    pytest tests/test_nip_rule.py -v
"""

import pytest
from llm_router_plugins.maskers.fast_masker.rules.nip_rule import NipRule

# Valid Polish NIPs:
# 1181934272 -> sum=189, 189%11=2 -> valid
# 5260250995 -> sum=148, 148%11=5 -> valid
_VALID_NIP = "1181934272"
_VALID_NIP_3322_HYPHEN = "118-193-42-72"
_VALID_NIP_3322_SPACE = "118 193 42 72"
_VALID_NIP_3223_HYPHEN = "118-19-34-272"
_VALID_NIP_3223_SPACE = "118 19 34 272"


class TestNipRule:
    def setup_method(self):
        self.rule = NipRule()

    # ---- valid numbers are masked ----

    def test_plain_10_digits_masked(self):
        text = f"NIP: {_VALID_NIP}"
        result, mappings = self.rule.apply(text)
        assert result == "NIP: {{NIP}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NIP

    def test_formatted_3322_with_hyphens(self):
        text = f"NIP {_VALID_NIP_3322_HYPHEN}"
        result, mappings = self.rule.apply(text)
        assert result == "NIP {{NIP}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NIP_3322_HYPHEN

    def test_formatted_3322_with_spaces(self):
        text = f"NIP {_VALID_NIP_3322_SPACE}"
        result, mappings = self.rule.apply(text)
        assert result == "NIP {{NIP}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NIP_3322_SPACE

    def test_formatted_3223_with_hyphens(self):
        text = f"NIP: {_VALID_NIP_3223_HYPHEN}"
        result, mappings = self.rule.apply(text)
        assert result == "NIP: {{NIP}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NIP_3223_HYPHEN

    def test_formatted_3223_with_spaces(self):
        text = f"NIP {_VALID_NIP_3223_SPACE}"
        result, mappings = self.rule.apply(text)
        assert result == "NIP {{NIP}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NIP_3223_SPACE

    def test_with_markdown_markers(self):
        text = f"NIP **{_VALID_NIP_3322_SPACE}**"
        result, mappings = self.rule.apply(text)
        assert result == "NIP **{{NIP}}**"
        assert len(mappings) == 1

    # ---- invalid inputs are skipped ----

    def test_invalid_checksum_not_masked(self):
        # 1181934273 has invalid checksum
        text = "NIP: 118 193 42 73"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_too_short_not_masked(self):
        text = "NIP: 123-456-78-9"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    def test_too_long_not_masked(self):
        text = "NIP: 118-193-42-720"
        result, mappings = self.rule.apply(text)
        assert result == text
        assert mappings == []

    # ---- anonymizer_fn ----

    def test_anonymizer_fn_used(self):
        def anon(val, tag):
            return f"anon_{val}"

        text = f"NIP: {_VALID_NIP_3322_SPACE}"
        result, mappings = self.rule.apply(text, anonymizer_fn=anon)
        assert result == f"NIP: {{anon_{_VALID_NIP_3322_SPACE}}}"
        assert len(mappings) == 1
        assert mappings[0]["original"] == _VALID_NIP_3322_SPACE


class TestNipKrsPipelineInteraction:
    """Ensure NIP formats are not hijacked by KRS rule in FastMasker pipeline."""

    def test_spaced_nip_masked_as_nip_not_krs(self):
        from llm_router_plugins.maskers.fast_masker.core.masker import FastMasker

        masker = FastMasker()
        text = "Firma ABC, NIP 118 193 42 72, wpis do rejestru."
        masked_text, mappings = masker.mask(text)

        assert "NIP_" in masked_text
        assert "KRS_" not in masked_text
        assert "118 193 42 72" in masker.mapping

    def test_invalid_spaced_nip_not_masked_as_krs(self):
        from llm_router_plugins.maskers.fast_masker.core.masker import FastMasker

        masker = FastMasker()
        text = "Niepoprawny NIP 118 193 42 73 w dokumencie."
        masked_text, mappings = masker.mask(text)

        assert "KRS_" not in masked_text
        assert "NIP_" not in masked_text
        assert masked_text == text
        assert mappings == {}
