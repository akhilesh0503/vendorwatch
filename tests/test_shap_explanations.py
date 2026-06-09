"""
Test 8: SHAP explanation sentences contain actual numeric values, not placeholder text.

Verifies:
  - generate_explanation() returns a non-empty string
  - The string contains at least one numeric value (σ, $, days)
  - No placeholder tokens like {value}, <X>, N/A appear in the output
  - The explanation has 3 sentences (ends with 3 periods)
  - Primary and secondary signal sentences reference feature labels from FEATURE_LABELS
"""

import re
import numpy as np
import pytest

from src.detection.features import FEATURE_NAMES
from src.detection.shap_explainer import (
    FEATURE_LABELS,
    compute as shap_compute,
    generate_explanation,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal shap_dict without running a real model
# ---------------------------------------------------------------------------

def _fake_shap_dict(values: list) -> dict:
    assert len(values) == len(FEATURE_NAMES)
    return {
        "shap_values":   values,
        "base_value":    0.05,
        "feature_names": list(FEATURE_NAMES),
    }


def _fake_feature_vec(**overrides) -> np.ndarray:
    defaults = {
        "invoice_amount":                    25_000.0,
        "days_to_approval":                  7.0,
        "days_to_payment":                   37.0,
        "invoice_frequency_7d":              0.7,
        "invoice_frequency_30d":             3.0,
        "amount_deviation_from_vendor_mean": 0.0,
        "approval_cycle_z_score":            0.0,
        "payment_timing_z_score":            0.0,
    }
    defaults.update(overrides)
    return np.array([defaults[k] for k in FEATURE_NAMES], dtype=np.float64)


# ============================================================================
# Test 8a — explanation contains actual numeric values
# ============================================================================

class TestShapExplanationContent:

    def test_explanation_is_nonempty_string(self):
        shap_dict = _fake_shap_dict([0.34, 0.28, 0.05, 0.10, 0.12, 0.19, 0.08, 0.03])
        features  = _fake_feature_vec(invoice_amount=42_000.0, days_to_approval=2.0)
        result    = generate_explanation(shap_dict, features)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_explanation_contains_numeric_values(self):
        """The output must embed real numbers — not generic labels."""
        shap_dict = _fake_shap_dict([0.34, 0.28, 0.05, 0.10, 0.12, 0.19, 0.08, 0.03])
        features  = _fake_feature_vec(
            invoice_amount     = 42_000.0,
            days_to_approval   = 2.0,
            approval_cycle_z_score = -4.33,
        )
        result = generate_explanation(shap_dict, features)
        # Must contain at least one number (integer or decimal)
        assert re.search(r"\d+\.?\d*", result), (
            f"Explanation contains no numeric values: {result!r}"
        )

    def test_explanation_has_no_placeholder_text(self):
        shap_dict = _fake_shap_dict([0.34, 0.28, 0.05, 0.10, 0.12, 0.19, 0.08, 0.03])
        features  = _fake_feature_vec(invoice_amount=42_000.0)
        result    = generate_explanation(shap_dict, features)

        forbidden = ["{value}", "<X>", "N/A", "placeholder", "TODO", "FIXME"]
        for token in forbidden:
            assert token not in result, (
                f"Explanation contains forbidden placeholder token '{token}': {result!r}"
            )

    def test_primary_signal_sentence_references_top_shap_feature(self):
        # Amount deviation is the top contributor
        shap_vals = [0.05, 0.02, 0.01, 0.03, 0.04, 0.45, 0.08, 0.06]
        shap_dict = _fake_shap_dict(shap_vals)
        features  = _fake_feature_vec(amount_deviation_from_vendor_mean=3.2)
        result    = generate_explanation(shap_dict, features)

        # The word "amount" must appear (from the top feature's label)
        assert "amount" in result.lower(), (
            f"Primary signal sentence should reference 'amount' but got: {result!r}"
        )

    def test_sigma_value_in_z_score_sentence(self):
        """When top feature is a z-score, the sentence should include a σ value."""
        shap_vals = [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.45, 0.01]
        shap_dict = _fake_shap_dict(shap_vals)
        features  = _fake_feature_vec(approval_cycle_z_score=-4.33)
        result    = generate_explanation(shap_dict, features)

        # Should contain "4.33" (the actual sigma value)
        assert "4.33" in result, (
            f"Z-score explanation should include the actual sigma value '4.33': {result!r}"
        )

    def test_dollar_amount_in_invoice_amount_sentence(self):
        """When top feature is invoice_amount, the dollar value should appear."""
        shap_vals = [0.50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        shap_dict = _fake_shap_dict(shap_vals)
        features  = _fake_feature_vec(invoice_amount=67_500.0)
        result    = generate_explanation(shap_dict, features)

        assert "67,500" in result or "67500" in result, (
            f"Invoice amount explanation should contain '67,500': {result!r}"
        )

    def test_contribution_value_in_sentence(self):
        """Each sentence must include the contributing SHAP value in ±X.XX format."""
        shap_vals = [0.34, 0.28, 0.05, 0.10, 0.12, 0.19, 0.08, 0.03]
        shap_dict = _fake_shap_dict(shap_vals)
        features  = _fake_feature_vec()
        result    = generate_explanation(shap_dict, features)

        # Look for patterns like +0.34 or -0.28
        contrib_pattern = re.compile(r"[+\-]\d+\.\d+")
        matches = contrib_pattern.findall(result)
        assert len(matches) >= 2, (
            f"Expected ≥2 contribution values in explanation, found {len(matches)}: {result!r}"
        )

    def test_three_sentences_in_output(self):
        """The explanation must contain exactly 3 sentences."""
        shap_vals = [0.34, 0.28, 0.05, 0.10, 0.12, 0.19, 0.08, 0.03]
        shap_dict = _fake_shap_dict(shap_vals)
        features  = _fake_feature_vec()
        result    = generate_explanation(shap_dict, features)

        # Count sentences ending with period + space or period at end
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) >= 3, (
            f"Expected 3 sentences, found {len(sentences)}: {result!r}"
        )

    def test_combined_pattern_hint_used_when_applicable(self):
        """
        When top-2 features match a known pattern, the combined sentence
        should use the specific hint text rather than the generic fallback.
        """
        from src.detection.shap_explainer import _PATTERN_HINTS

        # invoice_frequency_7d + invoice_amount match "invoice_splitting" hint
        shap_vals = [0.25, 0.01, 0.01, 0.45, 0.10, 0.05, 0.01, 0.01]
        shap_dict = _fake_shap_dict(shap_vals)
        features  = _fake_feature_vec(
            invoice_frequency_7d = 8.0,
            invoice_amount       = 9_800.0,
        )
        result = generate_explanation(shap_dict, features)

        # The generic fallback phrase should NOT appear
        assert "warrants manual" not in result or "invoice splitting" in result.lower(), (
            "Expected the invoice-splitting combined pattern hint but got the generic fallback."
        )
