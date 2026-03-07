"""
Tests for multi-case experiment definitions.
"""

from experiments.reproduce_paper import build_multi_case_definitions


def test_multi_case_definitions():
    cases = build_multi_case_definitions()
    assert len(cases) == 4

    names = [c["name"] for c in cases]
    assert "Case 0 (EPA only)" in names
    assert "Case 1 (LUR prior only)" in names
    assert "Case 3 (LUR prior + EPA + Satellite)" in names

    for case in cases:
        assert "sources" in case
        assert "mode" in case
        assert "use_prior" in case
        assert case["mode"] in {"train", "prior_only"}
        # LCS (low-cost sensors) not available in this dataset
        assert "low_cost" not in case["sources"]
