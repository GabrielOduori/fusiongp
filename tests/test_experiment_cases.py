"""
Tests for multi-case experiment definitions.
"""

from experiments.reproduce_paper import build_multi_case_definitions


def test_multi_case_definitions():
    cases = build_multi_case_definitions()
    assert len(cases) == 5

    names = [c["name"] for c in cases]
    assert "Case 0 (EPA only)" in names
    assert "Case 1 (Prior only)" in names
    assert "Case 4 (Prior + LCS + Satellite)" in names

    for case in cases:
        assert "sources" in case
        assert "mode" in case
        assert "use_prior" in case
        assert case["mode"] in {"train", "prior_only"}
