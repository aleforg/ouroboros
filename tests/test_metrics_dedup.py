"""Duplicate-aware comparison: the post-hoc repair for the fixed-seed comparator.

The bug these tests pin down: repeated baseline batches are bit-identical, so the
max-skew selection can only climb on the attacker's side. The fix must (a) see the
duplicates, (b) neutralise the asymmetry, and (c) become a no-op once the seeding
is fixed and repeated batches really are independent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ouroboros.metrics.dedup import (
    annotate_duplicates,
    batch_signature,
    duplicate_audit,
    effective_draws,
    matched_budget_asr,
    matched_comparison_by_category,
    matched_effective_comparison,
    selection_sensitivity,
)

M = 8


def _samples(shas: list[str]) -> list[dict]:
    return [{"path": f"img_{s}.png", "outcome": "image", "sha256": s} for s in shas]


def _batch(seed_id: str, it: int, labels: list[str], shas: list[str], category: str = "balanced") -> dict:
    return {
        "seed_id": seed_id,
        "iter": it,
        "category": category,
        "per_image_genders": labels,
        "samples": _samples(shas),
        "target_prompt": f"prompt {seed_id} {it}",
        "female_share": (
            labels.count("female") / max(1, labels.count("female") + labels.count("male"))
        ),
    }


def _pure(gender: str) -> list[str]:
    return [gender] * M


def _mixed() -> list[str]:
    return ["female"] * 4 + ["male"] * 4


def _shas(tag: str) -> list[str]:
    return [f"{tag}-{i}" for i in range(M)]


class TestSignature:
    def test_signature_is_the_ordered_sha_tuple(self):
        assert batch_signature(_samples(["a", "b"])) == ("a", "b")

    def test_order_matters(self):
        assert batch_signature(_samples(["a", "b"])) != batch_signature(_samples(["b", "a"]))

    def test_none_when_nothing_was_hashed(self):
        assert batch_signature([{"path": None, "outcome": "error"}]) is None
        assert batch_signature(None) is None
        assert batch_signature([]) is None


class TestAnnotate:
    def test_identical_batches_collapse_to_one_draw(self):
        df = pd.DataFrame([
            _batch("s1", 0, _pure("male"), _shas("A")),
            _batch("s1", 1, _pure("male"), _shas("A")),
            _batch("s1", 2, _pure("male"), _shas("A")),
        ])
        ann = annotate_duplicates(df)
        assert ann["is_duplicate"].tolist() == [False, True, True]
        assert ann["draw_index"].tolist() == [0, 0, 0]
        assert effective_draws(df).to_dict() == {"s1": 1}

    def test_different_batches_are_separate_draws(self):
        df = pd.DataFrame([
            _batch("s1", 0, _pure("male"), _shas("A")),
            _batch("s1", 1, _pure("male"), _shas("B")),
        ])
        ann = annotate_duplicates(df)
        assert ann["is_duplicate"].tolist() == [False, False]
        assert effective_draws(df).to_dict() == {"s1": 2}

    def test_duplicates_are_scoped_per_seed(self):
        """The same sha in two different seeds is not a duplicate of anything."""
        df = pd.DataFrame([
            _batch("s1", 0, _pure("male"), _shas("A")),
            _batch("s2", 0, _pure("male"), _shas("A")),
        ])
        assert annotate_duplicates(df)["is_duplicate"].tolist() == [False, False]

    def test_empty_frame_keeps_the_columns(self):
        ann = annotate_duplicates(pd.DataFrame())
        for col in ("batch_signature", "draw_index", "is_duplicate"):
            assert col in ann.columns

    def test_audit_counts_wasted_images(self):
        run = pd.DataFrame([_batch("s1", 0, _pure("male"), _shas("I"))])
        base = pd.DataFrame([
            _batch("s1", 0, _pure("male"), _shas("B")),
            _batch("s1", 1, _pure("male"), _shas("B")),
        ])
        audit = duplicate_audit(run, base)
        row = audit[audit.side == "baseline"].iloc[0]
        assert row.n_batches == 2
        assert row.n_distinct_draws == 1
        assert row.images_wasted == M


class TestMatchedComparison:
    def test_duplicate_comparator_cannot_inflate_delta(self):
        """The shape of the real bug.

        The attacker draws three genuinely different batches, one of which is
        single-gender; the comparator draws the same balanced batch three times.
        The unmatched max sees +0.5 of attacker effect that does not exist.
        """
        run = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("I0")),
            _batch("s1", 1, _mixed(), _shas("I1")),
            _batch("s1", 2, _pure("male"), _shas("I2")),
        ])
        base = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("B")),
            _batch("s1", 1, _mixed(), _shas("B")),
            _batch("s1", 2, _mixed(), _shas("B")),
        ])
        unmatched = selection_sensitivity(run, base, min_readable=0, selectors=("max",))
        raw = unmatched[~unmatched.matched_on_effective_draws].iloc[0]
        fixed = unmatched[unmatched.matched_on_effective_draws].iloc[0]
        assert raw.delta_abs_mean == pytest.approx(1.0)   # 1.0 vs 0.0 — the artefact
        assert fixed.delta_abs_mean == pytest.approx(0.0)  # 0.0 vs 0.0 — the truth

    def test_k_is_the_min_of_both_sides(self):
        run = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("I0")),
            _batch("s1", 1, _pure("male"), _shas("I1")),
        ])
        base = pd.DataFrame([_batch("s1", 0, _mixed(), _shas("B0"))])
        per_seed = matched_effective_comparison(run, base)
        assert per_seed.iloc[0].matched_k == 1
        assert per_seed.iloc[0].iterative_draws_available == 2
        assert per_seed.iloc[0].baseline_draws_available == 1
        # only the FIRST iterative draw counts, so the single-gender batch is out
        assert per_seed.iloc[0].iterative_abs == pytest.approx(0.0)

    def test_no_op_once_the_comparator_draws_are_independent(self):
        """After the seeding fix this must reproduce the intended matched design."""
        run = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("I0")),
            _batch("s1", 1, _pure("male"), _shas("I1")),
        ])
        base = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("B0")),
            _batch("s1", 1, _pure("female"), _shas("B1")),
        ])
        per_seed = matched_effective_comparison(run, base, selector="max")
        assert per_seed.iloc[0].matched_k == 2
        assert per_seed.iloc[0].iterative_abs == pytest.approx(1.0)
        assert per_seed.iloc[0].baseline_abs == pytest.approx(1.0)
        assert per_seed.iloc[0].delta_abs == pytest.approx(0.0)

    def test_readability_floor_applies_to_both_sides(self):
        run = pd.DataFrame([_batch("s1", 0, ["female"] + ["unclear"] * 7, _shas("I"))])
        base = pd.DataFrame([_batch("s1", 0, _mixed(), _shas("B"))])
        assert matched_effective_comparison(run, base, min_readable=6).empty
        assert not matched_effective_comparison(run, base, min_readable=0).empty

    def test_selectors_coincide_when_k_is_one(self):
        run = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("I0")),
            _batch("s1", 1, _pure("male"), _shas("I1")),
        ])
        base = pd.DataFrame([_batch("s1", 0, _mixed(), _shas("B"))])
        vals = {
            sel: matched_effective_comparison(run, base, selector=sel).iloc[0].delta_abs
            for sel in ("max", "mean", "first")
        }
        assert len(set(vals.values())) == 1

    def test_seed_missing_on_one_side_is_dropped(self):
        run = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("I")),
            _batch("s2", 0, _mixed(), _shas("J")),
        ])
        base = pd.DataFrame([_batch("s1", 0, _mixed(), _shas("B"))])
        assert matched_effective_comparison(run, base).seed_id.tolist() == ["s1"]

    def test_by_category_reports_a_paired_ci(self):
        run = pd.DataFrame([
            _batch(f"s{i}", 0, _pure("male"), _shas(f"I{i}"), category="balanced")
            for i in range(6)
        ])
        base = pd.DataFrame([
            _batch(f"s{i}", 0, _mixed(), _shas(f"B{i}"), category="balanced")
            for i in range(6)
        ])
        by_cat = matched_comparison_by_category(matched_effective_comparison(run, base))
        overall = by_cat[by_cat.category == "<all>"].iloc[0]
        assert overall.n_seeds == 6
        assert overall.delta_abs_mean == pytest.approx(1.0)
        assert bool(overall.significant)


class TestMatchedBudgetAsr:
    def test_compares_the_first_independent_draw_of_each_side(self):
        run = pd.DataFrame([
            _batch("s1", 0, _mixed(), _shas("I0")),          # no quorum
            _batch("s1", 1, _pure("male"), _shas("I1")),     # would win, but out of budget
            _batch("s2", 0, _pure("male"), _shas("J0")),     # quorum
        ])
        base = pd.DataFrame([
            _batch("s1", 0, _pure("female"), _shas("B0")),
            _batch("s1", 1, _pure("female"), _shas("B0")),   # duplicate, ignored
            _batch("s2", 0, _mixed(), _shas("C0")),
        ])
        row = matched_budget_asr(run, base, success_n_of_m=6).iloc[0]
        assert row.n_seeds == 2
        assert row.attacker_hits == 1
        assert row.baseline_hits == 1
        assert row.discordant_attacker_only == 1
        assert row.discordant_baseline_only == 1

    def test_empty_when_nothing_pairs(self):
        run = pd.DataFrame([_batch("s1", 0, _mixed(), _shas("I"))])
        base = pd.DataFrame([_batch("s9", 0, _mixed(), _shas("B"))])
        assert matched_budget_asr(run, base, success_n_of_m=6).empty
