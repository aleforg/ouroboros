"""Per-call noise seeding: the fix for the identical-comparator-batch bug.

The comparator in `baseline.py` calls `generate_m()` repeatedly with the *same*
neutral prompt. While the seeds depended only on the position inside the batch,
every repeat returned bit-identical images, so `matched` mode spent a quarter of
its budget on copies and handed the max-over-batches selection an advantage the
control could not have.

Two properties matter and pull in opposite directions:
  * consecutive calls must draw different noise (the fix);
  * the FIRST call of a freshly built target must keep the seeds it had before,
    or `ouroboros run --replay` stops reproducing the runs already on disk.

None of this touches a model: `_next_seeds` is pure, so all three backends are
testable with no GPU and no heavy import.
"""
from __future__ import annotations

import pytest

from ouroboros.targets.base import CALL_SEED_STRIDE, build_target, call_seeds

M = 8
BACKENDS = ["flux", "diffusers", "qwen-image"]
# per-sample spacing each backend used before the fix
SAMPLE_STEP = {"flux": 1, "diffusers": 1000, "qwen-image": 1000}


class TestCallSeeds:
    def test_first_call_is_the_legacy_block(self):
        assert call_seeds(42, 0, 4, sample_step=1000) == [42, 1042, 2042, 3042]
        assert call_seeds(42, 0, 4, sample_step=1) == [42, 43, 44, 45]

    def test_each_call_gets_a_disjoint_block(self):
        blocks = [set(call_seeds(42, k, M)) for k in range(5)]
        union = set().union(*blocks)
        assert len(union) == 5 * M, "i blocchi di semi si sovrappongono"

    def test_stride_zero_is_the_legacy_behaviour(self):
        assert call_seeds(42, 0, M, stride=0) == call_seeds(42, 7, M, stride=0)

    def test_stride_clears_the_widest_possible_block(self):
        """A block must never reach into the next one, for any plausible M."""
        assert CALL_SEED_STRIDE > 1000 * 64


@pytest.mark.parametrize("backend", BACKENDS)
class TestBackends:
    """Constructing a target imports nothing heavy, so this runs anywhere."""

    def test_repeated_calls_draw_different_noise(self, backend):
        t = build_target(backend)
        first = t._next_seeds(M)
        second = t._next_seeds(M)
        assert first != second
        assert not set(first) & set(second)

    def test_first_call_reproduces_the_pre_fix_seeds(self, backend):
        t = build_target(backend)
        expected = [42 + i * SAMPLE_STEP[backend] for i in range(M)]
        assert t._next_seeds(M) == expected

    def test_legacy_stride_repeats_forever(self, backend):
        t = build_target(backend, target_call_seed_stride=0)
        assert t._next_seeds(M) == t._next_seeds(M) == t._next_seeds(M)

    def test_seed_base_is_honoured(self, backend):
        t = build_target(backend, target_seed_base=7)
        assert t._next_seeds(1)[0] == 7

    def test_two_targets_start_from_the_same_block(self, backend):
        """A fresh target is a fresh experiment — the counter is per instance."""
        a, b = build_target(backend), build_target(backend)
        a._next_seeds(M)
        a._next_seeds(M)
        assert b._next_seeds(M) == [42 + i * SAMPLE_STEP[backend] for i in range(M)]


def test_the_comparator_pattern_no_longer_repeats_itself():
    """The exact shape of the bug: one target, one prompt, several batches."""
    t = build_target("diffusers")
    drawn = [tuple(t._next_seeds(M)) for _ in range(5)]
    assert len(set(drawn)) == 5, "batch di controllo ripetuti pescano lo stesso rumore"
