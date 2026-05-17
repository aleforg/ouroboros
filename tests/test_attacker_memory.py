from mirtage.attacker import Memory, MemoryEntry


def _entry(iter: int, score: int = 5) -> MemoryEntry:
    return MemoryEntry(
        iter=iter,
        target_prompt=f"prompt_{iter}",
        strategy_label="test",
        bias_score=score,
        per_axis_scores={},
        judge_rationale="rationale",
        outcome="fail",
    )


def test_snapshot_empty():
    m = Memory(top_k=3)
    assert m.snapshot() == []


def test_snapshot_single_entry():
    m = Memory(top_k=3)
    m.add(_entry(0, score=7))
    snap = m.snapshot()
    assert len(snap) == 1


def test_snapshot_top_k_by_score():
    m = Memory(top_k=2)
    for i, score in enumerate([3, 9, 5, 7, 1]):
        m.add(_entry(i, score=score))
    snap = m.snapshot()
    scores = [e.bias_score for e in snap]
    # top-2 by score: 9, 7 + most recent (iter=4, score=1)
    assert 9 in scores
    assert 7 in scores


def test_snapshot_deduplicates_recent_in_top_k():
    m = Memory(top_k=3)
    # Add 3 entries; most recent has highest score — appears once
    for i, score in enumerate([8, 7, 9]):
        m.add(_entry(i, score=score))
    snap = m.snapshot()
    iters = [e.iter for e in snap]
    assert len(iters) == len(set(iters)), "Duplicate iters in snapshot"


def test_snapshot_most_recent_included_even_if_low_score():
    m = Memory(top_k=1)
    m.add(_entry(0, score=9))
    m.add(_entry(1, score=9))
    m.add(_entry(2, score=1))  # most recent, low score
    snap = m.snapshot()
    iters = [e.iter for e in snap]
    assert 2 in iters  # most recent must be present


def test_add_idempotent_same_iter():
    m = Memory(top_k=3)
    m.add(_entry(0, score=5))
    m.add(_entry(0, score=9))  # same iter, updated score
    snap = m.snapshot()
    assert len(snap) == 1
    assert snap[0].bias_score == 9


def test_snapshot_size_capped():
    m = Memory(top_k=2)
    for i in range(10):
        m.add(_entry(i, score=i))
    snap = m.snapshot()
    # top-2 (iters 8 and 9) + most_recent (iter 9) → deduped → 2 or 3 entries
    assert len(snap) <= 3
