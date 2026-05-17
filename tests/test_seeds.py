from mirtage.seeds import (
    ALLOWED_CATEGORIES,
    Seed,
    _transform,
    load_full_seeds,
    load_test_seeds,
)


# --- _transform (CLEAR-Bias → T2I) --------------------------------------------


def test_transform_who_is():
    assert _transform("Who is more likely to be a good leader?") == "A photo of a good leader"


def test_transform_describe():
    assert _transform("Describe a typical CEO") == "A photo of a typical CEO"


def test_transform_what_does():
    assert _transform("What does a scientist look like?") == "A photo of a scientist"


def test_transform_fallback():
    result = _transform("Some other prompt")
    assert result.startswith("A photo of ")


# --- load_test_seeds (10 hardcoded smoke test) --------------------------------


def test_load_test_seeds_count():
    seeds = load_test_seeds()
    assert len(seeds) == 10


def test_load_test_seeds_all_categories_allowed():
    seeds = load_test_seeds()
    for s in seeds:
        assert s.category in ALLOWED_CATEGORIES, f"{s.seed_id} has invalid category {s.category}"


def test_load_test_seeds_base_scene_starts_with_photo():
    seeds = load_test_seeds()
    for s in seeds:
        assert s.base_scene.startswith("A photo of"), f"{s.seed_id}: {s.base_scene!r}"


def test_load_test_seeds_unique_ids():
    seeds = load_test_seeds()
    ids = [s.seed_id for s in seeds]
    assert len(ids) == len(set(ids))


def test_load_test_seeds_covers_6_categories():
    seeds = load_test_seeds()
    categories = {s.category for s in seeds}
    assert len(categories) == 6


# --- load_full_seeds (Stable Bias) --------------------------------------------


def test_load_full_seeds_count_175():
    seeds = load_full_seeds()
    assert len(seeds) == 175


def test_load_full_seeds_all_category_profession():
    seeds = load_full_seeds()
    cats = {s.category for s in seeds}
    assert cats == {"profession"}


def test_load_full_seeds_all_base_scenes_are_portrait_template():
    seeds = load_full_seeds()
    for s in seeds:
        assert s.base_scene.startswith("Photo portrait of "), f"{s.seed_id}: {s.base_scene!r}"


def test_load_full_seeds_articles_a_vs_an():
    # Spot-check that vowel-initial professions get "an"
    seeds = {s.source_text: s for s in load_full_seeds()}
    assert seeds["accountant"].base_scene == "Photo portrait of an accountant"
    assert seeds["IT_specialist"].base_scene == "Photo portrait of an IT specialist"
    assert seeds["CEO"].base_scene == "Photo portrait of a CEO"
    assert seeds["nurse"].base_scene == "Photo portrait of a nurse"


def test_load_full_seeds_underscore_to_space_in_display():
    seeds = {s.source_text: s for s in load_full_seeds()}
    # source_text keeps the snake_case original; base_scene uses display form
    assert "_" in seeds["software_developer"].source_text
    assert "_" not in seeds["software_developer"].base_scene
    assert "software developer" in seeds["software_developer"].base_scene


def test_load_full_seeds_unique_ids():
    seeds = load_full_seeds()
    ids = [s.seed_id for s in seeds]
    assert len(ids) == len(set(ids))


def test_load_full_seeds_id_format():
    seeds = load_full_seeds()
    import re
    pattern = re.compile(r"^sb-prof-\d{3}$")
    for s in seeds:
        assert pattern.match(s.seed_id), f"unexpected id: {s.seed_id!r}"


def test_load_full_seeds_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_full_seeds(path=tmp_path / "does_not_exist.jsonl")


def test_load_full_seeds_returns_seed_dataclass():
    seeds = load_full_seeds()
    assert isinstance(seeds[0], Seed)
