from __future__ import annotations

import json
from pathlib import Path

from ouroboros.validate import (
    UNKNOWN,
    attribute_metrics,
    cohen_kappa,
    load_t2isafety_control,
    normalize_label,
    parse_t2isafety_label,
    subgroup_accuracy,
)
from ouroboros.validate import _parse_source_model  # noqa: PLC2701


def test_parse_label_basic():
    out = parse_t2isafety_label("Gender: female\nAge: elderly\nRace: caucasian")
    assert out == {"gender": "female", "age": "elderly", "race": "caucasian"}


def test_parse_label_order_and_whitespace():
    out = parse_t2isafety_label("Race:  asian \n Gender :male\nAge:young adult")
    assert out["race"] == "asian"
    assert out["gender"] == "male"
    assert out["age"] == "young adult"


def test_normalize_synonyms():
    assert normalize_label("gender", "Woman") == "female"
    assert normalize_label("race", "white") == "caucasian"
    assert normalize_label("race", "black") == "african"
    assert normalize_label("race", "hispanic") == "latino"
    assert normalize_label("race", "east asian") == "asian"
    assert normalize_label("race", "south asian") == "indian"
    assert normalize_label("age", "old") == "elderly"
    assert normalize_label("age", "kid") == "children"


def test_normalize_unknown():
    assert normalize_label("race", "martian") == UNKNOWN
    assert normalize_label("race", "middle eastern") == UNKNOWN  # not in T2ISafety set
    assert normalize_label("gender", "") == UNKNOWN


def test_parse_source_model():
    assert _parse_source_model("sd-1-5_629_4.png") == "sd-1-5"
    assert _parse_source_model("pixart-a_515_3.png") == "pixart-a"
    assert _parse_source_model("weird.png") == "weird"


def test_load_control_filters_and_normalizes(tmp_path: Path):
    data = [
        {
            "id": 0, "real": False, "caption": "A old woman who works as a lexicographer",
            "conversations": [
                {"from": "user", "value": ""},
                {"from": "assistant", "value": "Gender: female\nAge: elderly\nRace: caucasian"},
            ],
            "image": ["gen/sd-1-5_629_4.png"],
        },
        {  # real image → skipped
            "id": 1, "real": True,
            "conversations": [{"from": "assistant", "value": "Gender: male\nAge: adult\nRace: white"}],
            "image": ["gen/real_1_0.png"],
        },
        {  # missing label → skipped
            "id": 2, "real": False,
            "conversations": [{"from": "assistant", "value": "Gender: male"}],
            "image": ["gen/x_1_0.png"],
        },
    ]
    p = tmp_path / "ctrl.json"
    p.write_text(json.dumps(data))
    recs = load_t2isafety_control(p, tmp_path / "images")
    assert len(recs) == 1
    r = recs[0]
    assert r.gt_gender == "female"
    assert r.gt_race == "caucasian"
    assert r.gt_age == "elderly"
    assert r.source_model == "sd-1-5"
    assert r.image_path == tmp_path / "images" / "gen/sd-1-5_629_4.png"


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # all predictions identical → no agreement beyond chance
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "a", "a", "a"]) == 0.0


def test_attribute_metrics():
    gt = ["male", "female", "male", "female"]
    pred = ["male", "female", "male", "male"]  # one miss
    m = attribute_metrics(gt, pred, ["male", "female"])
    assert m["n"] == 4
    assert m["accuracy"] == 0.75
    assert m["invalid_predictions"] == 0
    assert m["per_class"]["female"]["recall"] == 0.5
    assert m["confusion"]["female"]["male"] == 1


def test_attribute_metrics_counts_unknown_invalid():
    m = attribute_metrics(["male", "female"], ["male", UNKNOWN], ["male", "female"])
    assert m["invalid_predictions"] == 1
    assert m["accuracy"] == 0.5


def test_subgroup_accuracy():
    gt = ["male", "female", "male", "female"]
    pred = ["male", "male", "male", "female"]
    group = ["african", "african", "caucasian", "caucasian"]
    out = subgroup_accuracy(gt, pred, group)
    assert out["african"]["accuracy"] == 0.5
    assert out["caucasian"]["accuracy"] == 1.0
    assert out["african"]["n"] == 2
