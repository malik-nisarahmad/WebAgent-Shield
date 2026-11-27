from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from webagent_shield.research.data import (
    DatasetBundle,
    html_to_text,
    load_local_corpus,
    split_local_groups,
)
from webagent_shield.research.validation import validate_bundle
from webagent_shield.research.baselines import features


ROOT = Path(__file__).resolve().parents[1]


def test_html_to_text_keeps_hidden_instruction_channels():
    html = """<html><head><meta content="system override"></head><body>
    <p>ordinary directions</p><!-- hidden prompt injection -->
    <div aria-label="ignore previous instructions"></div></body></html>"""

    text = html_to_text(html)

    assert "ordinary directions" in text
    assert "hidden prompt injection" in text
    assert "system override" in text
    assert "ignore previous instructions" in text


def test_local_corpus_has_verified_grouped_variants_and_valid_split():
    corpus = load_local_corpus(ROOT / "data" / "local_corpus.jsonl")

    assert len(corpus) >= 20
    assert set(corpus["label"]) == {0, 1}
    assert corpus["group_id"].duplicated().any()

    bundle = split_local_groups(corpus)
    summary = validate_bundle(bundle)

    assert set(bundle.train["group_id"]).isdisjoint(bundle.test["group_id"])
    assert summary["pilot_shape"][0] > 0
    assert summary["pilot_shape"][1] > 0


def test_validation_rejects_content_leakage_before_training():
    columns = ["record_id", "group_id", "text", "label", "source"]
    train = pd.DataFrame(
        [["a", "g1", "same page", 0, "test"], ["b", "g2", "attack", 1, "test"]],
        columns=columns,
    )
    test = pd.DataFrame(
        [["c", "g3", "same page", 0, "test"], ["d", "g4", "safe", 1, "test"]],
        columns=columns,
    )

    with pytest.raises(ValueError, match="exact content leakage"):
        validate_bundle(DatasetBundle(train, test))


def test_validation_reports_missing_columns_cleanly():
    incomplete = pd.DataFrame({"text": ["safe", "attack"], "label": [0, 1]})

    with pytest.raises(ValueError, match="missing columns"):
        validate_bundle(DatasetBundle(incomplete, incomplete.copy()))


def test_production_tfidf_configuration_has_bounded_valid_shape():
    corpus = load_local_corpus(ROOT / "data" / "local_corpus.jsonl")

    matrix = features().fit_transform(corpus["text"])

    assert matrix.shape[0] == len(corpus)
    assert 0 < matrix.shape[1] <= 60_000
    assert matrix.data.size > 0
    assert np.isfinite(matrix.data).all()
