import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import combine_datasets, load_browsesafe, load_local_corpus, split_local_groups
from .validation import validate_bundle


def features():
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.995,
        sublinear_tf=True, max_features=60_000, dtype=np.float32,
    )


def classifiers():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1_000, class_weight="balanced", random_state=42, solver="liblinear"
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=42),
    }


def evaluate(labels, predictions):
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": report}


def main():
    parser = argparse.ArgumentParser(description="Validate data and train classical WebAgent Shield baselines")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/browsesafe"))
    parser.add_argument("--local-corpus", type=Path, default=Path("data/local_corpus.jsonl"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/baselines"))
    args = parser.parse_args()

    local = split_local_groups(load_local_corpus(args.local_corpus))
    public = load_browsesafe(args.cache_dir)
    bundle = combine_datasets(public, local)
    validation = validate_bundle(bundle)
    print(json.dumps({"validation": validation}, indent=2))
    if args.validate_only:
        return

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"validation": validation, "models": {}}
    vectorizer = features()
    train_features = vectorizer.fit_transform(bundle.train["text"])
    test_features = vectorizer.transform(bundle.test["text"])
    if train_features.shape[0] != len(bundle.train) or test_features.shape[0] != len(bundle.test):
        raise ValueError(
            f"Full feature/row mismatch: train={train_features.shape}, test={test_features.shape}"
        )
    if train_features.shape[1] == 0 or test_features.shape[1] != train_features.shape[1]:
        raise ValueError(
            f"Invalid full feature shapes: train={train_features.shape}, test={test_features.shape}"
        )
    if not np.isfinite(train_features.data).all() or not np.isfinite(test_features.data).all():
        raise ValueError("Full TF-IDF matrices contain non-finite values")
    metrics["feature_matrix"] = {
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_nonzero": int(train_features.nnz),
        "test_nonzero": int(test_features.nnz),
    }

    for name, classifier in classifiers().items():
        classifier.fit(train_features, bundle.train["label"])
        predictions = classifier.predict(test_features)
        if predictions.shape != (len(bundle.test),):
            raise ValueError(f"{name} returned invalid prediction shape {predictions.shape}")
        model_metrics = evaluate(bundle.test["label"], predictions)
        metrics["models"][name] = model_metrics
        model = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
        joblib.dump(model, args.models_dir / f"{name}.joblib")
        print(json.dumps({name: model_metrics}, indent=2))
    (args.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
