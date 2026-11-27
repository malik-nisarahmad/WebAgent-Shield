import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import DatasetBundle, fingerprint


def validate_bundle(bundle: DatasetBundle) -> dict:
    train, test = bundle.train, bundle.test
    errors = []
    for name, frame in (("train", train), ("test", test)):
        required = {"record_id", "group_id", "text", "label", "source"}
        missing = required - set(frame.columns)
        if missing:
            errors.append(f"{name} missing columns {sorted(missing)}")
            continue
        if frame.empty:
            errors.append(f"{name} is empty")
        if frame["record_id"].duplicated().any():
            errors.append(f"{name} contains duplicate record IDs")
        if frame["text"].isna().any() or (frame["text"].str.strip() == "").any():
            errors.append(f"{name} contains empty text")
        if not set(frame["label"].unique()).issubset({0, 1}):
            errors.append(f"{name} contains invalid labels")
        if set(frame["label"].unique()) != {0, 1}:
            errors.append(f"{name} must contain both classes")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    overlap = set(train["group_id"]) & set(test["group_id"])
    if overlap:
        errors.append(f"group leakage detected for {len(overlap)} groups")
    train_hashes = train["text"].map(fingerprint)
    test_hashes = test["text"].map(fingerprint)
    duplicate_hashes = set(train_hashes) & set(test_hashes)
    if duplicate_hashes:
        errors.append(f"exact content leakage detected for {len(duplicate_hashes)} documents")
    conflicting = train.assign(_hash=train_hashes).groupby("_hash")["label"].nunique()
    if (conflicting > 1).any():
        errors.append("training data contains identical text with conflicting labels")
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    pilot_size = min(200, len(train))
    per_class = max(2, pilot_size // 2)
    pilot = pd.concat(
        [train.loc[train["label"] == label].head(per_class) for label in (0, 1)],
        ignore_index=True,
    ).head(pilot_size)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2_000)
    matrix = vectorizer.fit_transform(pilot["text"])
    if matrix.shape[0] != len(pilot) or matrix.shape[1] == 0:
        raise ValueError(f"Invalid TF-IDF pilot shape {matrix.shape}")
    if not np.isfinite(matrix.data).all():
        raise ValueError("TF-IDF pilot contains non-finite values")
    labels = pilot["label"].to_numpy()
    if labels.shape != (matrix.shape[0],):
        raise ValueError(f"Feature/label shape mismatch: X={matrix.shape}, y={labels.shape}")
    smoke = LogisticRegression(max_iter=200, random_state=42).fit(matrix, labels)
    predictions = smoke.predict(matrix[:2])
    if predictions.shape != (min(2, matrix.shape[0]),):
        raise ValueError("Classifier smoke-test prediction shape is invalid")
    return {"train_rows": len(train), "test_rows": len(test),
            "train_positive": int(train["label"].sum()), "test_positive": int(test["label"].sum()),
            "pilot_shape": list(matrix.shape), "group_overlap": 0, "content_overlap": 0}
