import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import GroupShuffleSplit


REQUIRED_LOCAL_FIELDS = {"record_id", "group_id", "content", "label", "passages"}
LABELS = {"no": 0, "yes": 1}
BROWSESAFE_FILES = {
    "train": ("https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/train.parquet",
              "430881fa53da3898956048676f58db894fc760cfa348ab264ee8a07c44a3d9fb", 142_485_917),
    "test": ("https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/test.parquet",
             "00cbad96b60fee46e016d79af6981fb221384c61f12cf28b4f04b5a6420573d0", 46_938_712),
}


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    normalized = str(value).strip().lower()
    if normalized not in LABELS:
        raise ValueError(f"Unsupported label: {value!r}")
    return LABELS[normalized]


def html_to_text(content: str, max_chars: int = 50_000) -> str:
    soup = BeautifulSoup(content, "lxml")
    embedded = []
    for node in soup(["script", "style"]):
        embedded.append(node.get_text(" ", strip=True))
        node.decompose()
    visible = " ".join(soup.stripped_strings)
    comments_and_attributes = []
    for comment in soup.find_all(string=lambda value: value.__class__.__name__ == "Comment"):
        comments_and_attributes.append(str(comment))
    for element in soup.find_all(True):
        for attribute in ("content", "title", "alt", "aria-label", "data-instruction", "value"):
            if element.get(attribute):
                comments_and_attributes.append(str(element[attribute]))
    hidden = " ".join(comments_and_attributes + embedded)
    return (hidden[:20_000] + " " + visible[:30_000])[:max_chars].strip()


def load_local_corpus(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_LOCAL_FIELDS - record.keys()
            if missing:
                raise ValueError(f"Local corpus line {line_number} is missing {sorted(missing)}")
            if not record["passages"]:
                raise ValueError(f"Local corpus line {line_number} has no verified passages")
            passage_labels = {normalize_label(passage["label"]) for passage in record["passages"]}
            page_label = normalize_label(record["label"])
            if page_label == 1 and 1 not in passage_labels:
                raise ValueError(f"Positive record {record['record_id']} lacks a positive passage")
            records.append({"record_id": record["record_id"], "group_id": record["group_id"],
                            "content": record["content"], "text": record["content"],
                            "label": page_label, "source": "local"})
    return pd.DataFrame.from_records(records)


def split_local_groups(frame: pd.DataFrame, test_size=0.25, seed=42) -> DatasetBundle:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(frame, frame["label"], groups=frame["group_id"]))
    return DatasetBundle(frame.iloc[train_index].reset_index(drop=True),
                         frame.iloc[test_index].reset_index(drop=True))


def _download_verified(url: str, destination: Path, expected_sha256: str,
                       expected_size: int, retries: int = 8) -> None:
    if destination.exists() and fingerprint_file(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if (partial.exists() and partial.stat().st_size == expected_size
            and fingerprint_file(partial) == expected_sha256):
        partial.replace(destination)
        return
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.replace(partial.with_suffix(partial.suffix + ".invalid"))

    timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=timeout) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    offset = 0
                with partial.open(mode) as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        output.write(chunk)
            if partial.stat().st_size == expected_size:
                break
            if partial.stat().st_size > expected_size:
                raise ValueError(f"Downloaded {destination.name} is larger than expected")
        except (httpx.HTTPError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Could not download {destination.name} after {retries} attempts; "
                    f"{partial.stat().st_size if partial.exists() else 0} bytes are saved for resume"
                ) from error
            time.sleep(min(2 ** attempt, 30))

    if partial.stat().st_size != expected_size:
        raise ValueError(
            f"Size mismatch for {destination.name}: expected {expected_size}, "
            f"received {partial.stat().st_size}"
        )
    if fingerprint_file(partial) != expected_sha256:
        raise ValueError(f"Checksum mismatch while downloading {destination.name}")
    partial.replace(destination)


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_browsesafe(cache_dir: Path) -> DatasetBundle:
    raw_dir = cache_dir / "raw"
    for split, (url, checksum, size) in BROWSESAFE_FILES.items():
        _download_verified(url, raw_dir / f"{split}.parquet", checksum, size)

    frames = {}
    for split in ("train", "test"):
        frame = pd.read_parquet(raw_dir / f"{split}.parquet", columns=["content", "label"])
        frame["label"] = frame["label"].map(normalize_label)
        frames[split] = frame

    raw_hashes = {
        split: frame["content"].map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        for split, frame in frames.items()
    }
    test_hashes = set(raw_hashes["test"])
    frames["train"] = frames["train"].loc[
        ~raw_hashes["train"].isin(test_hashes) & ~raw_hashes["train"].duplicated()
    ].copy()

    for split, frame in frames.items():
        frame.reset_index(drop=True, inplace=True)
        frame["text"] = frame["content"]
        frame["record_id"] = [f"browsesafe-{split}-{index}" for index in range(len(frame))]
        frame["group_id"] = frame["record_id"]
        frame["source"] = "browsesafe"
    return DatasetBundle(frames["train"], frames["test"])


def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def combine_datasets(public: DatasetBundle, local: DatasetBundle) -> DatasetBundle:
    return DatasetBundle(pd.concat([public.train, local.train], ignore_index=True),
                         pd.concat([public.test, local.test], ignore_index=True))
