import json
import re
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.constants import TEXT_COLUMNS


def clean_text(value: object) -> str:
    """Make messy job-post text easier for TF-IDF to learn from."""
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text_columns(data: pd.DataFrame) -> pd.Series:
    frame = data.copy()

    for column in TEXT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].apply(clean_text)

    return frame[TEXT_COLUMNS].agg(" ".join, axis=1).str.strip()


def save_object(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path) -> object:
    return joblib.load(path)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
