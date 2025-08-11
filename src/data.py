from datasets import load_dataset, DatasetDict
import os
from typing import Optional

def load_svg_dataset(path: Optional[str] = None, text_field: str = "text") -> DatasetDict:
    """Load SVG dataset for SFT.

    - If `path` is a file, load as JSON/JSONL/CSV via datasets.load_dataset.
    - Expects a train split and a column named `text_field` (default: 'text').
    """
    if path is None:
        path = os.getenv("SVG_DATA_PATH", "data/svg_train.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Set SVG_DATA_PATH or pass `path`."
        )

    ext = os.path.splitext(path)[1].lower()
    if ext in [".jsonl", ".json"]:
        ds = load_dataset("json", data_files={"train": path})
    elif ext in [".csv"]:
        ds = load_dataset("csv", data_files={"train": path})
    else:
        # Try to treat as a datasets repo id (e.g., 'user/dsname')
        try:
            ds = load_dataset(path)
        except Exception as e:
            raise ValueError(
                f"Unsupported dataset path or extension: {path}. Error: {e}"
            )

    if "train" not in ds:
        raise ValueError("Expected a 'train' split in the dataset.")

    # Validate text field presence
    if text_field not in ds["train"].features:
        raise ValueError(
            f"Dataset must contain a text field named '{text_field}'. "
            f"Available fields: {list(ds['train'].features)}"
        )
    return ds
