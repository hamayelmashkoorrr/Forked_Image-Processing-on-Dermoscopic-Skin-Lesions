# """Ground-truth masks and benign/malignant labels for IMA++."""
# import os
# import re
# from collections import defaultdict
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd

# from .config import MASK_EXTENSIONS

# # One annotator's mask, e.g. ISIC_0000000_A04_T3_S1_5463934bbae47821f88025ad
# _SINGLE = re.compile(r"^(isic_\d+)_A\d+_T\d+_S\d+_\w+$")
# # Consensus mask, e.g. isic_0000004_MV_MV_MV_MV or isic_0000004_ST_ST_ST_ST
# _CONSENSUS = re.compile(r"^(isic_\d+)_(MV|ST)_(?:MV|ST)_(?:MV|ST)_(?:MV|ST)$")


# def index_masks(gt_dir: Path) -> dict:
#     """Scan the mask folder once.
#     Returns {isic_id: {"single": [paths], "MV": path or None, "ST": path or None}}."""
#     index = defaultdict(lambda: {"single": [], "MV": None, "ST": None})
#     with os.scandir(gt_dir) as it:
#         for entry in it:
#             p = Path(entry.path)
#             if p.suffix.lower() not in MASK_EXTENSIONS:
#                 continue
#             m = _CONSENSUS.match(p.stem)
#             if m:
#                 index[m.group(1)][m.group(2)] = p
#                 continue
#             m = _SINGLE.match(p.stem)
#             if m:
#                 index[m.group(1)]["single"].append(p)
#     return dict(index)


# def pick_ground_truth(entry: dict, consensus_tag: str):
#     """Choose one mask per image. Returns (path, source, n_annotators).
#     - consensus mask if the image has one (it has several annotators)
#     - otherwise the single annotator's mask"""
#     n = len(entry["single"])
#     if entry.get(consensus_tag) is not None:
#         return entry[consensus_tag], f"consensus_{consensus_tag}", n
#     if n == 1:
#         return entry["single"][0], "single", n
#     if n > 1:  # several annotators but no consensus mask: unexpected, flagged in the log
#         return sorted(entry["single"])[0], "single_first_of_many", n
#     return None, "missing", 0


# def read_mask(path: Path):
#     """Read a mask as a boolean array (True = lesion)."""
#     img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         return None
#     return img > (0 if img.max() <= 1 else 127)


# def load_labels(path: Path, column: str) -> dict:
#     """Return {isic_id: 'benign' | 'malignant'}. Other values (e.g. indeterminate) are dropped."""
#     assert path.exists(), f"Label file not found: {path}"
#     if path.suffix.lower() in {".xlsx", ".xls"}:
#         df = pd.read_excel(path)
#     else:
#         df = pd.read_csv(path)
#     assert "isic_id" in df.columns, f"No 'isic_id' column. Columns: {list(df.columns)}"
#     assert column in df.columns, \
#         f"No '{column}' column. Set LABEL_COLUMN in config.py. Columns: {list(df.columns)}"
#     lab = df[["isic_id", column]].drop_duplicates("isic_id").copy()
#     lab[column] = lab[column].astype(str).str.strip().str.lower()
#     print(f"Label file: {len(df)} rows | values in '{column}':")
#     print(lab[column].value_counts().to_string())
#     keep = lab[lab[column].isin(["benign", "malignant"])]
#     return dict(zip(keep["isic_id"], keep[column]))



"""Ground-truth masks and benign/malignant labels for IMA++."""

import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import MASK_EXTENSIONS


# One annotator's mask, for example:
# ISIC_0000000_A04_T3_S1_5463934bbae47821f88025ad.png
# re.IGNORECASE makes matching robust to ISIC/isic filename capitalization.
_SINGLE = re.compile(
    r"^(ISIC_\d+)_A\d+_T\d+_S\d+_\w+$",
    flags=re.IGNORECASE,
)

# Consensus masks, for example:
# ISIC_0000004_MV_MV_MV_MV.png
# ISIC_0000004_ST_ST_ST_ST.png
_CONSENSUS = re.compile(
    r"^(ISIC_\d+)_(MV|ST)_(?:MV|ST)_(?:MV|ST)_(?:MV|ST)$",
    flags=re.IGNORECASE,
)


def index_masks(gt_dir: Path) -> dict:
    """Index mask files directly inside ``gt_dir``.

    Returns a dictionary in this format::

        {
            "ISIC_0000004": {
                "single": [Path(...), ...],
                "MV": Path(...) or None,
                "ST": Path(...) or None,
            }
        }

    This function scans only the immediate contents of ``gt_dir``; it does
    not search nested subdirectories.
    """
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth folder does not exist: {gt_dir}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"Ground-truth path is not a directory: {gt_dir}")

    index = defaultdict(lambda: {"single": [], "MV": None, "ST": None})

    with os.scandir(gt_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue

            mask_path = Path(entry.path)
            if mask_path.suffix.lower() not in MASK_EXTENSIONS:
                continue

            # Consensus masks first, then individual annotator masks.
            match = _CONSENSUS.fullmatch(mask_path.stem)
            if match:
                image_id = match.group(1).upper()
                consensus_type = match.group(2).upper()
                index[image_id][consensus_type] = mask_path
                continue

            match = _SINGLE.fullmatch(mask_path.stem)
            if match:
                image_id = match.group(1).upper()
                index[image_id]["single"].append(mask_path)

    # Stable ordering makes fallback selection reproducible.
    for item in index.values():
        item["single"].sort(key=lambda path: path.name.lower())

    return dict(index)


def pick_ground_truth(entry: dict, consensus_tag: str):
    """Choose one reference mask for an image.

    Selection order:
      1. Requested consensus mask (MV or ST), if available.
      2. The sole individual annotator mask, if exactly one exists.
      3. First alphabetically sorted individual mask if multiple exist but
         the requested consensus mask is unavailable.
      4. Missing, if no usable mask exists.

    Returns ``(path, source, n_annotators)``.
    """
    consensus_tag = str(consensus_tag).upper()
    if consensus_tag not in {"MV", "ST"}:
        raise ValueError(
            f"consensus_tag must be 'MV' or 'ST', got {consensus_tag!r}"
        )

    individual_masks = entry.get("single", [])
    n_annotators = len(individual_masks)

    consensus_path = entry.get(consensus_tag)
    if consensus_path is not None:
        return consensus_path, f"consensus_{consensus_tag}", n_annotators

    if n_annotators == 1:
        return individual_masks[0], "single", n_annotators

    if n_annotators > 1:
        ordered_masks = sorted(individual_masks, key=lambda path: path.name.lower())
        return ordered_masks[0], "single_first_of_many", n_annotators

    return None, "missing", 0


def read_mask(path: Path):
    """Read a ground-truth mask as a boolean array (True = lesion)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    threshold = 0 if image.max() <= 1 else 127
    return image > threshold


def load_labels(path: Path, column: str) -> dict:
    """Return ``{ISIC_id: 'benign' | 'malignant'}``.

    Other diagnosis values, including indeterminate and missing values, are
    excluded from the benign-versus-malignant analysis.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        dataframe = pd.read_csv(path)

    # Be tolerant of accidental whitespace in CSV column headers.
    dataframe.columns = dataframe.columns.astype(str).str.strip()

    if "isic_id" not in dataframe.columns:
        raise KeyError(
            f"No 'isic_id' column in {path}. Columns: {list(dataframe.columns)}"
        )
    if column not in dataframe.columns:
        raise KeyError(
            f"No {column!r} column in {path}. "
            f"Set LABEL_COLUMN in config.py. Columns: {list(dataframe.columns)}"
        )

    labels = dataframe[["isic_id", column]].drop_duplicates("isic_id").copy()
    labels["isic_id"] = labels["isic_id"].astype(str).str.strip().str.upper()
    labels[column] = labels[column].astype(str).str.strip().str.lower()

    print(f"Label file: {len(dataframe)} rows | values in {column!r}:")
    print(labels[column].value_counts().to_string())

    eligible_labels = labels[labels[column].isin(["benign", "malignant"])]
    return dict(zip(eligible_labels["isic_id"], eligible_labels[column]))
