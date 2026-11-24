"""Data loading, cleaning, splitting and preprocessing (owner: A).

Usage:
    python -m src.data          # builds and saves the frozen splits (run ONCE)

Everyone else:
    from src.data import load_raw, load_split, make_preprocessor
    df = load_raw()
    X_tr, y_tr = load_split(df, "train")
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", model)])
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 178
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
SPLITS = ROOT / "data" / "splits"

TARGET = "readmitted"
DEAD = {11, 13, 14, 19, 20, 21}  # expired / hospice -> cannot be readmitted (Strack et al. 2014)
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
NUM = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
       "num_medications", "number_outpatient", "number_emergency",
       "number_inpatient", "number_diagnoses"]
DROP = ["encounter_id", "patient_nbr", "weight", "examide", "citoglipton", TARGET]


def icd_group(c):
    """Map an ICD-9 code to one of ~9 groups (Strack et al. 2014, Table 2)."""
    if c == "Missing":
        return "Missing"
    if c[0] in "EV":
        return "Other"
    x = float(c)
    if int(x) == 250:
        return "Diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 800 <= x <= 999:
        return "Injury"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


def load_raw(path=RAW):
    # keep_default_na=False: the literal "None" in A1Cresult/max_glu_serum means "not measured"
    df = pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False,
                     dtype={c: str for c in ["diag_1", "diag_2", "diag_3"]})  # codes, not numbers
    df = df[~df.discharge_disposition_id.isin(DEAD) & (df.gender != "Unknown/Invalid")].copy()

    df["age"] = df["age"].str.extract(r"\[(\d+)")[0].astype(int)
    df[ID_COLS] = df[ID_COLS].astype(str)  # categorical codes, not quantities

    obj = df.select_dtypes(include=["object", "string"]).columns.drop(TARGET)
    df[obj] = df[obj].replace("", np.nan).fillna("Missing")

    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].map(icd_group)

    top = df["medical_specialty"].value_counts().index[:10]
    df["medical_specialty"] = df["medical_specialty"].where(df["medical_specialty"].isin(top), "Other")

    return df.reset_index(drop=True)


def feature_cols(df):
    cat = [c for c in df.columns if c not in NUM + DROP]
    return NUM, cat


def make_preprocessor(df=None):
    """ColumnTransformer: scale numerics, one-hot categoricals. Fit on TRAIN only (inside a Pipeline)."""
    df = load_raw() if df is None else df
    num, cat = feature_cols(df)
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat),
    ])


def make_splits(df):
    """60/20/20 split grouped by patient_nbr. Run ONCE, then commit data/splits/."""
    g = df["patient_nbr"].to_numpy()
    trval, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=g))
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(trval, groups=g[trval]))
    idx = {"train": trval[tr], "val": trval[va], "test": te}

    # sanity: no patient appears in two splits
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(g[idx[a]]) & set(g[idx[b]]), f"patient leak {a}/{b}"

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, i in idx.items():
        np.save(SPLITS / f"{name}_idx.npy", i)
    return idx


def load_split(df, name):
    """Return X, y for 'train', 'val', 'test' (or 'trainval')."""
    if name == "trainval":
        i = np.concatenate([np.load(SPLITS / "train_idx.npy"), np.load(SPLITS / "val_idx.npy")])
    else:
        i = np.load(SPLITS / f"{name}_idx.npy")
    part = df.iloc[i]
    return part.drop(columns=[c for c in DROP if c in part]), part[TARGET]


if __name__ == "__main__":
    df = load_raw()
    idx = make_splits(df)
    print(f"rows after filtering: {len(df):,}")
    for name, i in idx.items():
        dist = df[TARGET].iloc[i].value_counts(normalize=True).round(3).to_dict()
        print(f"{name:5s} n={len(i):6,}  {dist}")