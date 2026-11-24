# patient-discharge-forecast

CS 178 (Machine Learning & Data Mining) final project. We predict whether a diabetic patient will be readmitted to the hospital after discharge, using the [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset, and compare several classifiers from the course.

**Task:** 3-class classification on `readmitted`:

| Class | Meaning | Share (after filtering) |
|---|---|---|
| `NO`  | not readmitted | ~53% |
| `>30` | readmitted after 30+ days | ~36% |
| `<30` | readmitted within 30 days | ~11% |

**Primary metric: macro-F1.** The classes are imbalanced: always guessing `NO` already gets ~53% accuracy. We also report accuracy, per-class recall (especially `<30`) and confusion matrices.

---

## Quick start

```bash
git clone <repo-url> && cd patient-discharge-forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from src.data import *; df = load_raw(); print(load_split(df, 'train')[0].shape)"
```

If that prints a shape around `(59648, 44)`, you're set up. The raw CSV and the frozen splits are already in the repo, so there's nothing to download.

---

## Repo layout

```
data/raw/            diabetic_data.csv, IDS_mapping.csv   (committed, don't edit)
data/splits/         train/val/test_idx.npy               (FROZEN, don't regenerate)
src/data.py          loading, cleaning, splits, preprocessor    [A] ✅
src/evaluate.py      metrics, confusion matrices, results JSON  [B]
src/run.py           tuning loop, final test run, learning curves [C]
src/errors.py        error overlap + misclassified examples     [D]
src/models/          one file per model family (see template below)
notebooks/eda.ipynb  data exploration for report §2              [A]
results/             <model>.json, one per model (committed)
figures/             report figures (committed)
```

---

## Using the data

```python
from sklearn.pipeline import Pipeline
from src.data import load_raw, load_split, make_preprocessor

df = load_raw()                       # cleaned DataFrame, ~99k rows
X_tr, y_tr = load_split(df, "train")  # 60%
X_va, y_va = load_split(df, "val")    # 20%; use this for hyperparameter tuning

pipe = Pipeline([("prep", make_preprocessor(df)), ("clf", your_model)])
pipe.fit(X_tr, y_tr)
pipe.score(X_va, y_va)
```

**Always put `make_preprocessor` inside the Pipeline.** That way the scaler and encoder are fit on the training data only, so validation and test statistics never leak into training.

### What `load_raw()` already handles

- `?` → missing; the literal `"None"` in `A1Cresult` / `max_glu_serum` is **kept** as a category (it means "not measured")
- drops encounters where the patient died or went to hospice (`discharge_disposition_id ∈ {11,13,14,19,20,21}`), since they can't be readmitted (Strack et al., 2014)
- drops 3 `Unknown/Invalid` gender rows
- `age` brackets → integers (`[60-70)` → 60)
- `diag_1..3` ICD-9 codes → ~9 groups (Circulatory, Diabetes, Respiratory, …)
- `medical_specialty` → top 10 + `Other`; missing categoricals → `"Missing"`
- `admission_type_id`, `discharge_disposition_id`, `admission_source_id` → treated as categories

`load_split` drops `encounter_id`, `patient_nbr`, `weight` (97% missing), the constant columns `examide`/`citoglipton`, and the target.

### Split details

60/20/20, **grouped by `patient_nbr`** with `GroupShuffleSplit`, seed `178`. Patients have multiple encounters, so splitting by row would leak the same patient into both train and test. The code checks that no patient appears in two splits.

---

## Rules (please read)

1. **Don't touch the test set.** Never call `load_split(df, "test")` while tuning. It gets used **once**, at the end, by `run.py --final`, for all models together.
2. **Don't re-run `python -m src.data`.** It regenerates the splits. They're frozen so everyone's results are comparable.
3. **Seed `178` everywhere** (`random_state=178`).
4. **Tune on validation macro-F1**, not accuracy.
5. **Log what you tried.** Every hyperparameter grid you search goes in your model file, because the report needs the exact ranges.

---

## Adding your model

Create `src/models/<name>.py` following this template:

```python
# src/models/knn.py
from sklearn.neighbors import KNeighborsClassifier

NAME = "knn"
PARAM_GRID = {
    "n_neighbors": [1, 5, 15, 51, 101],
    "weights": ["uniform", "distance"],
}

def make_model(**params):
    return KNeighborsClassifier(**params)
```

### Tuning protocol (same for every model, and goes in report §3)

```
for p in ParameterGrid(PARAM_GRID):
    fit Pipeline(prep, make_model(**p)) on TRAIN
    score macro-F1 on VAL
p* = argmax val macro-F1
refit on TRAIN+VAL with p*         # load_split(df, "trainval")
evaluate ONCE on TEST  →  results/<NAME>.json
```

Until `run.py` exists, write a small loop yourself that follows this protocol.

Tips:
- `class_weight="balanced"` is worth adding to the grid for any model that supports it (LR, trees, RF).
- kNN is slow on 60k rows, so tune on a 10k subsample of train (`random_state=178`) and mention that in the report.

---

## Team & ownership

| | Shared code | Model(s) | Report section |
|---|---|---|---|
| **A** | `src/data.py` ✅, EDA notebook | Logistic regression | §2 Data Description |
| **B** | `src/evaluate.py` | k-nearest neighbors | §3 Experimental Setup |
| **C** | `src/run.py`, learning curves | Decision tree + random forest | §4 Experimental Results |
| **D** | `src/errors.py` | MLP (neural net) | §5 Insights |

Everyone: write 1–2 sentences on your own classifier for §Classifiers, plus your own Contributions line. Write the Summary together at the end.

**Order:** splits ✅ → B & C finish `evaluate.py` / `run.py` early (everyone uses them) → everyone tunes on val → one final test run → D does error analysis, C makes figures → write the report.

---

## Git workflow

```bash
git checkout -b <your-name>/<feature>   # e.g. b/knn
# ...work...
git add <files> && git commit -m "kNN grid search on val"
git push -u origin <your-name>/<feature>
```
Open a PR into `main` and have one teammate glance at it before merging. Clear notebook outputs before committing:
```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## Reference

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014. This is the dataset's introductory paper; our filtering and ICD-9 grouping follow it. Cite it in §2.

Dataset license: CC BY 4.0.
