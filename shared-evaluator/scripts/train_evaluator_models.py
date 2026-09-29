"""Train the shared AMP likelihood and empirical hemolysis models.

The AMP classifier distinguishes curated AMP sequences from a UniProt-derived,
25%-identity negative set. The hemolysis classifier uses only explicit human
erythrocyte 50%-hemolysis concentration endpoints, converted to micromolar.
Neither output is a substitute for prospective biological validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import joblib
import Levenshtein
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = ROOT.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MAIN_ROOT / "data-engineering/src"))

from evaluator.activity_safety_models import extract_features

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MEASUREMENT = re.compile(
    r"^\s*(<=|>=|<|>)?\s*(\d+(?:\.\d*)?|\.\d+)"
    r"(?:\s*(?:±|\+/-)\s*(\d+(?:\.\d*)?|\.\d+))?\s*$"
)


def label_hemolysis(row: pd.Series, cutoff_uM: float = 100.0) -> int | None:
    """Label an explicit human-RBC HC50 endpoint against a micromolar cutoff.

    Returns 1 only when the measured interval establishes HC50 < cutoff_uM, 0
    only when it establishes HC50 >= cutoff_uM, and None for unsupported or
    ambiguous observations. The source's percent-lysis assay endpoints are not
    treated as concentration endpoints.
    """
    assay = re.sub(r"\s+", " ", str(row.get("assay_type", "")).strip().lower())
    cell_type = re.sub(
        r"\s+", " ", str(row.get("target_cell_type", "")).strip().lower()
    )
    if assay != "50% hemolysis" or cell_type != "human erythrocytes":
        return None

    match = MEASUREMENT.fullmatch(str(row.get("measurement", "")).strip())
    if match is None:
        return None
    unit = str(row.get("measurement_unit", "")).strip().lower()
    unit = unit.replace("µ", "u").replace("μ", "u").replace(" ", "")
    try:
        molecular_weight = float(row.get("molecular_weight_da"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(molecular_weight) or molecular_weight <= 0:
        return None

    if unit in {"um", "umol/l", "micromolar"}:
        factor = 1.0
    elif unit in {"ug/ml", "mg/l"}:
        factor = 1000.0 / molecular_weight
    elif unit == "mg/ml":
        factor = 1_000_000.0 / molecular_weight
    else:
        return None

    qualifier = match.group(1) or ""
    center = float(match.group(2)) * factor
    error = float(match.group(3)) * factor if match.group(3) is not None else 0.0
    if not math.isfinite(center) or not math.isfinite(error) or center < 0 or error < 0:
        return None

    # For a censored observation, assign a class only if the entire known bound
    # is on one side of the decision threshold.
    if qualifier in {"<", "<="}:
        # Strict HC50 < cutoff: an upper bound that could equal the cutoff is
        # positive only when the source qualifier itself is strict.
        if qualifier == "<=" and center == cutoff_uM:
            return None
        return 1 if center <= cutoff_uM else None
    if qualifier in {">", ">="}:
        return 0 if center >= cutoff_uM else None

    lower, upper = max(0.0, center - error), center + error
    if upper < cutoff_uM:
        return 1
    if lower >= cutoff_uM:
        return 0
    return None


def _similarity_groups(sequences: list[str], threshold: float = 0.80) -> np.ndarray:
    """Connected components under normalized Levenshtein similarity."""
    parent = list(range(len(sequences)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, sequence in enumerate(sequences):
        for j in range(i):
            if Levenshtein.ratio(sequence, sequences[j]) >= threshold:
                union(i, j)

    roots = [find(i) for i in range(len(sequences))]
    root_ids = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_ids[root] for root in roots], dtype=np.int64)


def _valid_sequences(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        sequence = str(value).strip().upper()
        if not (8 <= len(sequence) <= 50) or not set(sequence) <= STANDARD_AA:
            continue
        if sequence not in seen:
            seen.add(sequence)
            result.append(sequence)
    return result


def _load_uniprot_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required UniProt negative split is missing: {path}")
    frame = pd.read_csv(path)
    column = "Sequence" if "Sequence" in frame.columns else "sequence"
    if column not in frame.columns:
        raise ValueError(f"No sequence column in negative split {path}")
    sequences = _valid_sequences(frame[column])
    if not sequences:
        raise ValueError(f"No valid 8-50 aa sequences in negative split {path}")
    return sequences


def _balanced_pairs(positives: list[str], negatives: list[str], seed: int):
    positive_set = set(positives)
    negatives = [seq for seq in negatives if seq not in positive_set]
    count = min(len(positives), len(negatives))
    if count == 0:
        raise ValueError("Cannot make a balanced AMP/UniProt dataset")
    rng = np.random.default_rng(seed)
    positive_idx = rng.permutation(len(positives))[:count]
    negative_idx = rng.permutation(len(negatives))[:count]
    sequences = [positives[i] for i in positive_idx] + [negatives[i] for i in negative_idx]
    labels = np.concatenate([np.ones(count, dtype=int), np.zeros(count, dtype=int)])
    return sequences, labels


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def _new_forest(seed: int, *, balanced: bool = False) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=12 if not balanced else 8,
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=-1,
    )


def _train_amp_model():
    view_path = MAIN_ROOT / "data-engineering/data/processed/views/evaluator_view.parquet"
    frame = pd.read_parquet(view_path)
    required = {"sequence", "split", "is_amp"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Evaluator view is missing columns: {sorted(missing)}")
    positives = frame[frame["is_amp"].astype(bool)]
    train_pos = _valid_sequences(
        positives.loc[positives["split"].isin(["core_train_only", "train"]), "sequence"]
    )
    val_pos = _valid_sequences(positives.loc[positives["split"] == "validation", "sequence"])
    test_pos = _valid_sequences(positives.loc[positives["split"] == "test", "sequence"])
    if not train_pos or not val_pos or not test_pos:
        raise ValueError("AMP view must contain nonempty train, validation, and test splits")

    negative_dir = MAIN_ROOT / "vae-latent-models/data/training"
    train_neg = _load_uniprot_split(negative_dir / "Uniprot_0_25_train.csv")
    val_neg = _load_uniprot_split(negative_dir / "Uniprot_0_25_val.csv")
    test_neg = _load_uniprot_split(negative_dir / "Uniprot_0_25_test.csv")
    all_positive = set(train_pos + val_pos + test_pos)
    train_neg = [seq for seq in train_neg if seq not in all_positive]
    val_neg = [seq for seq in val_neg if seq not in all_positive]
    test_neg = [seq for seq in test_neg if seq not in all_positive]

    train_sequences, y_train = _balanced_pairs(train_pos, train_neg, seed=42)
    val_sequences, y_val = _balanced_pairs(val_pos, val_neg, seed=43)
    test_sequences, y_test = _balanced_pairs(test_pos, test_neg, seed=44)
    partitions = list(map(set, (train_sequences, val_sequences, test_sequences)))
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Exact sequences overlap between AMP train/validation/test partitions")

    model = _new_forest(seed=42)
    model.fit(np.asarray([extract_features(s) for s in train_sequences]), y_train)
    val_prob = model.predict_proba(np.asarray([extract_features(s) for s in val_sequences]))[:, 1]
    test_prob = model.predict_proba(np.asarray([extract_features(s) for s in test_sequences]))[:, 1]
    return model, {
        "train_n": len(y_train),
        "val_n": len(y_val),
        "test_n": len(y_test),
        "val": _metrics(y_val, val_prob),
        "test": _metrics(y_test, test_prob),
        "negative_definition": "UniProt 25%-identity split; unlabeled proteins treated as negatives",
    }


def _load_hemolysis_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    tox_path = MAIN_ROOT / "data-engineering/data/processed/toxicity.parquet"
    seq_path = MAIN_ROOT / "data-engineering/data/processed/sequences.parquet"
    tox = pd.read_parquet(tox_path)
    seq = pd.read_parquet(seq_path)
    required_tox = {"sequence_id", "assay_type", "measurement", "measurement_unit", "target_cell_type"}
    if missing := required_tox - set(tox.columns):
        raise ValueError(f"Toxicity data are missing columns: {sorted(missing)}")
    required_seq = {"sequence_id", "sequence", "molecular_weight_da"}
    if missing := required_seq - set(seq.columns):
        raise ValueError(f"Sequence data are missing columns: {sorted(missing)}")

    merged = tox.merge(seq[list(required_seq)], on="sequence_id", how="inner", validate="many_to_one")
    eligible = merged[
        merged["target_cell_type"].astype(str).str.strip().str.casefold().eq("human erythrocytes")
        & merged["assay_type"].astype(str).str.strip().str.casefold().eq("50% hemolysis")
    ].copy()
    eligible["label"] = eligible.apply(label_hemolysis, axis=1)
    eligible = eligible.dropna(subset=["label"])
    eligible["sequence"] = eligible["sequence"].astype(str).str.strip().str.upper()
    eligible = eligible[eligible["sequence"].map(lambda s: 8 <= len(s) <= 50 and set(s) <= STANDARD_AA)]

    # Multiple assays for one peptide are retained only when they agree after
    # unit conversion and thresholding. Conflicting peptides have no binary label.
    by_sequence = eligible.groupby("sequence", sort=True)["label"].agg(lambda labels: sorted(set(map(int, labels))))
    agreed = by_sequence[by_sequence.map(len) == 1]
    frame = pd.DataFrame({"sequence": agreed.index, "label": agreed.map(lambda values: values[0]).astype(int)})
    stats = {
        "eligible_assay_rows": int(len(merged[
            merged["target_cell_type"].astype(str).str.strip().str.casefold().eq("human erythrocytes")
            & merged["assay_type"].astype(str).str.strip().str.casefold().eq("50% hemolysis")
        ])),
        "labeled_assay_rows": int(len(eligible)),
        "unambiguous_sequences": int(len(frame)),
        "conflicting_sequences_excluded": int((by_sequence.map(len) > 1).sum()),
    }
    return frame.reset_index(drop=True), stats


def _cross_validated_hemolysis_metrics(frame: pd.DataFrame):
    sequences = frame["sequence"].tolist()
    labels = frame["label"].to_numpy(dtype=int)
    groups = _similarity_groups(sequences, threshold=0.80)
    if len(np.unique(labels)) != 2 or len(np.unique(groups)) < 5:
        raise ValueError("Hemolysis data need both classes and at least five sequence-similarity groups")

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    out_of_fold = np.full(len(frame), np.nan, dtype=float)
    fold_sizes = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(np.zeros(len(frame)), labels, groups)):
        if set(np.unique(labels[train_idx])) != {0, 1} or set(np.unique(labels[test_idx])) != {0, 1}:
            raise ValueError(f"Similarity-grouped fold {fold} is missing a class")
        model = _new_forest(seed=100 + fold, balanced=True)
        x_train = np.asarray([extract_features(sequences[i]) for i in train_idx])
        x_test = np.asarray([extract_features(sequences[i]) for i in test_idx])
        model.fit(x_train, labels[train_idx])
        out_of_fold[test_idx] = model.predict_proba(x_test)[:, 1]
        fold_sizes.append(int(len(test_idx)))
    if np.isnan(out_of_fold).any():
        raise ValueError("Grouped cross-validation did not predict every hemolysis sample")

    final_model = _new_forest(seed=42, balanced=True)
    x_all = np.asarray([extract_features(s) for s in sequences])
    final_model.fit(x_all, labels)
    return final_model, {
        "metrics": _metrics(labels, out_of_fold),
        "n_sequences": len(frame),
        "positive_n": int(np.sum(labels == 1)),
        "negative_n": int(np.sum(labels == 0)),
        "similarity_group_n": int(len(np.unique(groups))),
        "fold_sizes": fold_sizes,
        "evaluation": "5-fold out-of-fold predictions; no >=80% Levenshtein component spans folds",
    }


def main() -> None:
    started = time.time()
    print("Training Role 06 models from the checked-in data partitions")
    amp_model, amp_summary = _train_amp_model()
    hemolysis_rows, hemolysis_input_summary = _load_hemolysis_rows()
    hemo_model, hemo_summary = _cross_validated_hemolysis_metrics(hemolysis_rows)

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(amp_model, model_dir / "amp_classifier_ensemble.joblib")
    joblib.dump(hemo_model, model_dir / "empirical_hemolysis_ensemble.joblib")

    input_paths = [
        MAIN_ROOT / "data-engineering/data/processed/views/evaluator_view.parquet",
        MAIN_ROOT / "data-engineering/data/processed/toxicity.parquet",
        MAIN_ROOT / "data-engineering/data/processed/sequences.parquet",
        MAIN_ROOT / "vae-latent-models/data/training/Uniprot_0_25_train.csv",
        MAIN_ROOT / "vae-latent-models/data/training/Uniprot_0_25_val.csv",
        MAIN_ROOT / "vae-latent-models/data/training/Uniprot_0_25_test.csv",
    ]
    hashes = {str(path.relative_to(MAIN_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    summary = {
        "training_script": str(Path(__file__).relative_to(MAIN_ROOT)),
        "seed": 42,
        "feature_count": 27,
        "amp": amp_summary,
        "hemolysis_input": hemolysis_input_summary,
        "hemolysis": hemo_summary,
        "input_sha256": hashes,
        "limitations": [
            "UniProt sequences are unlabeled proteins used as negatives, not experimentally confirmed non-AMPs.",
            "The binary HC50 < 100 uM target is based only on explicit human erythrocyte 50% hemolysis records.",
            "Tree-to-tree spread is not a calibrated confidence interval.",
            "Metrics are retrospective and do not establish prospective performance.",
        ],
    }
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "reports/training_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    def metric_line(metrics):
        return (
            f"ROC-AUC {metrics['roc_auc']:.4f}; average precision "
            f"{metrics['average_precision']:.4f}; Brier score {metrics['brier_score']:.4f}"
        )

    model_card = f"""# Shared evaluator model cards

Generated from the checked-in data by `shared-evaluator/scripts/train_evaluator_models.py`.
See [`training_data_summary.json`](training_data_summary.json) for input hashes and counts.
These retrospective models are research screening tools, not validated clinical or
prospective predictors.

## AMP likelihood classifier

- Model: 50-tree Random Forest; 27 sequence and biophysical features.
- Positive class: curated AMP sequences in the evaluator view.
- Negative class: UniProt 25%-identity sequences, which are unlabeled proteins rather
  than experimentally confirmed non-AMPs.
- Rows (balanced): train {amp_summary['train_n']}; validation {amp_summary['val_n']};
  test {amp_summary['test_n']}.
- Validation: {metric_line(amp_summary['val'])}.
- Test: {metric_line(amp_summary['test'])}.

## Empirical human erythrocyte hemolysis classifier

- Model: 50-tree Random Forest with balanced class weights; 27 sequence and
  biophysical features.
- Target: HC50 < 100 µM from an explicit `50% Hemolysis` assay on human erythrocytes.
- Concentrations are converted from supported units to µM using peptide molecular
  weight. Censored or uncertainty intervals crossing 100 µM are unlabeled; sequences
  with conflicting labeled observations are excluded.
- Eligible assay rows: {hemolysis_input_summary['eligible_assay_rows']}; labeled rows:
  {hemolysis_input_summary['labeled_assay_rows']}; unambiguous unique peptides:
  {hemo_summary['n_sequences']} ({hemo_summary['positive_n']} positive,
  {hemo_summary['negative_n']} negative); similarity groups:
  {hemo_summary['similarity_group_n']}.
- Evaluation: {hemo_summary['evaluation']}.
- Out-of-fold metrics: {metric_line(hemo_summary['metrics'])}.

The split groups sequences connected by normalized Levenshtein similarity >= 0.80,
which reduces close-sequence leakage. The limited and heterogeneous assay corpus
still does not establish prospective performance.

## Biophysical membrane-disruption proxy

`biophysical_membrane_disruption_proxy` is a separate analytical heuristic using
hydropathy and hydrophobic moment. It is not an empirical assay model and is not
included in the hemolysis model's training target.

## Uncertainty and limitations

The runtime reports standard deviation across trees as ensemble spread. It is not a
calibrated confidence interval or a guarantee of out-of-domain detection. The exact
input hashes, class counts, split sizes, and known limitations are recorded in the
training summary.
"""
    (ROOT / "reports/model_cards.md").write_text(model_card)
    print(f"AMP test: {metric_line(amp_summary['test'])}")
    print(f"Hemolysis OOF: {metric_line(hemo_summary['metrics'])}")
    print(f"Saved models and summary under {ROOT}; elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
