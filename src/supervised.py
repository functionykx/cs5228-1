from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import xgboost as xgb

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_").lower()


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def _predict_score(model: BaseEstimator, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if isinstance(s, np.ndarray):
            return s
    return None


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    *,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    csv_path = Path(str(out_path)).with_suffix(".csv")
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(csv_path)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _supports_sample_weight(estimator: BaseEstimator) -> bool:
    try:
        sig = inspect.signature(estimator.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False


def _fit_with_optional_sample_weight(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray
) -> None:
    class_weight = getattr(estimator, "class_weight", None)
    if _supports_sample_weight(estimator) and class_weight is None:
        sw = compute_sample_weight(class_weight="balanced", y=y)
        estimator.fit(X, y, sample_weight=sw)
        return
    estimator.fit(X, y)


def oof_scores(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
    calibrate: bool = False,
    calib_method: str = "sigmoid",
    calib_size: float = 0.2,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)

    for fold_train_idx, fold_val_idx in cv.split(X, y):
        X_tr = X[fold_train_idx]
        y_tr = y[fold_train_idx]
        X_val = X[fold_val_idx]

        if calibrate:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=calib_size, random_state=seed
            )
            fit_rel, cal_rel = next(sss.split(X_tr, y_tr))
            base = clone(estimator)
            _fit_with_optional_sample_weight(base, X_tr[fit_rel], y_tr[fit_rel])
            cal = CalibratedClassifierCV(base, cv="prefit", method=calib_method)
            cal.fit(X_tr[cal_rel], y_tr[cal_rel])
            s = _predict_score(cal, X_val)
        else:
            base = clone(estimator)
            _fit_with_optional_sample_weight(base, X_tr, y_tr)
            s = _predict_score(base, X_val)

        if s is None:
            raise ValueError("estimator does not support probability/score prediction")
        scores[fold_val_idx] = s

    if np.isnan(scores).any():
        raise ValueError("failed to compute out-of-fold scores")
    return scores


def threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, *, steps: int = 201
) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, num=int(steps))
    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    sweep: pd.DataFrame,
    *,
    target_recall: float | None = None,
    target_precision: float | None = None,
) -> tuple[float, dict[str, object]]:
    df = sweep.copy()
    if target_recall is not None:
        df = df[df["recall"] >= float(target_recall)]
    if target_precision is not None:
        df = df[df["precision"] >= float(target_precision)]
    if df.empty:
        df = sweep.copy()
    best = df.sort_values(["f1", "threshold"], ascending=[False, True]).head(1).iloc[0]
    t = float(best["threshold"])
    meta = {
        "threshold": t,
        "precision_oof": float(best["precision"]),
        "recall_oof": float(best["recall"]),
        "f1_oof": float(best["f1"]),
        "target_recall": target_recall,
        "target_precision": target_precision,
    }
    return t, meta


def evaluate_thresholded_scores(
    name: str,
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_score_train: np.ndarray,
    y_score_test: np.ndarray,
    threshold: float,
    out_dir: str | Path,
    tag: str,
) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_pred_train = (y_score_train >= threshold).astype(int)
    y_pred_test = (y_score_test >= threshold).astype(int)

    train_metrics = evaluate_binary(y_train, y_pred_train, y_score_train)
    test_metrics = evaluate_binary(y_test, y_pred_test, y_score_test)

    cm_train_path = out / f"confusion_matrix_{_safe_name(name)}_{tag}_train.png"
    cm_test_path = out / f"confusion_matrix_{_safe_name(name)}_{tag}_test.png"
    save_confusion_matrix(
        y_train,
        y_pred_train,
        cm_train_path,
        title=f"{name} ({tag}) - Train",
    )
    save_confusion_matrix(
        y_test,
        y_pred_test,
        cm_test_path,
        title=f"{name} ({tag}) - Test",
    )

    return {
        "name": name,
        "tag": tag,
        "train": train_metrics,
        "test": test_metrics,
        "threshold": float(threshold),
        "confusion_train_png": str(cm_train_path),
        "confusion_test_png": str(cm_test_path),
    }


def baseline_models(*, seed: int = 42, model_set: str = "original") -> dict[str, BaseEstimator]:
    if model_set == "alt":
        return {
            "hgb": HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=6,
                max_iter=300,
                random_state=seed,
            ),
            "et": ExtraTreesClassifier(
                n_estimators=600,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "rf": RandomForestClassifier(
                n_estimators=400,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "xgb": xgb.XGBClassifier(
                random_state=seed, 
                n_jobs=-1, 
                eval_metric="logloss",
                use_label_encoder=False
            ) # try XGBoost
        }
    return {
        "logreg": LogisticRegression(
            max_iter=5000, solver="liblinear", class_weight="balanced", random_state=seed
        ),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", p=2),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def tuned_search_spaces(
    *, seed: int = 42, model_set: str = "original"
) -> dict[str, tuple[BaseEstimator, dict[str, list[object]]]]:
    if model_set == "alt":
        return {
            "hgb": (
                HistGradientBoostingClassifier(random_state=seed),
                {
                    "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, None],
                    "max_iter": [200, 400, 800],
                    "min_samples_leaf": [10, 20, 50],
                    "l2_regularization": [0.0, 0.1, 1.0],
                },
            ),
            "et": (
                ExtraTreesClassifier(random_state=seed, n_jobs=-1),
                {
                    "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "max_features": ["sqrt", "log2", None],
                    "class_weight": [None, "balanced"],
                },
            ),
            "rf": (
                RandomForestClassifier(random_state=seed, n_jobs=-1),
                {
                    "n_estimators": [200, 400, 800],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "max_features": ["sqrt", "log2", None],
                    "class_weight": [None, "balanced"],
                },
            ),
            "xgb": (
                xgb.XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss"),
                {
                    "n_estimators": [100, 200, 400],
                    "max_depth": [3, 6, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1, 0.2]
                }
            )
        }

    return {
        "logreg": (
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed),
            {
                "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                "penalty": ["l1", "l2"],
                "class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced"],
            },
        ),
    }


def tuned_search_spaces_strong_regularization(
    *, seed: int = 42, model_set: str = "alt"
) -> dict[str, tuple[BaseEstimator, dict[str, list[object]]]]:
    if model_set != "alt":
        raise ValueError("strong regularization search spaces are defined only for model_set='alt'")
    return {
        "hgb": (
            HistGradientBoostingClassifier(random_state=seed),
            {
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_depth": [3, 4, 5],
                "max_iter": [300, 600, 1000],
                "min_samples_leaf": [30, 50, 80, 120],
                "l2_regularization": [0.1, 1.0, 5.0, 10.0],
            },
        ),
        "et": (
            ExtraTreesClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [300, 600, 1000],
                "max_depth": [5, 8, 12, 16],
                "min_samples_split": [5, 10, 20],
                "min_samples_leaf": [5, 10, 20],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [300, 600, 1000],
                "max_depth": [5, 8, 12, 16],
                "min_samples_split": [5, 10, 20],
                "min_samples_leaf": [5, 10, 20],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        ),
    }


def fit_and_evaluate(
    name: str,
    model: BaseEstimator,
    data: Dataset,
    out_dir: str | Path,
    *,
    tag: str,
) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _fit_with_optional_sample_weight(model, data.X_train, data.y_train)

    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)
    y_score_train = _predict_score(model, data.X_train)
    y_score_test = _predict_score(model, data.X_test)

    train_metrics = evaluate_binary(data.y_train, y_pred_train, y_score_train)
    test_metrics = evaluate_binary(data.y_test, y_pred_test, y_score_test)

    cm_train_path = out / f"confusion_matrix_{_safe_name(name)}_{tag}_train.png"
    cm_test_path = out / f"confusion_matrix_{_safe_name(name)}_{tag}_test.png"
    save_confusion_matrix(
        data.y_train,
        y_pred_train,
        cm_train_path,
        title=f"{name} ({tag}) - Train",
    )
    save_confusion_matrix(
        data.y_test,
        y_pred_test,
        cm_test_path,
        title=f"{name} ({tag}) - Test",
    )

    dump(model, out / f"model_{_safe_name(name)}_{tag}.joblib")

    return {
        "name": name,
        "tag": tag,
        "train": train_metrics,
        "test": test_metrics,
        "confusion_train_png": str(cm_train_path),
        "confusion_test_png": str(cm_test_path),
    }


def tune_model(
    name: str,
    estimator: BaseEstimator,
    param_distributions: dict[str, list[object]],
    data: Dataset,
    out_dir: str | Path,
    *,
    seed: int = 42,
    n_iter: int = 30,
) -> tuple[BaseEstimator, dict[str, object]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        random_state=seed,
        refit=True,
        verbose=0,
        return_train_score=True,
    )
    sw = None
    class_weight = getattr(estimator, "class_weight", None)
    if _supports_sample_weight(estimator) and class_weight is None:
        sw = compute_sample_weight(class_weight="balanced", y=data.y_train)
    if sw is not None:
        search.fit(data.X_train, data.y_train, sample_weight=sw)
    else:
        search.fit(data.X_train, data.y_train)

    best_params = search.best_params_
    meta = {
        "model": name,
        "scoring": "f1",
        "n_iter": int(n_iter),
        "best_score_cv": float(search.best_score_),
        "best_params": best_params,
    }
    (out / f"best_params_{_safe_name(name)}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(out / f"cv_results_{_safe_name(name)}.csv", index=False)

    return search.best_estimator_, meta


def write_metrics_table(
    results: list[dict[str, object]], out_path: str | Path
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for r in results:
        name = str(r["name"])
        tag = str(r["tag"])
        for split in ["train", "test"]:
            m = r[split]
            rows.append(
                {
                    "model": name,
                    "variant": tag,
                    "split": split,
                    "accuracy": float(m["accuracy"]),
                    "precision": float(m["precision"]),
                    "recall": float(m["recall"]),
                    "f1": float(m["f1"]),
                    "roc_auc": float(m.get("roc_auc", float("nan"))),
                }
            )
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def write_summary_english(
    out_path: str | Path,
    *,
    metrics: pd.DataFrame,
    tuning_meta: list[dict[str, object]],
    best_confusion: dict[str, int] | None = None,
    feature_filter: dict[str, object] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Supervised Learning Summary (Churn Prediction)\n")
    lines.append("## Data\n")
    lines.append("- Target: Churn (1 = churn, 0 = non-churn).")
    lines.append("- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.\n")

    if feature_filter is not None:
        lines.append("## Feature set used in this run\n")
        lines.append(f"- Feature filter: {feature_filter}\n")

    lines.append("## Models\n")
    model_desc = {
        "logreg": "Logistic Regression (linear baseline)",
        "knn": "K-Nearest Neighbors (distance-based baseline)",
        "rf": "Random Forest (tree ensemble baseline)",
        "hgb": "HistGradientBoosting (boosted trees)",
        "et": "ExtraTrees (randomized tree ensemble)",
    }
    for m in sorted(metrics["model"].unique().tolist()):
        lines.append(f"- {model_desc.get(m, m)}")
    lines.append("")

    lines.append("## Hyperparameter tuning\n")
    lines.append("- Tuning uses training data only via stratified 5-fold cross-validation.")
    lines.append("- Optimization metric: F1-score.\n")
    for m in tuning_meta:
        lines.append(
            f"- {m['model']}: best_cv_f1={float(m['best_score_cv']):.4f}, best_params={m['best_params']}"
        )
    lines.append("")

    if metrics["variant"].astype(str).str.contains("threshold").any():
        lines.append("## Threshold optimization\n")
        lines.append("- For selected models, the decision threshold is optimized using training data only (out-of-fold probabilities).")
        lines.append("- Variants containing `threshold` in the table reflect this post-processing step.\n")

    lines.append("## Results (train vs test)\n")
    lines.append("```")
    lines.append(metrics.round(4).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Key observations\n")
    tuned_test = metrics[(metrics["variant"] == "tuned") & (metrics["split"] == "test")].copy()
    if not tuned_test.empty:
        best = tuned_test.sort_values(["f1", "recall"], ascending=[False, False]).head(1).iloc[0]
        lines.append(
            f"- Best tuned model on the test set (by F1): {best['model']} "
            f"(F1={best['f1']:.4f}, precision={best['precision']:.4f}, recall={best['recall']:.4f}, accuracy={best['accuracy']:.4f})."
        )
        if best_confusion is not None:
            tn = int(best_confusion.get("tn", 0))
            fp = int(best_confusion.get("fp", 0))
            fn = int(best_confusion.get("fn", 0))
            tp = int(best_confusion.get("tp", 0))
            lines.append(f"- Best model confusion matrix (test): TN={tn}, FP={fp}, FN={fn}, TP={tp}.")

    def _gap(df: pd.DataFrame, model: str, variant: str) -> float | None:
        t = df[(df["model"] == model) & (df["variant"] == variant) & (df["split"] == "train")]
        s = df[(df["model"] == model) & (df["variant"] == variant) & (df["split"] == "test")]
        if t.empty or s.empty:
            return None
        return float(t["f1"].iloc[0] - s["f1"].iloc[0])

    for model in sorted(metrics["model"].unique().tolist()):
        gap_b = _gap(metrics, model, "baseline")
        gap_t = _gap(metrics, model, "tuned")
        if gap_b is not None:
            lines.append(f"- {model} baseline: train-test F1 gap = {gap_b:.4f}.")
        if gap_t is not None:
            lines.append(f"- {model} tuned: train-test F1 gap = {gap_t:.4f}.")
    lines.append("")

    lines.append("## Interpretation and diagnostics\n")
    lines.append("- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).")
    lines.append("- If accuracy is high but recall is low, the model may miss churners due to class imbalance.")
    lines.append("- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.\n")

    lines.append("## Connection to EDA (feature considerations)\n")
    lines.append("- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).")
    lines.append("- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).")
    lines.append("- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.\n")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
