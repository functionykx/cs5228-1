from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

from src.supervised import (
    baseline_models,
    fit_and_evaluate,
    Dataset,
    choose_threshold,
    evaluate_thresholded_scores,
    oof_scores,
    tune_model,
    tuned_search_spaces,
    tuned_search_spaces_strong_regularization,
    threshold_sweep,
    write_metrics_table,
    write_summary_english,
)

def _load_processed_df(preprocess_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(preprocess_dir / "X_train_processed.csv")
    X_test = pd.read_csv(preprocess_dir / "X_test_processed.csv")
    y_train = pd.read_csv(preprocess_dir / "y_train.csv")["y"].astype(int)
    y_test = pd.read_csv(preprocess_dir / "y_test.csv")["y"].astype(int)
    return X_train, X_test, y_train, y_test


def _filter_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    drop_state: bool,
    drop_area_code: bool,
    drop_charges: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    drop_prefixes: list[str] = []
    drop_exact: set[str] = set()

    if drop_state:
        drop_prefixes.append("cat__State_")
    if drop_area_code:
        drop_prefixes.append("cat__Area code_")
    if drop_charges:
        drop_exact.update(
            {
                "num__Total day charge",
                "num__Total eve charge",
                "num__Total night charge",
                "num__Total intl charge",
            }
        )

    keep_cols: list[str] = []
    removed_cols: list[str] = []
    for c in X_train.columns.tolist():
        if c in drop_exact or any(c.startswith(p) for p in drop_prefixes):
            removed_cols.append(c)
        else:
            keep_cols.append(c)

    meta = {
        "drop_state": drop_state,
        "drop_area_code": drop_area_code,
        "drop_charges": drop_charges,
        "n_features_before": int(X_train.shape[1]),
        "n_features_after": int(len(keep_cols)),
        "removed_features_count": int(len(removed_cols)),
    }
    return X_train[keep_cols].copy(), X_test[keep_cols].copy(), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-dir", default="artifacts/preprocess")
    parser.add_argument("--out", default="artifacts/supervised")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--model-set", default="original", choices=["original", "alt"])
    parser.add_argument("--strong-regularization", action="store_true", default=False)
    parser.add_argument("--threshold-optimize", action="store_true", default=False)
    parser.add_argument("--threshold-models", default="logreg,knn,xgb")
    parser.add_argument("--threshold-steps", type=int, default=201)
    parser.add_argument("--target-recall", type=float, default=None)
    parser.add_argument("--target-precision", type=float, default=None)
    parser.add_argument("--calibrate-probabilities", action="store_true", default=False)
    parser.add_argument("--calibration-method", default="sigmoid", choices=["sigmoid", "isotonic"])
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--drop-state", action="store_true", default=False)
    parser.add_argument("--drop-area-code", action="store_true", default=False)
    parser.add_argument("--drop-charges", action="store_true", default=False)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    preprocess_dir = Path(args.preprocess_dir)
    X_train_df, X_test_df, y_train, y_test = _load_processed_df(preprocess_dir)
    X_train_df, X_test_df, feature_filter_meta = _filter_features(
        X_train_df,
        X_test_df,
        drop_state=args.drop_state,
        drop_area_code=args.drop_area_code,
        drop_charges=args.drop_charges,
    )
    (out_root / "feature_filter.json").write_text(
        json.dumps(feature_filter_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    data = Dataset(
        X_train=X_train_df.values,
        X_test=X_test_df.values,
        y_train=y_train.values,
        y_test=y_test.values,
    )

    results = []
    for name, model in baseline_models(seed=args.seed, model_set=args.model_set).items():
        results.append(
            fit_and_evaluate(name, model, data, out_root, tag="baseline")
        )

    tuning_meta = []
    tuned_results = []
    if args.strong_regularization:
        spaces = tuned_search_spaces_strong_regularization(seed=args.seed, model_set=args.model_set)
    else:
        spaces = tuned_search_spaces(seed=args.seed, model_set=args.model_set)

    threshold_models = set([s.strip() for s in str(args.threshold_models).split(",") if s.strip()])

    for name, (est, space) in spaces.items():
        best_est, meta = tune_model(
            name,
            est,
            space,
            data,
            out_root,
            seed=args.seed,
            n_iter=args.n_iter,
        )
        tuning_meta.append(meta)
        tuned_results.append(
            fit_and_evaluate(name, best_est, data, out_root, tag="tuned")
        )

        if args.threshold_optimize and name in threshold_models:
            tag = "threshold_tuned"
            calibrated = False
            if args.calibrate_probabilities:
                tag = "calibrated_threshold_tuned"
                calibrated = True

            oof = oof_scores(
                clone(best_est),
                data.X_train,
                data.y_train,
                seed=args.seed,
                n_splits=5,
                calibrate=calibrated,
                calib_method=args.calibration_method,
                calib_size=float(args.calibration_size),
            )
            sweep = threshold_sweep(data.y_train, oof, steps=int(args.threshold_steps))
            t_star, t_meta = choose_threshold(
                sweep,
                target_recall=args.target_recall,
                target_precision=args.target_precision,
            )

            sweep_path = out_root / f"threshold_sweep_{name}_{tag}.csv"
            sweep.to_csv(sweep_path, index=False)

            meta_out = {
                "model": name,
                "variant": tag,
                "objective": "f1",
                "threshold_steps": int(args.threshold_steps),
                "threshold": float(t_star),
                "constraints": {
                    "target_recall": args.target_recall,
                    "target_precision": args.target_precision,
                },
                "oof_at_threshold": t_meta,
                "calibration": {
                    "enabled": calibrated,
                    "method": args.calibration_method if calibrated else None,
                    "size": float(args.calibration_size) if calibrated else None,
                },
            }
            (out_root / f"best_threshold_{name}_{tag}.json").write_text(
                json.dumps(meta_out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if calibrated:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=float(args.calibration_size), random_state=args.seed
                )
                fit_idx, cal_idx = next(sss.split(data.X_train, data.y_train))
                base = clone(best_est)
                base.fit(data.X_train[fit_idx], data.y_train[fit_idx])
                cal = CalibratedClassifierCV(base, cv="prefit", method=args.calibration_method)
                cal.fit(data.X_train[cal_idx], data.y_train[cal_idx])
                y_score_train = cal.predict_proba(data.X_train)[:, 1]
                y_score_test = cal.predict_proba(data.X_test)[:, 1]
            else:
                y_score_train = best_est.predict_proba(data.X_train)[:, 1]
                y_score_test = best_est.predict_proba(data.X_test)[:, 1]

            tuned_results.append(
                evaluate_thresholded_scores(
                    name,
                    y_train=data.y_train,
                    y_test=data.y_test,
                    y_score_train=np.asarray(y_score_train, dtype=float),
                    y_score_test=np.asarray(y_score_test, dtype=float),
                    threshold=float(t_star),
                    out_dir=out_root,
                    tag=tag,
                )
            )

    all_results = results + tuned_results
    metrics = write_metrics_table(all_results, out_root / "metrics.csv")

    best_conf = None
    tuned_test = metrics[(metrics["variant"] == "tuned") & (metrics["split"] == "test")].copy()
    if not tuned_test.empty:
        best_row = tuned_test.sort_values(["f1", "recall"], ascending=[False, False]).head(1).iloc[0]
        cm_path = out_root / f"confusion_matrix_{best_row['model']}_tuned_test.csv"
        if cm_path.exists():
            cm = pd.read_csv(cm_path, index_col=0)
            best_conf = {
                "tn": int(cm.loc["true_0", "pred_0"]),
                "fp": int(cm.loc["true_0", "pred_1"]),
                "fn": int(cm.loc["true_1", "pred_0"]),
                "tp": int(cm.loc["true_1", "pred_1"]),
            }

    write_summary_english(
        out_root / "supervised_summary.md",
        metrics=metrics,
        tuning_meta=tuning_meta,
        best_confusion=best_conf,
        feature_filter=feature_filter_meta,
    )

    print(f"Wrote supervised outputs to: {out_root}")


if __name__ == "__main__":
    main()
