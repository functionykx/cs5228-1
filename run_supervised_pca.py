from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

from src.supervised import Dataset, fit_and_evaluate, write_metrics_table


def load_filtered(preprocess_dir: Path) -> Dataset:
    X_train = pd.read_csv(preprocess_dir / "X_train_processed.csv")
    X_test = pd.read_csv(preprocess_dir / "X_test_processed.csv")
    y_train = pd.read_csv(preprocess_dir / "y_train.csv")["y"].astype(int).values
    y_test = pd.read_csv(preprocess_dir / "y_test.csv")["y"].astype(int).values

    keep_cols = []
    for c in X_train.columns.tolist():
        if c.startswith("cat__State_"):
            continue
        if c.startswith("cat__Area code_"):
            continue
        if c in {
            "num__Total day charge",
            "num__Total eve charge",
            "num__Total night charge",
            "num__Total intl charge",
        }:
            continue
        keep_cols.append(c)

    return Dataset(
        X_train=X_train[keep_cols].values,
        X_test=X_test[keep_cols].values,
        y_train=y_train,
        y_test=y_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-dir", default="artifacts/preprocess")
    parser.add_argument("--out", default="artifacts/supervised_nocollinear_nogeo_pca")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    data = load_filtered(Path(args.preprocess_dir))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results = []

    model_spaces = {
        "logreg_pca": (
            Pipeline(
                [
                    ("pca", PCA(random_state=args.seed)),
                    (
                        "clf",
                        LogisticRegression(
                            solver="liblinear",
                            max_iter=5000,
                            random_state=args.seed,
                        ),
                    ),
                ]
            ),
            {
                "pca__n_components": [2, 3, 4, 5, 6, 8, 10, 12, 15],
                "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "knn_pca": (
            Pipeline(
                [
                    ("pca", PCA(random_state=args.seed)),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                "pca__n_components": [2, 3, 4, 5, 6, 8, 10, 12, 15],
                "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "hgb_pca": (
            Pipeline([
                ("pca", PCA(random_state=args.seed)),
                ("clf", HistGradientBoostingClassifier(random_state=args.seed))
            ]),
            {
                "pca__n_components": [5, 10, 15],
                "clf__max_iter": [100, 200, 300],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [3, 5, None],
            },
        ),
        "xgb_pca": (
            Pipeline([
                ("pca", PCA(random_state=args.seed)),
                ("clf", xgb.XGBClassifier(random_state=args.seed, n_jobs=-1, eval_metric="logloss"))
            ]),
            {
                "pca__n_components": [5, 10, 15],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.1, 0.2],
            },
        ),
    }

    for name, (pipeline, grid) in model_spaces.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
            return_train_score=True,
        )
        search.fit(data.X_train, data.y_train)

        meta = {
            "model": name,
            "scoring": "f1",
            "best_score_cv": float(search.best_score_),
            "best_params": search.best_params_,
        }
        (out_root / f"best_params_{name}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(search.cv_results_).to_csv(out_root / f"cv_results_{name}.csv", index=False)

        best_pipe = search.best_estimator_
        results.append(fit_and_evaluate(name, best_pipe, data, out_root, tag="tuned"))

    metrics = write_metrics_table(results, out_root / "metrics.csv")
    summary_lines = [
        "# PCA Models Summary",
        "",
        "- Feature set: drop geo + drop collinear charge features (15 dimensions before PCA)",
        "- Models: PCA + Logistic Regression, PCA + KNN, PCA + XGB, PCA + HGB"
        "- Selection: train-only 5-fold CV, optimize F1",
        "",
        "```",
        metrics.round(4).to_string(index=False),
        "```",
        "",
    ]
    (out_root / "pca_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote PCA supervised outputs to: {out_root}")


if __name__ == "__main__":
    main()

