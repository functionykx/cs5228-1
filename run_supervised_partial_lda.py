from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.supervised import Dataset, fit_and_evaluate, write_metrics_table


def load_filtered_df(preprocess_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(preprocess_dir / "X_train_processed.csv")
    X_test = pd.read_csv(preprocess_dir / "X_test_processed.csv")
    y_train = pd.read_csv(preprocess_dir / "y_train.csv")["y"].astype(int)
    y_test = pd.read_csv(preprocess_dir / "y_test.csv")["y"].astype(int)

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

    return X_train[keep_cols].copy(), X_test[keep_cols].copy(), y_train, y_test


def build_preprocessor(columns: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [c for c in columns if c.startswith("num__")]
    plan_cols = [
        c
        for c in columns
        if c.startswith("cat__International plan_") or c.startswith("cat__Voice mail plan_")
    ]
    num_idx = [columns.index(c) for c in num_cols]
    plan_idx = [columns.index(c) for c in plan_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("lda_num", LinearDiscriminantAnalysis(n_components=1), num_idx),
            ("plan_passthrough", "passthrough", plan_idx),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, plan_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-dir", default="artifacts/preprocess")
    parser.add_argument("--out", default="artifacts/supervised_nocollinear_nogeo_partial_lda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    X_train_df, X_test_df, y_train, y_test = load_filtered_df(Path(args.preprocess_dir))
    preprocessor, num_cols, plan_cols = build_preprocessor(X_train_df.columns.tolist())
    data = Dataset(
        X_train=X_train_df.values,
        X_test=X_test_df.values,
        y_train=y_train.values,
        y_test=y_test.values,
    )

    (out_root / "feature_partition.json").write_text(
        json.dumps(
            {
                "numeric_cols": num_cols,
                "plan_cols": plan_cols,
                "numeric_count": len(num_cols),
                "plan_count": len(plan_cols),
                "lda_output_dim": 1,
                "final_dim_after_transform": 1 + len(plan_cols),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results = []

    model_spaces = {
        "logreg_partial_lda": (
            Pipeline(
                [
                    ("prep", preprocessor),
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
                "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "knn_partial_lda": (
            Pipeline(
                [
                    ("prep", preprocessor),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
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
            "transform": {
                "numeric_to_lda_components": 1,
                "plan_passthrough": plan_cols,
            },
        }
        (out_root / f"best_params_{name}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(search.cv_results_).to_csv(out_root / f"cv_results_{name}.csv", index=False)

        results.append(fit_and_evaluate(name, search.best_estimator_, data, out_root, tag="tuned"))

    metrics = write_metrics_table(results, out_root / "metrics.csv")
    summary = [
        "# Partial LDA Models Summary",
        "",
        "- Feature set: 15 filtered features (drop geo + drop collinear charges)",
        "- Transformation: numeric features -> 1 LDA component; plan binary features are kept uncompressed",
        "- Models: partial-LDA + Logistic Regression, partial-LDA + KNN",
        "- Selection: train-only 5-fold CV, optimize F1",
        "",
        "```",
        metrics.round(4).to_string(index=False),
        "```",
        "",
    ]
    (out_root / "partial_lda_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote partial-LDA supervised outputs to: {out_root}")


if __name__ == "__main__":
    main()
