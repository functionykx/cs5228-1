from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    target_col: str = "Churn"
    area_code_as_categorical: bool = True


@dataclass(frozen=True)
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    feature_names: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_types(
    X: pd.DataFrame, *, area_code_as_categorical: bool
) -> tuple[list[str], list[str]]:
    categorical = []
    numerical = []

    redundant_cols = [
        "Total day charge", 
        "Total eve charge", 
        "Total night charge", 
        "Total intl charge"
    ]

    for col in X.columns:
        if col in redundant_cols:
            continue
        if col == "Area code" and area_code_as_categorical:
            categorical.append(col)
            continue
        if X[col].dtype == "object":
            categorical.append(col)
        else:
            numerical.append(col)
    return categorical, numerical


def build_preprocessor(
    categorical_cols: list[str], numerical_cols: list[str]
) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot()),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_cols),
            ("num", numeric_pipe, numerical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_transform(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    area_code_as_categorical: bool,
) -> PreprocessArtifacts:
    categorical_cols, numerical_cols = infer_feature_types(
        X_train, area_code_as_categorical=area_code_as_categorical
    )
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    preprocessor.fit(X_train, y_train)

    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(Xt_train.shape[1])]

    X_train_out = pd.DataFrame(Xt_train, columns=feature_names, index=X_train.index)
    X_test_out = pd.DataFrame(Xt_test, columns=feature_names, index=X_test.index)

    return PreprocessArtifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_train=X_train_out,
        X_test=X_test_out,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def save_artifacts(artifacts: PreprocessArtifacts, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    preprocessor_path = out / "preprocessor.joblib"
    joblib.dump(artifacts.preprocessor, preprocessor_path)

    feature_names_path = out / "feature_names.json"
    feature_names_path.write_text(
        json.dumps(artifacts.feature_names, ensure_ascii=False, indent=2)
    )

    x_train_path = out / "X_train_processed.csv"
    x_test_path = out / "X_test_processed.csv"
    y_train_path = out / "y_train.csv"
    y_test_path = out / "y_test.csv"
    train_with_y_path = out / "train_processed_with_y.csv"
    test_with_y_path = out / "test_processed_with_y.csv"

    artifacts.X_train.to_csv(x_train_path, index=False)
    artifacts.X_test.to_csv(x_test_path, index=False)
    artifacts.y_train.to_frame("y").to_csv(y_train_path, index=False)
    artifacts.y_test.to_frame("y").to_csv(y_test_path, index=False)
    pd.concat([artifacts.X_train.reset_index(drop=True), artifacts.y_train.rename("y")], axis=1).to_csv(
        train_with_y_path, index=False
    )
    pd.concat([artifacts.X_test.reset_index(drop=True), artifacts.y_test.rename("y")], axis=1).to_csv(
        test_with_y_path, index=False
    )

    meta_path = out / "meta.json"
    meta = {
        "n_train": int(artifacts.X_train.shape[0]),
        "n_test": int(artifacts.X_test.shape[0]),
        "n_features": int(artifacts.X_train.shape[1]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "preprocessor": str(preprocessor_path),
        "feature_names": str(feature_names_path),
        "X_train": str(x_train_path),
        "X_test": str(x_test_path),
        "y_train": str(y_train_path),
        "y_test": str(y_test_path),
        "train_with_y": str(train_with_y_path),
        "test_with_y": str(test_with_y_path),
        "meta": str(meta_path),
    }
