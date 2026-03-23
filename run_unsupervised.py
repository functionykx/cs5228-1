from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.unsupervised import (
    churn_by_cluster,
    cluster_sizes,
    dbscan_search,
    fit_dbscan,
    fit_kmeans,
    kmeans_sweep,
    pca_2d,
    plot_churn_by_cluster,
    plot_dbscan_search,
    plot_kmeans_sweep,
    plot_pca_scatter,
    profile_clusters,
    save_json,
    write_unsupervised_summary,
)


def _load_processed(preprocess_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    x_path = preprocess_dir / "X_train_processed.csv"
    feature_path = preprocess_dir / "feature_names.json"
    y_path = preprocess_dir / "y_train.csv"
    if not x_path.exists() or not y_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            f"Missing preprocess artifacts under {preprocess_dir}. "
            "Run: python3 run_preprocess_eda.py --out artifacts"
        )
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)["y"]
    return X, y


def _filter_features(
    X: pd.DataFrame,
    *,
    drop_state: bool,
    drop_area_code: bool,
    drop_charges: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
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
    for c in X.columns:
        if c in drop_exact:
            removed_cols.append(c)
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            removed_cols.append(c)
            continue
        keep_cols.append(c)

    meta = {
        "drop_state": drop_state,
        "drop_area_code": drop_area_code,
        "drop_charges": drop_charges,
        "n_features_before": int(X.shape[1]),
        "n_features_after": int(len(keep_cols)),
        "removed_features_count": int(len(removed_cols)),
    }
    return X[keep_cols].copy(), meta


def _load_raw_train(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="churn-bigml-80.csv")
    parser.add_argument("--preprocess-dir", default="artifacts/preprocess")
    parser.add_argument("--out", default="artifacts/unsupervised")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=10)
    parser.add_argument("--drop-state", action="store_true", default=False)
    parser.add_argument("--drop-area-code", action="store_true", default=False)
    parser.add_argument("--drop-charges", action="store_true", default=False)
    args = parser.parse_args()

    preprocess_dir = Path(args.preprocess_dir)
    out_root = Path(args.out)
    kmeans_dir = out_root / "kmeans"
    dbscan_dir = out_root / "dbscan"
    out_root.mkdir(parents=True, exist_ok=True)

    X_df, y = _load_processed(preprocess_dir)
    X_df, feature_filter_meta = _filter_features(
        X_df,
        drop_state=args.drop_state,
        drop_area_code=args.drop_area_code,
        drop_charges=args.drop_charges,
    )
    X = X_df.values
    raw_train = _load_raw_train(Path(args.train))
    coords = pca_2d(X, random_state=args.seed)
    X_db = PCA(n_components=min(10, X.shape[1]), random_state=args.seed).fit_transform(X)

    numeric_cols = [c for c in raw_train.columns if raw_train[c].dtype != "object"]
    categorical_cols = [c for c in raw_train.columns if raw_train[c].dtype == "object"]

    sweep = kmeans_sweep(
        X,
        k_values=list(range(args.kmin, args.kmax + 1)),
        random_state=args.seed,
    )
    kmeans_plots = plot_kmeans_sweep(sweep.scores, kmeans_dir)
    k_labels, _ = fit_kmeans(X, k=sweep.chosen_k, random_state=args.seed)

    pd.DataFrame({"cluster": k_labels.astype(int)}).to_csv(
        kmeans_dir / "kmeans_labels.csv", index=False
    )
    cluster_sizes(k_labels).to_csv(kmeans_dir / "kmeans_cluster_sizes.csv", index=False)
    k_churn = churn_by_cluster(k_labels, y)
    k_churn.to_csv(kmeans_dir / "kmeans_churn_by_cluster.csv", index=False)
    plot_churn_by_cluster(
        k_churn,
        kmeans_dir / "kmeans_churn_by_cluster.png",
        title="K-Means: churn rate by cluster (analysis only)",
    )
    plot_pca_scatter(
        coords,
        k_labels,
        kmeans_dir / "kmeans_pca_scatter.png",
        title="K-Means clusters (PCA 2D)",
    )
    k_profile = profile_clusters(
        raw_train,
        k_labels,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    k_profile["numeric_profile"].to_csv(kmeans_dir / "kmeans_cluster_profile_numeric.csv", index=False)
    k_profile["categorical_profile"].to_csv(
        kmeans_dir / "kmeans_cluster_profile_categorical.csv", index=False
    )

    dbscan_search_res = dbscan_search(
        X_db,
        min_samples_grid=[3, 5, 10],
        eps_quantiles=[0.90, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999],
        random_state=args.seed,
    )
    dbscan_plots = plot_dbscan_search(dbscan_search_res.scores, dbscan_dir)
    d_labels = fit_dbscan(
        X_db, eps=dbscan_search_res.chosen_eps, min_samples=dbscan_search_res.chosen_min_samples
    )

    pd.DataFrame({"cluster": d_labels.astype(int)}).to_csv(
        dbscan_dir / "dbscan_labels.csv", index=False
    )
    cluster_sizes(d_labels).to_csv(dbscan_dir / "dbscan_cluster_sizes.csv", index=False)
    d_churn = churn_by_cluster(d_labels, y)
    d_churn.to_csv(dbscan_dir / "dbscan_churn_by_cluster.csv", index=False)
    plot_churn_by_cluster(
        d_churn,
        dbscan_dir / "dbscan_churn_by_cluster.png",
        title="DBSCAN: churn rate by cluster (analysis only)",
    )
    plot_pca_scatter(
        coords,
        d_labels,
        dbscan_dir / "dbscan_pca_scatter.png",
        title="DBSCAN clusters (PCA 2D)",
    )
    d_profile = profile_clusters(
        raw_train,
        d_labels,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    d_profile["numeric_profile"].to_csv(dbscan_dir / "dbscan_cluster_profile_numeric.csv", index=False)
    d_profile["categorical_profile"].to_csv(
        dbscan_dir / "dbscan_cluster_profile_categorical.csv", index=False
    )

    k_meta = {
        "chosen_k": int(sweep.chosen_k),
        "silhouette": float(
            sweep.scores.loc[sweep.scores["k"] == float(sweep.chosen_k), "silhouette"].iloc[0]
        ),
        "plots": kmeans_plots,
        "feature_filter": feature_filter_meta,
    }
    save_json(k_meta, kmeans_dir / "kmeans_meta.json")

    d_sizes = cluster_sizes(d_labels)
    n_clusters = int((d_sizes["cluster"] != -1).sum())
    noise_ratio = float(d_sizes.loc[d_sizes["cluster"] == -1, "count"].sum() / len(d_labels))
    best_row = (
        dbscan_search_res.scores[
            (dbscan_search_res.scores["eps"] == dbscan_search_res.chosen_eps)
            & (dbscan_search_res.scores["min_samples"] == float(dbscan_search_res.chosen_min_samples))
        ]
        .head(1)
    )
    d_sil = float(best_row["silhouette"].iloc[0]) if not best_row.empty else float("nan")
    d_meta = {
        "eps": float(dbscan_search_res.chosen_eps),
        "min_samples": int(dbscan_search_res.chosen_min_samples),
        "n_clusters": int(n_clusters),
        "noise_ratio": float(noise_ratio),
        "silhouette": float(d_sil),
        "plots": dbscan_plots,
        "feature_filter": feature_filter_meta,
    }
    save_json(d_meta, dbscan_dir / "dbscan_meta.json")

    write_unsupervised_summary(
        out_root / "unsupervised_summary.md",
        kmeans_meta=k_meta,
        dbscan_meta=d_meta,
        kmeans_churn=k_churn,
        dbscan_churn=d_churn,
    )

    print(f"Wrote K-Means outputs to: {kmeans_dir}")
    print(f"Wrote DBSCAN outputs to: {dbscan_dir}")
    print(f"Wrote summary to: {out_root / 'unsupervised_summary.md'}")


if __name__ == "__main__":
    main()
