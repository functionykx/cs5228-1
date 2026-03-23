from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class KMeansSweepResult:
    scores: pd.DataFrame
    chosen_k: int


@dataclass(frozen=True)
class DBSCANSearchResult:
    scores: pd.DataFrame
    chosen_eps: float
    chosen_min_samples: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def pca_2d(X: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)


def kmeans_sweep(
    X: np.ndarray,
    *,
    k_values: list[int],
    random_state: int = 42,
) -> KMeansSweepResult:
    rows: list[dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if k >= 2 else float("nan")
        rows.append({"k": float(k), "silhouette": sil, "inertia": float(model.inertia_)})

    scores = pd.DataFrame(rows).sort_values("k")
    best = scores.dropna().sort_values(["silhouette", "k"], ascending=[False, True]).head(1)
    chosen_k = int(best["k"].iloc[0]) if not best.empty else int(k_values[0])
    return KMeansSweepResult(scores=scores, chosen_k=chosen_k)


def plot_kmeans_sweep(scores: pd.DataFrame, out_dir: Path) -> list[str]:
    _ensure_dir(out_dir)
    paths: list[str] = []

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=scores, x="k", y="silhouette", marker="o")
    plt.title("K-Means: silhouette vs k")
    p1 = out_dir / "kmeans_silhouette_vs_k.png"
    _savefig(p1)
    paths.append(p1.name)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=scores, x="k", y="inertia", marker="o")
    plt.title("K-Means: inertia vs k (elbow)")
    p2 = out_dir / "kmeans_inertia_vs_k.png"
    _savefig(p2)
    paths.append(p2.name)

    scores.to_csv(out_dir / "kmeans_k_sweep.csv", index=False)
    paths.append("kmeans_k_sweep.csv")
    return paths


def fit_kmeans(
    X: np.ndarray,
    *,
    k: int,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model


def _dbscan_score(
    X: np.ndarray, labels: np.ndarray
) -> tuple[int, float, float | None]:
    noise_ratio = float(np.mean(labels == -1))
    clusters = sorted(set(int(x) for x in labels.tolist()) - {-1})
    n_clusters = len(clusters)

    sil: float | None = None
    if n_clusters >= 2:
        mask = labels != -1
        if int(mask.sum()) >= 10:
            sil = float(silhouette_score(X[mask], labels[mask]))
    return n_clusters, noise_ratio, sil


def _eps_candidates_from_knn(
    X: np.ndarray, *, min_samples: int, quantiles: list[float]
) -> list[float]:
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kdist = np.sort(distances[:, -1])
    base = [float(np.quantile(kdist, q)) for q in quantiles]
    mult = [0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    cand = []
    for b in base:
        for m in mult:
            cand.append(float(b * m))
    cand = sorted(set([c for c in cand if c > 0]))
    return cand


def dbscan_search(
    X: np.ndarray,
    *,
    min_samples_grid: list[int],
    eps_quantiles: list[float],
    random_state: int = 42,
) -> DBSCANSearchResult:
    _ = random_state
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, int] | None = None

    for ms in min_samples_grid:
        eps_candidates = _eps_candidates_from_knn(X, min_samples=ms, quantiles=eps_quantiles)
        for eps in eps_candidates:
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X)
            n_clusters, noise_ratio, sil = _dbscan_score(X, labels)
            rows.append(
                {
                    "eps": float(eps),
                    "min_samples": float(ms),
                    "n_clusters": float(n_clusters),
                    "noise_ratio": float(noise_ratio),
                    "silhouette": float(sil) if sil is not None else float("nan"),
                }
            )

            if n_clusters < 2:
                continue
            if noise_ratio >= 0.8:
                continue
            if sil is None:
                continue
            cand = (float(sil), -float(noise_ratio), -float(ms), int(ms))
            if best is None or cand > best:
                best = cand
                chosen_eps = float(eps)
                chosen_ms = int(ms)

    scores = pd.DataFrame(rows).sort_values(["silhouette", "n_clusters"], ascending=[False, False])
    if best is None:
        nondeg = scores[scores["n_clusters"] >= 2].copy()
        if not nondeg.empty:
            nondeg = nondeg.sort_values(
                ["silhouette", "noise_ratio", "min_samples"],
                ascending=[False, True, True],
                na_position="last",
            )
            chosen_eps = float(nondeg["eps"].iloc[0])
            chosen_ms = int(nondeg["min_samples"].iloc[0])
        else:
            raise ValueError("DBSCAN search produced only degenerate results (n_clusters < 2).")

    return DBSCANSearchResult(scores=scores, chosen_eps=chosen_eps, chosen_min_samples=chosen_ms)


def plot_dbscan_search(scores: pd.DataFrame, out_dir: Path) -> list[str]:
    _ensure_dir(out_dir)
    paths: list[str] = []

    scores.to_csv(out_dir / "dbscan_param_search.csv", index=False)
    paths.append("dbscan_param_search.csv")

    plt.figure(figsize=(8, 4))
    tmp = scores.copy()
    tmp["min_samples"] = tmp["min_samples"].astype(int)
    sns.scatterplot(
        data=tmp,
        x="eps",
        y="silhouette",
        hue="min_samples",
        size="noise_ratio",
        sizes=(20, 200),
        alpha=0.8,
    )
    plt.title("DBSCAN: silhouette vs eps (size=noise ratio)")
    p1 = out_dir / "dbscan_silhouette_vs_eps.png"
    _savefig(p1)
    paths.append(p1.name)
    return paths


def fit_dbscan(X: np.ndarray, *, eps: float, min_samples: int) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    s = pd.Series(labels, name="cluster").astype(int)
    out = s.value_counts(dropna=False).sort_index().rename("count").reset_index()
    out["cluster"] = out["cluster"].astype(int)
    return out


def churn_by_cluster(labels: np.ndarray, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"cluster": labels.astype(int), "y": y.astype(int).values})
    out = df.groupby("cluster")["y"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"mean": "churn_rate"})
    return out.sort_values("churn_rate", ascending=False)


def plot_churn_by_cluster(churn_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 3))
    tmp = churn_df.copy()
    tmp["cluster"] = tmp["cluster"].astype(str)
    sns.barplot(data=tmp, x="cluster", y="churn_rate")
    plt.title(title)
    plt.ylabel("churn rate")
    _savefig(out_path)


def plot_pca_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    df = pd.DataFrame({"x": coords_2d[:, 0], "y": coords_2d[:, 1], "cluster": labels.astype(int)})
    plt.figure(figsize=(7, 5))
    if (df["cluster"] == -1).any():
        df2 = df.copy()
        df2["cluster_str"] = df2["cluster"].astype(str)
        levels = sorted(df2["cluster_str"].unique(), key=lambda s: int(s))
        colors = sns.color_palette("tab10", n_colors=max(len(levels) - 1, 1))
        palette: dict[str, object] = {}
        idx = 0
        for lv in levels:
            if lv == str(-1):
                palette[lv] = "#9e9e9e"
            else:
                palette[lv] = colors[idx % len(colors)]
                idx += 1
        sns.scatterplot(
            data=df2,
            x="x",
            y="y",
            hue="cluster_str",
            palette=palette,
            alpha=0.7,
            s=20,
            linewidth=0,
        )
        plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="cluster",
            palette="tab10",
            alpha=0.7,
            s=20,
            linewidth=0,
        )
        plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title(title)
    _savefig(out_path)


def profile_clusters(
    raw_train: pd.DataFrame,
    labels: np.ndarray,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    top_n_categories: int = 5,
) -> dict[str, pd.DataFrame]:
    df = raw_train.copy()
    df["cluster"] = labels.astype(int)

    numeric_profile = (
        df.groupby("cluster")[numeric_cols]
        .agg(["mean", "median"])
        .sort_index()
    )
    numeric_profile.columns = [f"{a}__{b}" for a, b in numeric_profile.columns]
    numeric_profile = numeric_profile.reset_index()

    cat_rows: list[dict[str, object]] = []
    for col in categorical_cols:
        tmp = (
            df.groupby(["cluster", col])
            .size()
            .rename("count")
            .reset_index()
        )
        tmp["pct_in_cluster"] = tmp["count"] / tmp.groupby("cluster")["count"].transform("sum")
        tmp = tmp.sort_values(["cluster", "pct_in_cluster"], ascending=[True, False])
        tmp = tmp.groupby("cluster").head(top_n_categories)
        for _, r in tmp.iterrows():
            cat_rows.append(
                {
                    "cluster": int(r["cluster"]),
                    "feature": col,
                    "category": str(r[col]),
                    "pct_in_cluster": float(r["pct_in_cluster"]),
                    "count": int(r["count"]),
                }
            )
    categorical_profile = pd.DataFrame(cat_rows).sort_values(
        ["cluster", "feature", "pct_in_cluster"], ascending=[True, True, False]
    )

    return {"numeric_profile": numeric_profile, "categorical_profile": categorical_profile}


def write_unsupervised_summary(
    out_path: Path,
    *,
    kmeans_meta: dict[str, object],
    dbscan_meta: dict[str, object],
    kmeans_churn: pd.DataFrame,
    dbscan_churn: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Unsupervised Learning Summary (Train Set)\n")

    lines.append("## Methods\n")
    lines.append("- K-Means: choose K via silhouette (and inertia as reference), then fit K-Means on the processed train features.")
    lines.append("- DBSCAN: run on a PCA-reduced feature space for density-based clustering; search eps/min_samples using kNN-distance quantiles, filter degenerate results, and select a non-degenerate setting.\n")

    lines.append("## K-Means (selected)\n")
    lines.append(f"- chosen_k: {kmeans_meta.get('chosen_k')}")
    if "silhouette" in kmeans_meta:
        lines.append(f"- silhouette(chosen_k): {float(kmeans_meta.get('silhouette')):.4f}")
    lines.append("\n### Churn rate by cluster (analysis only)\n")
    for _, r in kmeans_churn.iterrows():
        lines.append(f"- cluster {int(r['cluster'])}: churn_rate={float(r['churn_rate']):.4f}, n={int(r['count'])}")
    lines.append("")

    lines.append("## DBSCAN (selected)\n")
    lines.append(f"- eps: {dbscan_meta.get('eps')}")
    lines.append(f"- min_samples: {dbscan_meta.get('min_samples')}")
    lines.append(f"- n_clusters(excl. noise): {dbscan_meta.get('n_clusters')}")
    lines.append(f"- noise_ratio: {dbscan_meta.get('noise_ratio')}")
    if "silhouette" in dbscan_meta:
        lines.append(f"- silhouette(excl. noise): {float(dbscan_meta.get('silhouette')):.4f}")
    lines.append("\n### Churn rate by cluster (analysis only)\n")
    for _, r in dbscan_churn.iterrows():
        lines.append(f"- cluster {int(r['cluster'])}: churn_rate={float(r['churn_rate']):.4f}, n={int(r['count'])}")
    lines.append("")

    lines.append("## Key findings (interpretation)\n")
    if not kmeans_churn.empty:
        k_hi = kmeans_churn.sort_values("churn_rate", ascending=False).head(1).iloc[0]
        k_lo = kmeans_churn.sort_values("churn_rate", ascending=True).head(1).iloc[0]
        lines.append(
            f"- K-Means separates the training set into clusters with churn rates ranging from "
            f"{float(k_lo['churn_rate']):.4f} to {float(k_hi['churn_rate']):.4f}."
        )
        if "silhouette" in kmeans_meta:
            lines.append(
                f"- The silhouette score is relatively low, suggesting weak cluster separation in this feature space."
            )
    if not dbscan_churn.empty:
        noise_row = dbscan_churn[dbscan_churn["cluster"] == -1]
        if not noise_row.empty:
            nr = noise_row.iloc[0]
            lines.append(
                f"- DBSCAN identifies a small noise/outlier group (cluster -1) with churn_rate={float(nr['churn_rate']):.4f} "
                f"(n={int(nr['count'])}), which can indicate higher-risk outliers."
            )
    lines.append("")

    lines.append("## Limitations and next steps\n")
    lines.append("- Clustering quality depends on feature representation; try alternative embeddings (e.g., different PCA dims) and distance metrics if clusters are not well separated.")
    lines.append("- Consider removing redundant minutes/charge pairs or clustering on a compact feature subset to improve interpretability.")
    lines.append("- Use cluster profiling tables (numeric means/medians, top categorical proportions) to name clusters and propose targeted retention strategies.\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(obj: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
