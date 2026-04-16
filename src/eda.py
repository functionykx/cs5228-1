from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


@dataclass(frozen=True)
class EDAConfig:
    top_n_state: int = 15


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def churn_overview(y: pd.Series, out_dir: Path) -> dict[str, float]:
    _ensure_dir(out_dir)
    counts = y.value_counts().sort_index()
    total = float(counts.sum())
    rate = float(counts.get(1, 0.0) / total) if total > 0 else 0.0

    plt.figure(figsize=(5, 3))
    sns.barplot(x=["FALSE", "TRUE"], y=[counts.get(0, 0), counts.get(1, 0)])
    plt.title("Churn distribution (train)")
    plt.ylabel("count")
    _savefig(out_dir / "label_distribution.png")

    return {"n": int(total), "churn_rate": rate}


def missing_value_table(X: pd.DataFrame) -> pd.DataFrame:
    miss = X.isna().sum().sort_values(ascending=False)
    return pd.DataFrame({"missing": miss, "missing_rate": miss / len(X)}).query("missing > 0")


def numeric_summary(X: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    desc = X[numeric_cols].describe(include="all").T
    return desc


def plot_univariate_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    categorical_cols: list[str],
    numeric_cols: list[str],
    out_dir: Path,
    top_n_state: int,
) -> list[str]:
    _ensure_dir(out_dir)
    saved: list[str] = []

    for col in numeric_cols:
        fig = plt.figure(figsize=(10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        sns.histplot(X[col], kde=True, ax=ax1)
        ax1.set_title(f"{col}: histogram")

        ax2 = fig.add_subplot(1, 2, 2)
        tmp = pd.DataFrame({col: X[col], "Churn": y})
        sns.boxplot(data=tmp, x="Churn", y=col, ax=ax2)
        ax2.set_title(f"{col}: by churn")

        path = out_dir / f"num_{_safe_name(col)}.png"
        _savefig(path)
        saved.append(path.name)

    for col in categorical_cols:
        s = X[col].astype(str).fillna("NA")
        if col == "State":
            vc = s.value_counts()
            top = vc.head(top_n_state)
            other = vc.iloc[top_n_state:].sum()
            plot_series = top.copy()
            if other > 0:
                plot_series.loc["Other"] = other
        else:
            plot_series = s.value_counts().head(30)

        plt.figure(figsize=(10, 3))
        sns.barplot(x=plot_series.index, y=plot_series.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{col}: frequency")
        path = out_dir / f"cat_{_safe_name(col)}_freq.png"
        _savefig(path)
        saved.append(path.name)

        rate_df = (
            pd.DataFrame({col: s, "Churn": y})
            .groupby(col)["Churn"]
            .agg(["mean", "count"])
            .sort_values("count", ascending=False)
        )
        rate_df = rate_df[rate_df["count"] >= 20].sort_values("mean", ascending=False).head(30)
        if not rate_df.empty:
            plt.figure(figsize=(10, 3))
            sns.barplot(x=rate_df.index, y=rate_df["mean"].values)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{col}: churn rate (min count=20)")
            plt.ylabel("churn rate")
            path = out_dir / f"cat_{_safe_name(col)}_churn_rate.png"
            _savefig(path)
            saved.append(path.name)

    return saved


def correlation_analysis(
    X: pd.DataFrame, numeric_cols: list[str], out_dir: Path, *, threshold: float = 0.95
) -> tuple[list[tuple[str, str, float]], str | None]:
    _ensure_dir(out_dir)
    if len(numeric_cols) < 2:
        return [], None

    corr = X[numeric_cols].corr(numeric_only=True).fillna(0.0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr, 
        cmap="RdBu_r",
        center=0,
        square=True, 
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        cbar_kws={"shrink": .8},
        vmin=-1, vmax=1
    )
    
    plt.title("Correlation heatmap (numeric features)")
    heatmap_path = out_dir / "correlation_heatmap.png"
    _savefig(heatmap_path)

    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iloc[i, j])
            if abs(v) >= threshold:
                pairs.append((cols[i], cols[j], v))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs, heatmap_path.name


def top_features_by_mutual_info(
    X_processed: pd.DataFrame, y: pd.Series, top_k: int = 20
) -> list[tuple[str, float]]:
    if X_processed.empty:
        return []
    mi = mutual_info_classif(X_processed.values, y.values, random_state=42)
    scores = list(zip(X_processed.columns.tolist(), mi.tolist()))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:top_k]


def write_summary(
    out_path: Path,
    *,
    overview: dict[str, float],
    missing: pd.DataFrame,
    numeric_desc: pd.DataFrame,
    corr_pairs: list[tuple[str, str, float]],
    top_features: list[tuple[str, float]],
    plots: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# EDA Summary (train)\n")
    lines.append(f"- n_train: {int(overview['n'])}")
    lines.append(f"- churn_rate: {overview['churn_rate']:.4f}\n")

    lines.append("## Missing Values\n")
    if missing.empty:
        lines.append("- None\n")
    else:
        lines.append("```")
        lines.append(missing.round(6).to_string())
        lines.append("```")
        lines.append("")

    lines.append("## Numeric Summary\n")
    if numeric_desc.empty:
        lines.append("- None\n")
    else:
        lines.append("```")
        lines.append(numeric_desc.round(6).to_string())
        lines.append("```")
        lines.append("")

    lines.append("## High Correlation Pairs (|corr| >= 0.95)\n")
    if not corr_pairs:
        lines.append("- None\n")
    else:
        for a, b, v in corr_pairs[:20]:
            lines.append(f"- {a} vs {b}: {v:.4f}")
        lines.append("")

    lines.append("## Top Features (Mutual Information)\n")
    if not top_features:
        lines.append("- None\n")
    else:
        for name, score in top_features:
            lines.append(f"- {name}: {score:.6f}")
        lines.append("")

    lines.append("## Plots\n")
    for p in plots:
        lines.append(f"- {p}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_").lower()
