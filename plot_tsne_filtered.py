from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def load_filtered(preprocess_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
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

    X = pd.concat([X_train[keep_cols], X_test[keep_cols]], axis=0, ignore_index=True)
    y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    split = ["train"] * len(X_train) + ["test"] * len(X_test)
    X["split"] = split
    X["target"] = y.values
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-dir", default="artifacts/preprocess")
    parser.add_argument("--out-dir", default="artifacts/visualizations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--compare-perplexities", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, _ = load_filtered(Path(args.preprocess_dir))
    meta = X[["split", "target"]].copy()
    feats = X.drop(columns=["split", "target"])

    if str(args.compare_perplexities).strip():
        perps = [float(x.strip()) for x in str(args.compare_perplexities).split(",") if x.strip()]
        fig, axes = plt.subplots(1, len(perps), figsize=(6 * len(perps), 5), squeeze=False)
        all_frames = []
        for ax, perp in zip(axes[0], perps):
            tsne = TSNE(
                n_components=2,
                perplexity=float(perp),
                random_state=args.seed,
                init="pca",
                learning_rate="auto",
                max_iter=1000,
            )
            coords = tsne.fit_transform(feats.values)
            plot_df = pd.DataFrame(
                {
                    "tsne_1": coords[:, 0],
                    "tsne_2": coords[:, 1],
                    "split": meta["split"].values,
                    "target": meta["target"].map({0: "non-churn", 1: "churn"}).values,
                    "perplexity": float(perp),
                }
            )
            all_frames.append(plot_df)
            sns.scatterplot(
                data=plot_df,
                x="tsne_1",
                y="tsne_2",
                hue="target",
                style="split",
                alpha=0.7,
                s=22,
                palette={"non-churn": "#4C78A8", "churn": "#E45756"},
                ax=ax,
                legend=(ax is axes[0][0]),
            )
            ax.set_title(f"perplexity={perp:g}")
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_filtered_15d_compare.png", dpi=220)
        plt.close()
        pd.concat(all_frames, axis=0, ignore_index=True).to_csv(
            out_dir / "tsne_filtered_points_compare.csv", index=False
        )
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=float(args.perplexity),
            random_state=args.seed,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        )
        coords = tsne.fit_transform(feats.values)

        plot_df = pd.DataFrame(
            {
                "tsne_1": coords[:, 0],
                "tsne_2": coords[:, 1],
                "split": meta["split"].values,
                "target": meta["target"].map({0: "non-churn", 1: "churn"}).values,
            }
        )
        plot_df.to_csv(out_dir / "tsne_filtered_points.csv", index=False)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x="tsne_1",
            y="tsne_2",
            hue="target",
            style="split",
            alpha=0.75,
            s=28,
            palette={"non-churn": "#4C78A8", "churn": "#E45756"},
        )
        plt.title("t-SNE on filtered 15D features")
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_filtered_15d.png", dpi=220)
        plt.close()

    (out_dir / "tsne_filtered_meta.json").write_text(
        json.dumps(
            {
                "feature_space": "drop geo + drop collinear charges",
                "input_dim": int(feats.shape[1]),
                "samples": int(feats.shape[0]),
                "perplexity": float(args.perplexity),
                "compare_perplexities": str(args.compare_perplexities),
                "note": "Visualization only. Train and test are embedded together, so this is not for model evaluation.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote t-SNE visualization to: {out_dir}")


if __name__ == "__main__":
    main()
