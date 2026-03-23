# Unsupervised Learning Summary (Train Set)

## Methods

- K-Means: choose K via silhouette (and inertia as reference), then fit K-Means on the processed train features.
- DBSCAN: run on a PCA-reduced feature space for density-based clustering; search eps/min_samples using kNN-distance quantiles, filter degenerate results, and select a non-degenerate setting.

## K-Means (selected)

- chosen_k: 2
- silhouette(chosen_k): 0.0728

### Churn rate by cluster (analysis only)

- cluster 0: churn_rate=0.1698, n=1325
- cluster 1: churn_rate=0.1216, n=1341

## DBSCAN (selected)

- eps: 2.905258219346325
- min_samples: 3
- n_clusters(excl. noise): 2
- noise_ratio: 0.030382595648912228
- silhouette(excl. noise): 0.2768

### Churn rate by cluster (analysis only)

- cluster -1: churn_rate=0.2469, n=81
- cluster 0: churn_rate=0.1425, n=2582
- cluster 1: churn_rate=0.0000, n=3

## Key findings (interpretation)

- K-Means separates the training set into clusters with churn rates ranging from 0.1216 to 0.1698.
- The silhouette score is relatively low, suggesting weak cluster separation in this feature space.
- DBSCAN identifies a small noise/outlier group (cluster -1) with churn_rate=0.2469 (n=81), which can indicate higher-risk outliers.

## Limitations and next steps

- Clustering quality depends on feature representation; try alternative embeddings (e.g., different PCA dims) and distance metrics if clusters are not well separated.
- Consider removing redundant minutes/charge pairs or clustering on a compact feature subset to improve interpretability.
- Use cluster profiling tables (numeric means/medians, top categorical proportions) to name clusters and propose targeted retention strategies.
