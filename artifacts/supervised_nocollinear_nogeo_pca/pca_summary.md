# PCA Models Summary

- Feature set: drop geo + drop collinear charge features (15 dimensions before PCA)
- Models: PCA + Logistic Regression, PCA + KNN, PCA + XGB, PCA + HGB- Selection: train-only 5-fold CV, optimize F1

```
     model variant split  accuracy  precision  recall     f1  roc_auc
logreg_pca   tuned train    0.7693     0.3604  0.7552 0.4879   0.8250
logreg_pca   tuned  test    0.7751     0.3618  0.7579 0.4898   0.8308
   knn_pca   tuned train    0.9269     0.9177  0.5464 0.6850   0.9711
   knn_pca   tuned  test    0.8966     0.7955  0.3684 0.5036   0.7470
   hgb_pca   tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
   hgb_pca   tuned  test    0.9130     0.7846  0.5368 0.6375   0.8834
   xgb_pca   tuned train    0.9861     0.9944  0.9098 0.9502   0.9982
   xgb_pca   tuned  test    0.9130     0.8033  0.5158 0.6282   0.8884
```
