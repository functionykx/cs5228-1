# LDA Models Summary

- Feature set: drop geo + drop collinear charge features (15 dimensions before LDA)
- Models: LDA + Logistic Regression, LDA + KNN
- Binary classification implies LDA projects to 1 discriminant axis
- Selection: train-only 5-fold CV, optimize F1

```
     model variant split  accuracy  precision  recall     f1  roc_auc
logreg_lda   tuned train    0.7738     0.3634  0.7371 0.4868   0.8244
logreg_lda   tuned  test    0.7841     0.3717  0.7474 0.4965   0.8200
   knn_lda   tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
   knn_lda   tuned  test    0.8141     0.3210  0.2737 0.2955   0.6988
```
