# Partial LDA Models Summary

- Feature set: 15 filtered features (drop geo + drop collinear charges)
- Transformation: numeric features -> 1 LDA component; plan binary features are kept uncompressed
- Models: partial-LDA + Logistic Regression, partial-LDA + KNN
- Selection: train-only 5-fold CV, optimize F1

```
             model variant split  accuracy  precision  recall     f1  roc_auc
logreg_partial_lda   tuned train    0.7629     0.3509  0.7397 0.4760   0.8192
logreg_partial_lda   tuned  test    0.7766     0.3636  0.7579 0.4915   0.8215
   knn_partial_lda   tuned train    0.8773     0.6419  0.3557 0.4577   0.8958
   knn_partial_lda   tuned  test    0.8561     0.4906  0.2737 0.3514   0.7616
```
