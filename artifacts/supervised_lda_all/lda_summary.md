# LDA Models Summary

- Feature set: drop geo + drop collinear charge features (15 dimensions before LDA)
- Models: LDA + Logistic Regression, LDA + KNN, LDA + HGB, LDA + XGB
- Binary classification implies LDA projects to 1 discriminant axis
- Selection: train-only 5-fold CV, optimize F1

```
     model variant split  accuracy  precision  recall     f1  roc_auc
logreg_lda   tuned train    0.7746     0.3636  0.7320 0.4859   0.8244
logreg_lda   tuned  test    0.7856     0.3723  0.7368 0.4947   0.8200
   knn_lda   tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
   knn_lda   tuned  test    0.8141     0.3210  0.2737 0.2955   0.6988
   hgb_lda   tuned train    0.8687     0.6056  0.2809 0.3838   0.8735
   hgb_lda   tuned  test    0.8471     0.4340  0.2421 0.3108   0.7872
   xgb_lda   tuned train    0.8653     0.6526  0.1598 0.2567   0.8537
   xgb_lda   tuned  test    0.8606     0.5455  0.1263 0.2051   0.8114
```
