# Supervised Learning Summary (Churn Prediction)

## Data

- Target: Churn (1 = churn, 0 = non-churn).
- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.

## Feature set used in this run

- Feature filter: {'drop_state': False, 'drop_area_code': False, 'drop_charges': True, 'n_features_before': 69, 'n_features_after': 69, 'removed_features_count': 0}

## Models

- ExtraTrees (randomized tree ensemble)
- HistGradientBoosting (boosted trees)
- Random Forest (tree ensemble baseline)
- xgb

## Hyperparameter tuning

- Tuning uses training data only via stratified 5-fold cross-validation.
- Optimization metric: F1-score.

- hgb: best_cv_f1=0.8482, best_params={'min_samples_leaf': 10, 'max_iter': 400, 'max_depth': None, 'learning_rate': 0.2, 'l2_regularization': 1.0}
- et: best_cv_f1=0.8067, best_params={'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': None, 'class_weight': None}
- rf: best_cv_f1=0.8022, best_params={'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': None, 'max_depth': None, 'class_weight': None}
- xgb: best_cv_f1=0.8395, best_params={'subsample': 0.8, 'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1.0}

## Threshold optimization

- For selected models, the decision threshold is optimized using training data only (out-of-fold probabilities).
- Variants containing `threshold` in the table reflect this post-processing step.

## Results (train vs test)

```
model         variant split  accuracy  precision  recall     f1  roc_auc
  hgb        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
  hgb        baseline  test    0.9520     0.8987  0.7474 0.8161   0.9060
   et        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   et        baseline  test    0.9040     0.9189  0.3579 0.5152   0.8999
   rf        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   rf        baseline  test    0.9280     0.9796  0.5053 0.6667   0.9223
  xgb        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
  xgb        baseline  test    0.9535     0.8636  0.8000 0.8306   0.9125
  hgb           tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
  hgb           tuned  test    0.9520     0.8621  0.7895 0.8242   0.9185
   et           tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
   et           tuned  test    0.9415     0.8256  0.7474 0.7845   0.9212
   rf           tuned train    0.9786     0.9321  0.9201 0.9261   0.9978
   rf           tuned  test    0.9505     0.8444  0.8000 0.8216   0.9226
  xgb           tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
  xgb           tuned  test    0.9580     0.8851  0.8105 0.8462   0.9161
  xgb threshold_tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
  xgb threshold_tuned  test    0.9550     0.9012  0.7684 0.8295   0.9161
```

## Key observations

- Best tuned model on the test set (by F1): xgb (F1=0.8462, precision=0.8851, recall=0.8105, accuracy=0.9580).
- Best model confusion matrix (test): TN=562, FP=10, FN=18, TP=77.
- et baseline: train-test F1 gap = 0.4848.
- et tuned: train-test F1 gap = 0.2155.
- hgb baseline: train-test F1 gap = 0.1839.
- hgb tuned: train-test F1 gap = 0.1758.
- rf baseline: train-test F1 gap = 0.3333.
- rf tuned: train-test F1 gap = 0.1044.
- xgb baseline: train-test F1 gap = 0.1694.
- xgb tuned: train-test F1 gap = 0.1538.

## Interpretation and diagnostics

- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).
- If accuracy is high but recall is low, the model may miss churners due to class imbalance.
- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.

## Connection to EDA (feature considerations)

- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).
- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).
- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.
