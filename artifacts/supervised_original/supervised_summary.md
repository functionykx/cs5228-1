# Supervised Learning Summary (Churn Prediction)

## Data

- Target: Churn (1 = churn, 0 = non-churn).
- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.

## Feature set used in this run

- Feature filter: {'drop_state': False, 'drop_area_code': False, 'drop_charges': True, 'n_features_before': 69, 'n_features_after': 69, 'removed_features_count': 0}

## Models

- K-Nearest Neighbors (distance-based baseline)
- Logistic Regression (linear baseline)
- Random Forest (tree ensemble baseline)

## Hyperparameter tuning

- Tuning uses training data only via stratified 5-fold cross-validation.
- Optimization metric: F1-score.

- logreg: best_cv_f1=0.4840, best_params={'penalty': 'l2', 'class_weight': None, 'C': 0.05}
- knn: best_cv_f1=0.4091, best_params={'weights': 'uniform', 'p': 2, 'n_neighbors': 3}
- rf: best_cv_f1=0.8022, best_params={'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': None, 'max_depth': None, 'class_weight': None}

## Threshold optimization

- For selected models, the decision threshold is optimized using training data only (out-of-fold probabilities).
- Variants containing `threshold` in the table reflect this post-processing step.

## Results (train vs test)

```
 model         variant split  accuracy  precision  recall     f1  roc_auc
logreg        baseline train    0.7776     0.3730  0.7758 0.5038   0.8455
logreg        baseline  test    0.7766     0.3650  0.7684 0.4949   0.8188
   knn        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   knn        baseline  test    0.8786     0.8889  0.1684 0.2832   0.8675
    rf        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
    rf        baseline  test    0.9280     0.9796  0.5053 0.6667   0.9223
logreg           tuned train    0.7757     0.3681  0.7552 0.4949   0.8346
logreg           tuned  test    0.7736     0.3627  0.7789 0.4950   0.8295
logreg threshold_tuned train    0.7997     0.3969  0.7242 0.5128   0.8346
logreg threshold_tuned  test    0.7961     0.3829  0.7053 0.4963   0.8295
   knn           tuned train    0.9224     0.9132  0.5155 0.6590   0.9688
   knn           tuned  test    0.8816     0.7222  0.2737 0.3969   0.7328
   knn threshold_tuned train    0.9029     0.5997  1.0000 0.7498   0.9688
   knn threshold_tuned  test    0.7841     0.3515  0.6105 0.4462   0.7328
    rf           tuned train    0.9786     0.9321  0.9201 0.9261   0.9978
    rf           tuned  test    0.9505     0.8444  0.8000 0.8216   0.9226
```

## Key observations

- Best tuned model on the test set (by F1): rf (F1=0.8216, precision=0.8444, recall=0.8000, accuracy=0.9505).
- Best model confusion matrix (test): TN=558, FP=14, FN=19, TP=76.
- knn baseline: train-test F1 gap = 0.7168.
- knn tuned: train-test F1 gap = 0.2620.
- logreg baseline: train-test F1 gap = 0.0089.
- logreg tuned: train-test F1 gap = -0.0001.
- rf baseline: train-test F1 gap = 0.3333.
- rf tuned: train-test F1 gap = 0.1044.

## Interpretation and diagnostics

- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).
- If accuracy is high but recall is low, the model may miss churners due to class imbalance.
- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.

## Connection to EDA (feature considerations)

- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).
- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).
- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.
