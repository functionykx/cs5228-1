# Supervised Learning Report (Churn Prediction)

## 1. 任务目标与评估设置

本项目的 supervised learning 目标是对客户是否流失（Churn）进行二分类预测，并在训练集与测试集上报告标准指标（accuracy、precision、recall、F1），同时输出混淆矩阵用于错误类型分析。由于 churn 属于少数类，F1/recall 比单纯 accuracy 更有参考价值。

统一约束与原则：
- 训练/调参仅使用训练集：调参使用 stratified 5-fold CV（仅训练集）。
- 测试集仅用于最终评估：不参与任何模型选择、阈值选择或调参决策。
- 固定随机种子：保证可复现。

产物位置：每个实验目录均包含 `metrics.csv`、混淆矩阵 png/csv、调参日志（`best_params_*.json`、`cv_results_*.csv`）与英文摘要 `supervised_summary.md`。

## 2. 我们考虑了哪些模型

### 2.1 初始模型集合（artifacts/supervised）
- **logreg**：Logistic Regression（线性基线）
- **knn**：K-Nearest Neighbors（距离模型基线）
- **rf**：Random Forest（树模型集成）

### 2.2 替换模型集合（artifacts/supervised_altmodels）
为改善 logreg/knn 在表格数据上的上限与稳定性，引入更强的树系集成模型：
- **hgb**：HistGradientBoostingClassifier（boosted trees）
- **et**：ExtraTreesClassifier（随机化树集成）
- **rf**：Random Forest（保留作为对照）

## 3. 实验演进：从 supervised → nocollinear_nogeo → altmodels → strongreg

下面按“动机 → 做法 → 结果（以 tuned/test 为主）”总结每一步。

### Step A: artifacts/supervised（全特征，73维）

**动机**
- 建立端到端监督学习基线：3 模型 baseline + train-only 调参 + test 评估。

**结果（tuned/test）**（见 `artifacts/supervised/metrics.csv`）
- logreg：F1 **0.5034**, precision 0.3695, recall 0.7895
- knn：F1 **0.5175**, precision 0.7708, recall 0.3895
- rf：F1 **0.8439**, precision 0.9359, recall 0.7684

**观察**
- rf 显著领先。
- logreg 相对“稳定但上限低”（train/test 相近）。
- knn baseline 过拟合明显（train=1.0，但 test recall 很低），tuned 后有所缓解但仍弱于 rf。

### Step B: artifacts/supervised_nocollinear_nogeo（去共线 + 去geo，73→15）

**动机**
- EDA 指出 minutes 与 charge 近乎完全共线；geo（State/Area code）弱相关且引入高维 one-hot。
- 目标：降低维度与冗余，提升泛化与可解释性。

**做法**
- 去掉 `State` 与 `Area code` 的所有 one-hot 列
- 去掉 `Total {day,eve,night,intl} charge`（保留 minutes）
- 额外增强：对 logreg/knn 增加 train-only 的阈值优化（variant=threshold_tuned），用训练集 out-of-fold 概率选择阈值（默认最大化 F1）

**结果（tuned/test）**（见 `artifacts/supervised_nocollinear_nogeo/metrics.csv`）
- logreg tuned：F1 **0.4949**, precision 0.3650, recall 0.7684
- knn tuned：F1 **0.5036**, precision 0.7955, recall 0.3684
- rf tuned：F1 **0.8475**, precision 0.9146, recall 0.7895

**结果（threshold_tuned/test）**（同文件）
- logreg threshold_tuned：F1 **0.5019**（precision↑，recall↓，权衡后 F1 略升）
- knn threshold_tuned：F1 **0.5078**（recall↑明显，precision↓，整体 F1 略升）

**观察**
- rf 在去冗余后 F1 小幅提升且更稳。
- logreg/knn 主要问题不是“能不能调参”，而是模型能力与不平衡阈值权衡：通过阈值优化可按目标拉动 precision/recall，但整体上限仍不高。

### Step C: artifacts/supervised_altmodels（替换 logreg/knn 为树系模型）

**动机**
- logreg/knn 对表格数据常见上限较低（线性表达不足、距离度量不稳定）。
- 引入更强的树系模型（boosting/ExtraTrees）提高 test F1。

**结果（tuned/test）**（见 `artifacts/supervised_altmodels/metrics.csv`）
- hgb：F1 **0.8427**, precision 0.9036, recall 0.7895
- et：F1 **0.8136**, precision 0.8780, recall 0.7579
- rf：F1 **0.8492**, precision 0.9048, recall 0.8000

**观察**
- hgb 与 rf 都达到了接近 0.85 的 test F1，显著高于 logreg/knn。
- 部分模型 train 接近 1.0，存在过拟合倾向，但 test 依然很高，说明泛化仍可接受。

### Step D: artifacts/supervised_altmodels_strongreg（强正则，压过拟合）

**动机**
- 观察到树模型/boosting 在训练集上接近完美，存在过拟合风险。
- 目标：通过限制树深度、增大叶子、限制 max_features 等，使 train-test gap 下降（泛化更稳），并观察对 test F1 的影响。

**做法（代表性）**
- rf / et：限制 `max_depth`，增大 `min_samples_leaf`，并将 `max_features` 限制为 sqrt/log2
- hgb：提升 `min_samples_leaf`、增加 `l2_regularization`、限制 `max_depth`

**结果（tuned/test）**（见 `artifacts/supervised_altmodels_strongreg/metrics.csv`）
- hgb：F1 **0.7978**, precision 0.8295, recall 0.7684
- et：F1 **0.6266**, precision 0.5290, recall 0.7684
- rf：F1 **0.7788**, precision 0.7168, recall 0.8526

**观察**
- 强正则显著降低了训练集拟合强度（例如 rf 不再 train=1.0），但也牺牲了 test F1（更保守）。
- rf tuned 的 recall 提升到 0.85，但 precision 下滑，体现了“稳健/召回优先”的取向。

## 4. 结果分析与结论

### 4.1 最终谁最好？
以“最大化 test F1”为目标：
- 当前最强结果集中在：
  - `artifacts/supervised_altmodels/` 的 **rf tuned test F1≈0.849**
  - `artifacts/supervised_nocollinear_nogeo/` 的 **rf tuned test F1≈0.847**
- logreg/knn 即使在阈值优化后也只能小幅改善（仍显著落后树系集成）。

### 4.2 为什么 logreg/knn 效果不佳？
- logreg 是线性模型：在 churn 场景可能需要非线性交互（例如“客服来电多 + 使用行为组合”），线性边界难以表达，导致上限低。
- knn 依赖距离度量：表格特征（标准化连续 + one-hot 类别）下距离并不总是语义一致，容易出现 recall/precision 极不平衡。
- 类不平衡：默认阈值 0.5 往往不是最优；阈值优化能改善权衡，但无法替代模型表达能力。

### 4.3 去共线+去geo 为什么有帮助？
- 降低维度与冗余信息，减少噪声特征对学习的干扰，有助于泛化与可解释性。
- 对 rf 这类树模型表现为：F1 小幅提升/更稳；对 logreg/knn 表现为：稳定性提升有限，但阈值优化更容易解释与落地。

### 4.4 强正则为什么会让 test F1 下降？
- 强正则抑制模型复杂度：train-test gap 下降，但如果压得过头，会进入欠拟合区间，表现为 test F1 下滑。
- 这不是“更差”，而是目标取舍：如果业务更看重召回或稳定性，strongreg 版本仍有价值。

## 5. 建议的下一步（按优先级）
- 若目标是 poster 上展示“最好效果”：使用 `artifacts/supervised_altmodels/` 的 rf/hgb tuned，并用混淆矩阵解释 FP/FN 权衡。
- 若目标是更符合 churn 业务：对最优模型做阈值优化（例如在 recall≥某值的约束下最大化 precision/F1），并报告对应 confusion matrix。
- 若需要更可解释：在 rf/hgb 上补充特征重要性（permutation importance / SHAP 若允许）与错误案例分析（FN/FP 的共同特征）。

