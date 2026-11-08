# Project Report – Predictive Analytics & AI Model Deployment

**Name:** <your name>  **Date:** <date>  **Programme:** Cognevance Technologies – AI & ML, Level 3

## 1. Objective
Build and deploy a complete machine-learning pipeline for a healthcare problem: predict whether a breast
tumour is malignant or benign from measurements of cell nuclei, compare several models, and serve the best one
through an API, a web page and a dashboard.

## 2. Data
Breast Cancer Wisconsin (Diagnostic): 569 patients, 30 numeric features (10 measurements × mean / standard
error / worst value), 212 malignant (37.3 %) and 357 benign. No missing values. Source: UCI Machine Learning
Repository (bundled with scikit-learn).
Exploratory charts: `outputs/figures/01_class_balance.png`, `02_correlation_heatmap.png`,
`03_feature_distributions.png`. Many size features (radius, perimeter, area) are almost perfectly correlated.

## 3. Preprocessing and feature engineering
1. Median imputation (needed when the API receives incomplete input).
2. Outlier clipping at the 1st and 99th percentiles.
3. Six engineered features (ratios and interactions, see `docs/ARCHITECTURE.md`).
4. Removal of redundant features (|correlation| > 0.95). Nine of the 36 features were dropped, leaving 27.
5. Standardisation.
All steps are fitted on the training data only and saved inside the model pipeline.

## 4. Models and evaluation method
Six models were trained: Logistic Regression, SVM (RBF), K-Nearest Neighbours, Random Forest, Gradient
Boosting and a Neural Network (scikit-learn MLP, 32-16 hidden units, used as the deep-learning model).
Each was tuned with a small grid search using 5-fold stratified cross-validation on the 80 % training split
(455 patients). The best model was chosen by cross-validated ROC-AUC. The 20 % test split (114 patients) was
used once, for reporting.

## 5. Results
| Model | CV ROC-AUC | Test accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| SVM (RBF) | 0.9965 | 0.974 | 1.000 | 0.929 | 0.963 | 0.996 |
| Logistic Regression | 0.9956 | 0.965 | 0.975 | 0.929 | 0.951 | 0.997 |
| Neural Network (MLP) | 0.9941 | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| Gradient Boosting | 0.9940 | 0.956 | 0.974 | 0.905 | 0.938 | 0.998 |
| K-Nearest Neighbours | 0.9935 | 0.930 | 0.947 | 0.857 | 0.900 | 0.991 |
| Random Forest | 0.9916 | 0.956 | 0.974 | 0.905 | 0.938 | 0.998 |

**Chosen model: SVM (RBF), C = 1.** On the test set it caught 39 of 42 malignant tumours (recall 0.929),
raised no false alarms (precision 1.000) and reached accuracy 97.4 % and ROC-AUC 0.996
(`outputs/figures/06_confusion_matrix.png`, `05_roc_curves.png`).

**Most influential inputs** (permutation importance, `07_feature_importance.png`): worst radius, mean concave
points, mean radius and worst concave points. Importance is measured on the raw inputs, so "worst radius" still
ranks first even though the correlation filter removed that column. Its information still reaches the model
through the engineered ratio `worst_mean_radius_ratio`, and through the correlated mean-radius feature.

## 6. Deployment
A Flask app serves the saved pipeline: `POST /predict` (JSON in, prediction, probability and risk level out),
a web form with 30 input fields and sample patients, and a dashboard with the metrics and charts. Invalid
input returns HTTP 400. The app and the pipeline are covered by 10 automated tests.

## 7. Limitations (please read)
- **Small dataset and small test set.** 569 patients is not a "large" dataset, and the test set has only 42
  malignant cases, so one patient changes recall by 2.4 points. The top models differ by less than this, so
  the ranking between SVM, Logistic Regression and the neural network is not statistically meaningful.
  Logistic Regression is simpler and nearly as good.
- **Missed cancers matter more than false alarms.** The chosen model missed 3 of 42 malignant tumours. In a
  real screening setting the decision threshold would be lowered to raise recall, at the cost of more false alarms.
- Hyper-parameters were tuned on the same cross-validation folds used to compare models, so the CV scores are
  slightly optimistic; the test scores are not affected.
- The data comes from one hospital and one imaging method, so results would not automatically transfer to
  other populations. The API fills missing measurements with medians, which lowers reliability for very
  incomplete input.
- This is an educational demo, not a medical device.

## 8. Conclusion and future work
A standard preprocessing and feature-engineering pipeline with a tuned SVM separates malignant from benign
tumours with about 97 % accuracy on unseen patients. Future work: a larger multi-source dataset, threshold tuning
for higher recall, probability calibration, model explanations (SHAP), and hosting the app in the cloud.
