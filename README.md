# Malignancy Detection in Cellular Biopsies

Project for the IDEAS (ISI Kolkata) Autumn Internship 2026. Mentor: Dr. Chandan Biswas.

## What it does
Predicts whether a breast tumour is malignant or benign from measurements of cell nuclei, using the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn (569 samples, 30 numeric features such as radius, texture, area and concavity, no missing values). The goal was to go through the full workflow, from data checks to final evaluation. About 37% of samples are malignant, a mild imbalance, so no rebalancing was used.

## Steps in the notebook
- Data checks (missing values, class balance, correlation heatmap: 21 feature pairs above 0.90, mostly size measures like radius, perimeter and area)
- Scaling, one new feature (mean area / worst area), stratified 80/20 split
- Outlier removal with Isolation Forest (23 training rows removed) and PCA (31 features to 10 components, 95% variance)
- Logistic Regression baseline with 5-fold cross-validation, and a tuned MLP neural network (grid search)
- ROC-AUC comparison, a custom penalty score, and feature importance

Outlier removal and PCA were fitted on the training data only. The test set was left untouched.

## Results
- Logistic Regression: about 98.6% cross-validation accuracy, 98% test accuracy (one false negative, one false positive), ROC-AUC 0.997
- Tuned MLP (two hidden layers, 32 and 16 nodes, learning rate 0.001): ROC-AUC 0.996
- Custom penalty (10 per false negative, 1 per false positive): 11 for Logistic Regression, 12 for the MLP
- Top features: mean-to-worst area ratio, worst texture, mean concave points

## Conclusion
Both models did very well, and the simpler Logistic Regression worked about as well as the neural network and is easier to explain, which matters in medicine. Limits: small dataset and a single train/test split. Future work: try Random Forest or SVM and adjust the decision threshold to reduce false negatives.

## How to run
Open the notebook in Google Colab and click Runtime > Run all.
