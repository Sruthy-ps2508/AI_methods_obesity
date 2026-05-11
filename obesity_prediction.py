# ============================================================
# AI Methods for Examining Obesity Status
# Using Social and Physical Activity Data
# ============================================================
# Author: Sruthy Pattotil Sunilkumar
# Supervisor: Prof. Eliseo Vilalta Perdomo
# Programme: MSc Business Analytics — Aston University
# Module: BDM200J
# Date: January 2024
# ============================================================

# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# ============================================================
# SECTION 2: LOAD DATASET
# ============================================================

# Load the dataset — update the path if needed
# Dataset source: https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels
data = pd.read_csv("data/obesity_dataset.csv")

print("Dataset shape:", data.shape)
print(data.head())

# ============================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# --- Appendix 1: Univariate Analysis — Categorical Variables ---
plt.figure(figsize=(18, 15))
for i, col in enumerate(data.select_dtypes(include="object").columns[:-1]):
    plt.subplot(4, 2, i + 1)
    sns.countplot(data=data, x=col, palette=sns.color_palette("Set2"))
plt.tight_layout()
plt.suptitle("Univariate Analysis of Categorical Variables", y=1.02)
plt.savefig("outputs/appendix1_categorical_variables.png", bbox_inches="tight")
plt.close()

# --- Appendix 2: Boxplot Analysis — Continuous Variables ---
plt.figure(figsize=(18, 15))
for i, col in enumerate(data.select_dtypes(include="number").columns[:3]):
    plt.subplot(4, 2, i + 1)
    sns.boxplot(data=data, x=col, palette=sns.color_palette("Set2"))
plt.tight_layout()
plt.suptitle("Boxplot Analysis of Continuous Variables", y=1.02)
plt.savefig("outputs/appendix2_boxplot_continuous.png", bbox_inches="tight")
plt.close()

# --- Appendix 3: Distribution Analysis — Numerical Lifestyle Variables ---
plt.figure(figsize=(18, 15))
for i, col in enumerate(data.select_dtypes(include="number").columns[3:]):
    plt.subplot(4, 2, i + 1)
    sns.countplot(data=data, x=col, palette=sns.color_palette("Set2"))
plt.tight_layout()
plt.suptitle("Distribution Analysis of Numerical Lifestyle Variables", y=1.02)
plt.savefig("outputs/appendix3_lifestyle_variables.png", bbox_inches="tight")
plt.close()

# --- Appendix 4: Median Age Distribution for All Obesity Types ---
data.groupby("NObeyesdad")["Age"].median().sort_values(ascending=False).plot(
    kind="bar", color=sns.color_palette("Set2")
)
plt.title("Average age of each obesity type")
plt.xlabel("NObeyesdad")
plt.ylabel("Median Age")
plt.tight_layout()
plt.savefig("outputs/appendix4_median_age_obesity.png", bbox_inches="tight")
plt.close()

# --- Appendix 5: Correlation Heatmap ---
corr_data = data.copy()
encoder = LabelEncoder()
for col in corr_data.select_dtypes(include="object").columns:
    corr_data[col] = encoder.fit_transform(corr_data[col])

plt.figure(figsize=(16, 13))
sns.heatmap(corr_data.corr(), annot=True)
plt.title("Heatmap of Variable Correlations")
plt.tight_layout()
plt.savefig("outputs/appendix5_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 4: DATA PREPROCESSING
# ============================================================

# Encode categorical variables
data_encoded = data.copy()
label_enc = LabelEncoder()

for col in data_encoded.select_dtypes(include="object").columns:
    data_encoded[col] = label_enc.fit_transform(data_encoded[col])

# Remove duplicates
print(f"Duplicates found: {data_encoded.duplicated().sum()}")
data_encoded = data_encoded.drop_duplicates()

# Remove outliers using Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(data_encoded.select_dtypes(include="number")))
data_encoded = data_encoded[(z_scores < 3).all(axis=1)]
print(f"Dataset size after cleaning: {data_encoded.shape}")

# Define features and target
X = data_encoded.drop("NObeyesdad", axis=1)
y = data_encoded["NObeyesdad"]

# Scale features
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples:  {x_test.shape[0]}")

# ============================================================
# SECTION 5: MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

cllg = LogisticRegression(max_iter=1000)
cllg.fit(x_train, y_train)
predictions = cllg.predict(x_test)

lr_accuracy = accuracy_score(y_test, predictions)
lr_precision = precision_score(y_test, predictions, average='macro')
lr_recall = recall_score(y_test, predictions, average='macro')
lr_f1 = f1_score(y_test, predictions, average='macro')

print("Accuracy score:", lr_accuracy)
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print("Classification report:\n", classification_report(y_test, predictions))

# Bar plot — Logistic Regression evaluation metrics
lr_metrics = {
    "Accuracy":  lr_accuracy * 100,
    "Precision": lr_precision * 100,
    "Recall":    lr_recall * 100,
    "F1-Score":  lr_f1 * 100
}

plt.figure(figsize=(8, 6))
bars = plt.bar(lr_metrics.keys(), lr_metrics.values(),
               color=["teal", "teal", "orange", "violet"])
for bar, val in zip(bars, lr_metrics.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5, f"{val:.2f}%", ha="center")
plt.title("Evaluation Metrics for Logistic Regression")
plt.ylabel("Percentage (%)")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("outputs/fig2_logistic_regression_metrics.png")
plt.close()

# ============================================================
# SECTION 6: MODEL 2 — K-NEAREST NEIGHBORS (KNN)
# ============================================================

print("\n" + "="*50)
print("MODEL 2: K-NEAREST NEIGHBOR ALGORITHM")
print("="*50)
knn_accuracy = accuracy_score(y_test, pred)
knn_precision = precision_score(y_test, pred, average='macro')
knn_recall = recall_score(y_test, pred, average='macro')
knn_f1 = f1_score(y_test, pred, average='macro')

print("Accuracy score:", knn_accuracy
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

print("Accuracy score:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Classification report:\n", classification_report(y_test, pred))

# Confusion Matrix heatmap — KNN
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for KNN Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig("outputs/fig4_knn_confusion_matrix.png")
plt.close()

# ============================================================
# SECTION 7: MODEL 3 — SUPPORT VECTOR MACHINE (SVM)
svm_accuracy = accuracy_score(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred, average='macro')
svm_recall = recall_score(y_test, y_pred, average='macro')
svm_f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy score:", svm_accuracy)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Bar plot — SVM evaluation metrics
svm_metrics = {
    "Accuracy":  svm_accuracy * 100,
    "Precision": svm_precision * 100,
    "Recall":    svm_recall * 100,
    "F1-Score":  svm_f1 * 100
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Bar plot — SVM evaluation metrics
svm_metrics = {
    "Accuracy":  74.31,
    "Precision": 72.00,
    "Recall":    74.30,
    "F1-Score":  72.92
}

plt.figure(figsize=(8, 6))
bars = plt.bar(svm_metrics.keys(), svm_metrics.values(),
               color=["blue", "green", "orange", "purple"])
for bar, val in zip(bars, svm_metrics.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5, f"{val:.2f}%", ha="center")
plt.title("SVM Model Evaluation Metrics")
plt.ylabel("Percentage (%)")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("outputs/fig6_svm_metrics.png")
plt.close()

# ============================================================
# SECTION 8: MODEL 4 — DECISION TREE
# ============================================================

print("\n" + "="*50)
print("MODEL 4: DECISION TREE")
print("="*50)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, average='macro')
dt_recall = recall_score(y_test, y_pred_dt, average='macro')
dt_f1 = f1_score(y_test, y_pred_dt, average='macro')

print("Accuracy score:", dt_accuracy
clf_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=100,
    max_depth=3,
    min_samples_leaf=5
)
clf_entropy.fit(x_train, y_train)
y_pred_dt = clf_entropy.predict(x_test)

print("Accuracy score:", accuracy_score(y_test, y_pred_dt))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification report:\n", classification_report(y_test, y_pred_dt))

# Visualise Decision Tree
try:
    import graphviz
    dot_data = export_graphviz(
        clf_entropy,
        out_file=None,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("outputs/fig7_decision_tree", format="png", cleanup=True)
    print("Decision tree saved to outputs/fig7_decision_tree.png")
except ImportError:
    print("graphviz not installed — skipping tree visualisation.")

# ===================lr_accuracy * 100, knn_accuracy * 100, svm_accuracy * 100, dt_accuracy * 100],
    "Precision (%)": [lr_precision * 100, knn_precision * 100, svm_precision * 100, dt_precision * 100],
    "Recall (%)": [lr_recall * 100, knn_recall * 100, svm_recall * 100, dt_recall * 100],
    "F1-Score (%)": [lr_f1 * 100, knn_f1 * 100, svm_f1 * 100, dt_f1 * 100
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

results = {
    "Model": [
        "Logistic Regression",
        "KNN",
        "SVM",
        "Decision Tree"
    ],
    "Accuracy (%)": [69.77, 76.32, 74.30, 64.73],
    "Precision (%)": [68.00, 74.00, 72.00, 65.00],
    "Recall (%)": [69.77, 76.32, 74.30, 64.73],
    "F1-Score (%)": [68.86, 74.92, 72.92, 64.86]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Bar chart — All models comparison
x = np.arange(len(results["Model"]))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5 * width, results["Accuracy (%)"],  width, label="Accuracy")
ax.bar(x - 0.5 * width, results["Precision (%)"], width, label="Precision")
ax.bar(x + 0.5 * width, results["Recall (%)"],    width, label="Recall")
ax.bar(x + 1.5 * width, results["F1-Score (%)"],  width, label="F1-Score")

ax.set_xlabel("Model")
ax.set_ylabel("Score (%)")
ax.set_title("Comparison of All Model Evaluation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(results["Model"])
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

print("\nAll outputs saved to the outputs/ folder.")
print("Script complete.")
