# AI Methods for Examining Obesity Status Using Social and Physical Activity Data

**Author:** Sruthy Pattotil Sunilkumar  
**Supervisor:** Prof. Eliseo Vilalta Perdomo  
**Programme:** MSc Business Analytics — Aston University  
**Module:** BDM200J – MSc Business Analytics Business Project  
**Date:** January 2024

---

## Overview

This project investigates the use of Artificial Intelligence (AI) methods to assess and predict an individual's obesity status based on their social behaviours and physical activity levels. Four machine learning classification models were trained, tested, and compared on a publicly available obesity dataset sourced from Kaggle.

---

## Research Objectives

- Identify significant factors influencing obesity (eating patterns, physical activity, water intake, etc.)
- Build AI-based predictive models to classify individuals into obesity categories
- Evaluate and compare model performance using standard classification metrics
- Provide actionable insights for public health interventions and lifestyle modifications

---

## Dataset

- **Source:** Kaggle
- **Records (after cleaning):** 1,981 entries
- **Features:** 17 variables
- **Target Variable:** `NObeyesdad` — Obesity Level (multi-class)

### Variables

| Variable | Type | Description |
|---|---|---|
| Gender | Categorical | Person's gender |
| Age | Continuous | Person's age |
| Height | Continuous | Height in metres |
| Weight | Continuous | Weight in kilograms |
| Family History of Overweight | Binary | Family history of being overweight |
| FAVC | Binary | Regular intake of high-calorie foods |
| FCVC | Integer | Frequency of vegetable consumption |
| NCP | Integer | Number of meals per day |
| CAEC | Categorical | Food consumption between meals |
| SMOKE | Binary | Smoking habit |
| CH2O | Continuous | Daily water intake |
| SCC | Binary | Calorie monitoring |
| FAF | Continuous | Physical activity frequency (hours/week) |
| TUE | Integer | Daily technology usage time |
| CALC | Categorical | Alcohol consumption frequency |
| MTRANS | Categorical | Primary mode of transport |
| NObeyesdad | Categorical | **Target** — Obesity level |

---

## Data Preprocessing

1. **Duplicate removal** — 24 duplicate rows identified and removed
2. **Outlier handling** — Detected and removed using Z-scores and boxplots
3. **Categorical encoding** — One-hot encoding for nominal variables; label encoding for ordinal/target variables
4. **Feature scaling** — `MaxAbsScaler` applied to continuous features
5. **Train/test split** — 80% training / 20% testing

---

## Models Used

### 1. Logistic Regression (LR)
- Library: `sklearn.linear_model.LogisticRegression`
- Config: `max_iter=1000`
- Role: Baseline model; interpretable and computationally efficient

### 2. K-Nearest Neighbors (KNN)
- Library: `sklearn.neighbors.KNeighborsClassifier`
- Config: `n_neighbors=3`, Euclidean distance
- Role: Non-parametric; effective for non-linear relationships

### 3. Support Vector Machine (SVM)
- Library: `sklearn.svm.SVC`
- Config: `kernel='linear'`
- Role: High-dimensional classification; kernel trick for non-linear data

### 4. Decision Tree (DT)
- Library: `sklearn.tree.DecisionTreeClassifier`
- Config: `criterion='entropy'`, `max_depth=3`, `min_samples_leaf=5`
- Role: Interpretable; visualises feature-based decision paths

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **KNN** | **76.32%** | **77.00%** | **77.00%** | **76.00%** |
| SVM | 74.30% | 77.00% | 74.00% | 74.00% |
| Logistic Regression | 69.77% | 77.00% | 70.00% | 70.00% |
| Decision Tree | 67.28% | 62.00% | 64.00% | 61.00% |

**KNN achieved the highest accuracy (76.32%)** and the most balanced metrics across all obesity classes, making it the best-performing model for this dataset.