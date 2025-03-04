# Predictive Model for Hepatitis C Stages: Development and Performance Evaluation

**Author:** Akanksha Pandey  
**Date:** Fall 2023

---

## Introduction

This study aims to forecast the progression stages of Hepatitis C by differentiating between blood donors and Hepatitis C patients (including stages of the disease such as Hepatitis C, Fibrosis, and Cirrhosis). For this analysis, the stages of Cirrhosis, Fibrosis, and Hepatitis C have been combined into a single category named 'Hepatitis,' while the contrasting group is labeled as 'Donor'.

**Algorithms Used:** 
- Random Forest
- Penalized Logistic Regression
- Boosted Trees

**Feature Engineering and Transformation:** 
- Impute using median
- Turn nominal variables into dummies
- Normalize all predictors
- Balance target classes using the SMOTE algorithm

## Libraries

The following R libraries are used:
- `utils`
- `psych`
- `caret`
- `tidyverse`
- `skimr`
- `stringr`
- `themis`
- `vip`
- `probably`
- `ggplot2`
- `GGally`
- `corrplot`
- `randomForest`
- `pROC`
- `tidymodels`
- `ranger`
- `xgboost`

## Data Acquisition

The dataset includes laboratory test results of blood donors and Hepatitis C patients, along with demographic details like age. It consists of 615 observations and 14 variables and was sourced from the UCI Machine Learning Repository's 'HCV data' section.

### Data Loading

The data is downloaded, unzipped, and loaded from the UCI Machine Learning Repository.

### Encoding Categorical Variable

The target variable (diagnosis) is categorical with values ('0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'). The stages of Cirrhosis, Fibrosis, and Hepatitis C have been amalgamated into 'Hepatitis', and the contrasting group is labeled as 'Donor'.

## Data Exploration

### Bar Plot: Diagnosis vs. Count

A bar graph shows the relation between diagnosis and the number of people for each case. It highlights that the count for Donors is significantly higher than Hepatitis.

### Relationship between Diagnosis and Other Variables

Various plots are generated to show the relationship between diagnosis and other variables like Age, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, and PROT.

### Detection of Outliers

Box plots are used to detect outliers in continuous features. The count of outliers for each feature is summarized.

### Correlation Analysis

Correlation matrices and plots are used to analyze the relationships between different numerical features.

### Evaluation of Distribution

Histograms and density plots for each selected variable are generated to evaluate their distributions.

## Data Cleaning & Shaping

### Identification of Missing Values

The dataset is checked for missing values, and the count of missing values for each column is summarized.

### Data Imputation

Missing data is imputed using the median.

### Normalization

The Min-Max Normalization technique is applied to rescale the feature values to a fixed range, typically between 0 and 1.

## Model Construction

### Creation of Training & Validation Subsets

The dataset is split into training and validation subsets using a 70/30 split ratio.

### Dummy Encoding for Nominal Predictors

Nominal predictors are encoded into dummy variables, and the SMOTE algorithm is used to balance the target classes.

### Model A: Logistic Regression

A logistic regression model with L2 regularization is trained and tuned using a grid search with cross-validation. The best hyperparameters are selected based on ROC-AUC.

### Model B: Random Forest

A random forest classification model is trained and tuned. The best hyperparameters are selected based on ROC-AUC.

### Model C: Boosted Trees

A boosted tree classification model is trained and tuned using the `xgboost` engine. The best hyperparameters are selected based on ROC-AUC.

## Model Evaluation

### Comparison of Models and Interpretation

The ROC-AUC curves of all models are compared. The Boosted Trees model shows marginally superior performance compared to Random Forest, and both significantly outperform Logistic Regression.

## Model Tuning & Performance Improvement

### Construction of Heterogeneous Ensemble Model

An ensemble model combining predictions from logistic regression, random forest, and boosted trees is constructed. The ensemble model achieves high accuracy, kappa, and balanced accuracy, with excellent specificity and positive predictive value.

### Comparison of Ensemble to Individual Models

The ensemble model is compared to individual models. While it excels in specificity and positive predictive value, there is room for improvement in sensitivity.

### Failure Analysis

A failure analysis is conducted for all models to identify instances of false negatives and false positives.

## Reference

[UCI Machine Learning Repository - HCV Data](https://archive.ics.uci.edu/dataset/571/hcv+data)

---

This README provides an overview of the analysis, including the objective, data preparation, modeling techniques, evaluation, and results. The detailed R code for each step is included in the accompanying script.
