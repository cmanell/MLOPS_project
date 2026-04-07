# MLOps Project – Credit Default Prediction

## 📌 Project Overview

This project was developed as part of a Sorbonne University MLOps course.  
The objective is to build an end-to-end machine learning pipeline to predict the probability of credit default in a retail banking context.

Default prediction is a key problem in finance, helping institutions manage risk, optimize lending decisions, and maintain financial stability.

## 🎯 Objectives

- Build and evaluate multiple classification models  
- Handle imbalanced data (SMOTE, class weighting)  
- Track experiments using MLflow  
- Select the best-performing model  
- Serialize and deploy the final model  

## 📊 Dataset

The dataset includes financial and behavioral client features:

- credit_lines_outstanding  
- loan_amt_outstanding  
- total_debt_outstanding  
- income  
- years_employed  
- fico_score  

**Target:**

- default (0 = no default, 1 = default)

**Class distribution:**

- ~81% non-default  
- ~19% default  

## ⚙️ Models Evaluated

Several models were trained and compared:

- Logistic Regression (baseline)  
- Logistic Regression with class balancing  
- Logistic Regression with SMOTE  
- Decision Tree  
- Random Forest  

All experiments were tracked using MLflow, including:

- Model parameters  
- Metrics:  
  - Recall  
  - F1-score  
  - ROC-AUC  
- Model artifacts  

## 🏆 Model Selection

The final model was selected based on test performance with a focus on:

- F1-score  
- ROC-AUC  
- Recall (critical for detecting defaults)  

**Selected model:**

- Logistic Regression with SMOTE  

## 💾 Model Serialization

The final model is saved as:
models/best_model.joblib


The model is integrated into a preprocessing pipeline (scaling + model) to ensure consistency in production.

## 🚀 Deployment

The application is deployed using:

- Streamlit (frontend interface)  
- Docker (containerization)  
- AWS ECS (Fargate) (cloud deployment)  
- AZURE
## 🌍 Live application
https://credit-default-app.victoriouscliff-1f833c45.francecentral.azurecontainerapps.io/

http://15.236.37.169:8501/

## 📂 Project Repository

GitHub:

https://github.com/cmanell/MLOPS_project

## ⚠️ Important Notes

- The model expects scaled input data (handled in the pipeline)  
- SMOTE is used only during training, not in inference  
- High performance may indicate:  
  - strong feature correlation  
  - or a relatively simple classification problem  

## Conclusion

This project demonstrates a complete MLOps workflow, including:

- Experiment tracking with MLflow  
- Model comparison and selection  
- Reproducible pipeline  
- Model serialization  
- Cloud deployment with Docker and AWS  
