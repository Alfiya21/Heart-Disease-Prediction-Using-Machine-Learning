# Heart Disease Prediction Using Machine Learning

## Overview
This project implements an end-to-end machine learning pipeline to predict the presence of heart disease using clinical and demographic patient data. The system emphasizes medical reliability, interpretability, and reproducibility.



## Problem Statement
Build a classification model that predicts whether a patient has heart disease based on structured medical attributes, supporting early diagnosis and clinical decision-making.



## Dataset
The dataset contains patient health indicators such as age, cholesterol, blood pressure, chest pain type, ECG results, and more.

**Target:**
- 0 → No heart disease
- 1 → Heart disease detected


## System Architecture
The project follows a layered ML architecture:
- Data Ingestion
- Data Preprocessing
- Feature Engineering
- Model Training
- Evaluation & Explainability
- Model Persistence
- Prediction Output



## Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The best model is selected based on Recall and ROC-AUC.


## Evaluation Metrics
- Accuracy
- Precision
- Recall (priority)
- F1 Score
- ROC-AUC
- Confusion Matrix



## Explainability
Model interpretability is achieved using:
- Feature Importance
- Permutation Importance
- Partial Dependence Plots

These techniques help understand key risk factors influencing predictions.


## Project Structure
heart-disease-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_data_analysis.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training_and_evaluation.ipynb
│ └── 04_model_explainability_no_shap.ipynb
│
├── models/
│ └── best_model.pkl
│
├── src/
│ ├── preprocess.py
│ ├── train_model.py
│ └── evaluate.py
│
├── run_pipeline.py
├── requirements.txt
└── README.md


## How to Run
1. Clone the repository
2. Install dependencies:
pip install -r requirements.txt

markdown
Copy code
3. Run the pipeline:
python run_pipeline.py


## Results
The final model demonstrates strong recall and ROC-AUC, making it suitable for medical decision support systems.


## Future Improvements
- Model deployment via web application
- Integration with hospital databases
- Advanced explainability using SHAP
- Real-time prediction APIs


## Author
**Alfiya Mulla**  
Data Science & Machine Learning Enthusiast

## Project Structure

