# Decoding-Alzheimer-s
This case study deploys a python machine learning application on Streamlit Community Cloud. The app assesses the probability of a patient being diagnosed with Alzheimer's Disease based on essential clinical characteristics. The prediction is powered by a binary classification model, trained on a public dataset <sup>[1]</sup> comprising detailed health records and Alzheimer's diagnoses for 2,149 individuals.

## Deployed App
Link to deployed app: https://decoding-alzheimer-s-cpnyuuyhark72fxelgioww.streamlit.app

## Model Details
The prediction model is a logistic regression ensemble model built from stacking 5 algorithms namely RandomForest, XG Boost, Light GBM, CatBoost, and the GradientBoosting classifier. The baseline models are tuned and the best parameters fed into the stacking classifier whose hyper-parameters of C(inverse of regularization strength) and penalty (l1, l2) are futher tuned to help optimize the performance of the final estimator.
The model achieves an accuracy score of 96.05% and cross validation score of 95.01% across 5 folds.

Sample app screen-shot.
![image](https://github.com/user-attachments/assets/64fad5a0-bd7b-4407-8589-cb245cc319fc)

## Classification Results
#### Class 0 (No Diagnosis):

- **Precision (0.96):** Of all instances predicted as class 0, 96% were correct. Thus the model has a low false positive rate for this class.
- **Recall (0.98):** Of all actual class 0 instances, 98% were correctly predicted. This indicates that the model is effective at identifying instances that belong to this class.
- **F1-Score (0.97):** This score is the harmonic mean of precision and recall, providing a balance between the two metrics for class 0. A score of 0.97 is very good.

#### Class 1 (Alzheimer's Diagnosis):

- **Precision (0.97):** Of all instances predicted as class 1, 97% were correct. This indicates a very low rate of false positives for class 1.
- **Recall (0.92):** Of all actual class 1 instances, 92% were correctly predicted. While still high, it suggests there is a slightly higher rate of false negatives compared to class 0.
- **F1-Score (0.94):** This score indicates a strong balance between precision and recall for class 1.

## Features
The following 5 features from the dataset are used for the predictions. It was determined through feature selection that they were the most immportant.

- **Functional Assessment Score (FA):** Between 0 and 10.
- **Activities of Daily Living (ADL) Score:** Between 0 and 10.
- **Mini-Mental State Examination (MMSE) Score:** Between 0 and 30.
- **Memory Complaints:** Indicates if the patient reports memory issues (Yes/No)
- **Behavioral Problems:** Indicates if the patient has behavioral issues (Yes/No)

## Dataset Citation
Rabie El Kharoua. Alzheimer's Disease Dataset. Kaggle, 2024, https://doi.org/10.34740/KAGGLE/DSV/8668279.
