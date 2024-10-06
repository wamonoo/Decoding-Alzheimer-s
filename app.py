import streamlit as st
import joblib
import numpy as np
#load the model
model = joblib.load('alzeimer_predictor_model.pkl')

#app layout

#st.image("header.jpg")

st.title("Decoding Alzheimer's")
st.write('''
### About this App
This application assesses the probability of a patient being diagnosed with Alzheimer's Disease based on essential clinical characteristics. 
The prediction is powered by a binary classification model, trained on a public dataset [1] comprising detailed health records and Alzheimer's 
diagnoses for 2,149 individuals.''')

# Get user input
st.write("""### Enter Patient Data
""")
f_a_score = st.text_input("Functional Assessment Score (0-10)", value="0.0")
adl_score = st.text_input(label = "Activities of Daily Living (ADL) Score (0-10)", value = "0.0")
mmse = st.text_input("Mini-Mental State Exam (MMSE) Score (0-30)", value="0.0")

memory_complaints = st.radio("Memory Complaints", ('No', 'Yes'))
behavioral_problems = st.radio("Behavioral Problems", ('No', 'Yes'))

#encode categorical variables
memory_complaints = 1 if memory_complaints =='Yes' else 0
behavioral_problems = 1 if behavioral_problems == 'Yes' else 0 

#run prediction
if st.button('Get Prediction'):
    try:
        f_a_score = float(f_a_score)
        adl_score = float(adl_score)
        mmse = float(mmse)
        # Check if the values are within the predefined ranges
        if not (0 <= f_a_score <= 10):
            st.error("Functional Assessment Score must be between 0 and 10.")
        elif not (0 <= adl_score <= 10):
            st.error("ADL Score must be between 0 and 10.")
        elif not (0 <= mmse <= 30):
            st.error("MMSE Score must be between 0 and 30.")
        else:
            input_data = np.array([[mmse, f_a_score, memory_complaints, behavioral_problems, adl_score]])    
          
            prediction = model.predict(input_data)
            st.write(f"Prediction: {'Alzheimer’s Likely' if prediction[0] == 1 else 'Alzheimer’s not Likely'}")
            prediction_probability = model.predict_proba(input_data)
            prob = prediction_probability.tolist()[0]
            st.write(f"Prediction Confidence: Alzheimer’s {prob[1]* 100:.2f}%  ------ No Alzheimer’s {prob[0]* 100:.2f}%")
    except ValueError:
        st.error("Please ensure that all inputs are within the ranges stated")

st.write('''
---
### Model Details
The prediction model is a logistic regression ensemble model built from stacking 5 algorithms namely RandomForest,  XG Boost, Light GBM, CatBoost, and the GradientBoosting classifier.
The baseline models are tuned and the best parameters fed into the stacking classifier whose hyper-parameters of C(inverse of regularization strength) and penalty (l1, l2) are futher tuned to help optimize the performance of the final estimator.

The model achieves an accuracy score of 96.05% and cross validation score of 95.01% across 5 folds.

### Classification Results

#### Class 0 (No Diagnosis):

- **Precision (0.96):** Of all instances predicted as class 0, 96% were correct. Thus the model has a low false positive rate for this class.
- **Recall (0.98):** Of all actual class 0 instances, 98% were correctly predicted. This indicates that the model is effective at identifying instances that belong to this class.
- **F1-Score (0.97):** This score is the harmonic mean of precision and recall, providing a balance between the two metrics for class 0. A score of 0.97 is very good.

#### Class 1 (Alzheimer's Diagnosis):

- **Precision (0.97):** Of all instances predicted as class 1, 97% were correct. This indicates a very low rate of false positives for class 1.
- **Recall (0.92):** Of all actual class 1 instances, 92% were correctly predicted. While still high, it suggests there is a slightly higher rate of false negatives compared to class 0.
- **F1-Score (0.94):** This score indicates a strong balance between precision and recall for class 1.

The following 5 features from the dataset are used for the predictions. It was determined through feature selection that they were the most immportant. 

- **Functional Assessment Score (FA)**: Between 0 and 10. Lower scores indicate greater impairment.
- **Activities of Daily Living (ADL) Score**: Between 0 and 10. Lower scores indicate greater impairment.
- **Mini-Mental State Examination (MMSE) Score**: Between 0 and 30. Lower scores indicate cognitive impairment.
- **Memory Complaints**: Indicates if the patient reports memory issues (Yes/No)
- **Behavioral Problems**: Indicates if the patient has behavioral issues (Yes/No)

##### Dataset Citation
Rabie El Kharoua. Alzheimer's Disease Dataset. Kaggle, 2024, https://doi.org/10.34740/KAGGLE/DSV/8668279.

''')