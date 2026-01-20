# gradio app

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("diabetes_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_diabetes(gender, age, hypertension, heart_disease,
                     smoking_history, bmi, hba1c_level, blood_glucose_level):
    
    # Pack inputs into a DataFrame
    input_df = pd.DataFrame([[
        gender, age, hypertension, heart_disease,
        smoking_history, bmi, hba1c_level, blood_glucose_level
    ]],
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]
    )
    
    # Predict (0 = no diabetes, 1 = diabetes)
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  
    
    if pred == 1:
        return f"Diabetes: YES (risk probability: {proba:.2f})"
    else:
        return f"Diabetes: NO (risk probability: {proba:.2f})"

# 3. The App Interface
inputs = [
    gr.Radio(["Male", "Female", "Other"], label="Gender"),
    gr.Number(label="Age (years)", value=30),
    gr.Radio([0, 1], label="Hypertension (1 = Yes, 0 = No)", value=0),
    gr.Radio([0, 1], label="Heart Disease (1 = Yes, 0 = No)", value=0),
    gr.Dropdown(
        ["never", "No Info", "current", "former", "ever", "not current"],
        label="Smoking History",
        value="never"
    ),
    gr.Slider(10, 50, step=0.1, label="BMI", value=25.0),
    gr.Slider(3.5, 9.0, step=0.1, label="HbA1c Level", value=5.5),
    gr.Slider(50, 300, step=1, label="Blood Glucose Level", value=120),
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Risk Predictor (Gradient Boosting)",
    description="Enter patient details to estimate diabetes risk using a trained Gradient Boosting model."
)

if __name__ == "__main__":
    app.launch(share=True)
   
