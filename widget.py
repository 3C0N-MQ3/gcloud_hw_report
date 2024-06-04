# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %% [markdown]
"""
<div style="text-align: left;">
    <h1>Stroke Prediction Widget</h1>
    <h4>Applications of Cloud Computing and Big Data - ECON 446</h3>
    <div style="padding: 20px 0;">
        <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
        <p><em>Bella Rakhlina</em><br>
        <em>Lora Yovcheva</em><br>
        <em>Mauricio Vargas-Estrada</em><br>
        <br>Master Of Quantitative Economics<br>
        University of California - Los Angeles</p>
        <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
    </div>
</div>
"""
# %%
import os
import requests
import joblib
import ipywidgets as widgets

from io import BytesIO
from flask import Flask, request, jsonify
from IPython.display import display, clear_output
from google.cloud import storage
# %%
app = Flask(__name__)
# %%
# Function to send POST request to external URL
def send_prediction_request(age, gender, heart_disease, avg_glucose_level, bmi, smoking_status):
    url = "https://us-central1-spring-cloud-econ-446.cloudfunctions.net/stroke_function"

    # Map gender and smoking status to their integer values
    gender_mapping = {'Male': 1, 'Female': 0}
    smoking_status_mapping = {
        'never smoked': 1,
        'unknown': 2,
        'formerly smoked': 3,
        'smokes': 4
    }

    gender = gender_mapping[gender]
    smoking_status = smoking_status_mapping[smoking_status]

    data = {
        "age": age,
        "gender": gender,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    r = requests.post(url, json=data)
    return r.content.decode('utf-8')
# %%
@app.route('/predict', methods=['POST'])
def stroke_presence():
    print("Models")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = load_scikit_model("stroke_NN.sav")
            preproc = load_scikit_model("stroke_scaler.sav")
            print("Models Loaded!")

            # Convert request to request dictionary
            dictionary = request.get_json()
            print(dictionary)

            # Extracting variables from the dictionary
            required_keys = ['age', 'gender', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
            missing_keys = [key for key in required_keys if key not in dictionary]
            if missing_keys:
                raise ValueError(f"Missing required parameter(s): {', '.join(missing_keys)}")

            age = float(dictionary['age'])
            gender = dictionary['gender']
            if gender == 'Male':
                gender = 0
            elif gender == 'Female':
                gender = 1
            else:
                raise ValueError("Invalid gender value")

            heart_disease = int(dictionary['heart_disease'])
            avg_glucose_level = float(dictionary['avg_glucose_level'])
            bmi = float(dictionary['bmi'])
            smoking_status = dictionary['smoking_status']
            if smoking_status == 'never smoked':
                smoking_status = 1
            elif smoking_status == 'unknown':
                smoking_status = 2
            elif smoking_status == 'formerly smoked':
                smoking_status = 3
            elif smoking_status == 'smokes':
                smoking_status = 4
            else:
                raise ValueError("Invalid smoking status value")

            # Preprocess and make predictions
            X = preproc.transform([[age, gender, heart_disease, avg_glucose_level, bmi, smoking_status]])
            predictions = model.predict(X)[0]
            probability = str(round(model.predict_proba(X)[0][1] * 100, 2)) + "%"
            print("Probabilities Calculated")
            print(predictions)
            print(probability)

            return jsonify({
                "prediction": int(predictions),
                "status": 200,
                "prob_of_stroke": probability
            })
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})
# %%
# Creating input widgets for the stroke prediction
age_input = widgets.FloatText(value=0, description='Age:')
gender_input = widgets.Dropdown(options=['Female', 'Male'], description='Gender:')
heart_disease_input = widgets.Checkbox(value=False, description='Heart Disease:')
avg_glucose_level_input = widgets.FloatText(value=0, description='Avg Glucose Level:')
bmi_input = widgets.FloatText(value=0, description='BMI:')
smoking_status_input = widgets.Dropdown(options=['never smoked', 'unknown', 'formerly smoked', 'smokes'], description='Smoking Status:')

# Button to submit prediction
predict_button = widgets.Button(description="Predict Stroke")
output_area = widgets.Output()

def predict_stroke(b):
    with output_area:
        clear_output()
        try:
            # Collect input values
            age = age_input.value
            gender = gender_input.value
            heart_disease = int(heart_disease_input.value)
            avg_glucose_level = avg_glucose_level_input.value
            bmi = bmi_input.value
            smoking_status = smoking_status_input.value

            # Send the request to the external URL
            response = send_prediction_request(age, gender, heart_disease, avg_glucose_level, bmi, smoking_status)
            print(f"Response from external service: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

predict_button.on_click(predict_stroke)
# %% [markdown]
"""
<div style="text-align: center; background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #4A90E2; font-family: 'Arial', sans-serif;">Stroke Prediction Application</h1>
    <p style="font-size: 18px; color: #333; font-family: 'Arial', sans-serif;">Welcome to the Stroke Prediction App. Use the form below to predict the probability of stroke based on the input parameters.</p>
</div>
"""
# %% [markdown]
"""
<div style="text-align: left; background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
    <h2 style="color: #4A90E2; font-family: 'Arial', sans-serif;">Instructions</h2>
    <ul style="font-size: 16px; color: #333; font-family: 'Arial', sans-serif;">
        <li><strong>Age:</strong> Your age in years.</li>
        <li><strong>Gender:</strong> Your gender (Male or Female).</li>
        <li><strong>Heart Disease:</strong> Check if you have any heart disease.</li>
        <li><strong>Avg Glucose Level:</strong> Your average glucose level (mg/dL).</li>
        <li><strong>BMI:</strong> Your Body Mass Index (kg/m²).</li>
        <li><strong>Smoking Status:</strong> Your smoking status (never smoked, unknown, formerly smoked, smokes).</li>
    </ul>
    <p style="font-size: 16px; color: #333; font-family: 'Arial', sans-serif;">
        Click on the "Predict Stroke" button to get the prediction.
    </p>
</div>
"""
# %%
display(age_input, gender_input, heart_disease_input, avg_glucose_level_input, bmi_input, smoking_status_input, predict_button, output_area)
