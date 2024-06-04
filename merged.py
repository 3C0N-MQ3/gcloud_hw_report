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
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
# %% [markdown]
# <div style="text-align: left;">
#     <h1>Cloud Functions and Storage</h1>
#     <h4>Applications of Cloud Computing and Big Data - ECON 446</h3>
#     <div style="padding: 20px 0;">
#         <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
#         <p><em>Bella Rakhlina</em><br>
#         <em>Lora Yovcheva</em><br>
#         <em>Mauricio Vargas-Estrada</em><br>
#         <br>Master Of Quantitative Economics<br>
#         University of California - Los Angeles</p>
#         <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));">
#     </div>
# </div>
# %%
import requests
import os 
import joblib
import subprocess
import pandas as pd
import numpy as np
import json

from google.cloud import storage
from io import StringIO, BytesIO
from toolz import pipe


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# %% [markdown]
"""
## First Cloud Function
"""
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        a.) 
#     </div>
#     <div style="padding: 10px;">
#         Post a cloud function that takes in a string of numbers and returns a json file that contains the the sum of all of the single digit numbers.
#     </div>
#     <div style="padding: 10px;">
#     
#     Example : input ="12345"
#     output = 1+2+3+4+5 = 15
#     returns({"answer":15})
# %% [markdown]
"""
The source code is available in the following GitHub repository:  

[ECON446-ORG/first-cloud-function](https://github.com/ECON446-ORG/first-cloud-function)

The cloud functions is deployed directly from the GitHub repository to Google Cloud Platform using Cloud Build and a trigger tracking commits to the `main` branch.
"""
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        b.) 
#     </div>
#     <div style="padding: 10px;">
#         Query your cloud function using requests for example input "012937", "2" and "9999999999999".
# </div>
# %%
url = 'https://us-west2-my-first-function-422520.cloudfunctions.net/first_cloud_function'
# %%
print(requests.post(url, data='012937').content.decode('utf-8'))
# %%
print(requests.post(url, data='2').content.decode('utf-8'))
# %%
print(requests.post(url, data='9999999999999').content.decode('utf-8'))
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        c.) 
#     </div>
#     <div style="padding: 10px;">
#         Add srborghese@g.ucla.edu to your cloud project via IAM.
# </div>
# %% [markdown]
"""
User added in `iam` with `Viewer` role.
"""
# %% [markdown]
"""
## Automated Webscraping
"""
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        a.) 
#     </div>
#     <div style="padding: 10px;">
#         Find a website that is scrapable with Beautiful soup that updates with some frequency. Build a cloud function to programatically scrape the useful content.
#     </div>
# %% [markdown]
"""
The source code is available in the following GitHub repository:  

[ECON446-ORG/automated-webscraping](https://github.com/ECON446-ORG/automated-webscraping)

The cloud functions is deployed directly from the GitHub repository to Google Cloud Platform using Cloud Build and a trigger tracking commits to the `main` branch.

The webscraped data is directly stored in a Google Cloud Storage bucket, and a scheduler is used to trigger the cloud function every day at 12:00 UTC.
"""
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        b.) 
#     </div>
#     <div style="padding: 10px;">
#         Query your stored files.
# </div>
# %%
url = 'https://us-central1-automated-webscraping.cloudfunctions.net/main'
# %%
print(requests.get(url).content.decode('utf-8'))
# %%
df = pipe(
    storage.Client(),
    lambda x: x.bucket('main_webscraping'),
    lambda x: x.blob('1 Day.csv'),
    lambda x: x.download_as_string(),
    lambda x: pd.read_csv(StringIO(x.decode('utf-8'))),
)
# %%
print(df)
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        c.) 
#     </div>
#     <div style="padding: 10px;">
#         State how this could be useful in a business setting.
#     </div>
# %% [markdown]
"""
The RuneScape is a free-to-play massively multiplayer online role-playing game. We web scraped the price of OSRS (Old School RuneScape) Gold, which serves as the primary means to facilitate trades, elevate skill levels, obtain powerful equipment, and indulge in a multitude of entertaining activities in Old School RuneScape. According to Jagex, the developer of RuneScape, selling gold and account names is illegal, but the demand and supply for these goods still exist. On the supply side, Venezuelans, who are facing a severe socioeconomic and political crisis that has led to hyperinflation, are playing RuneScape for up to 10 hours a day. This is because mining gold in the game for 10 hours can be more profitable than working two weeks in their local economy. For gold sellers, staying updated on the daily prices is crucial. This information can help them maximize profits by timing their sales when prices are high. Additionally, businesses that operate in the secondary market for virtual goods can use this data to make informed decisions about buying, selling, and trading OSRS gold. Knowing the trends and fluctuations in gold prices can help them optimize their inventory, set competitive prices, and anticipate market movements. 
"""
# %% [markdown]
"""
## Machine Learning Model
"""
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        a.) 
#     </div>
#     <div style="padding: 10px;">
#         Build some machine learning model using scikit learn and make it queriable using cloud functions.
#     </div>
# %% [markdown]
"""
### Fitting the Model
"""
# %%
pd.options.display.float_format = '{:,.0f}'.format
# %%
data = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
# %%
print(data.head(5))
# %%
data['age'] = data['age'].round()
data['avg_glucose_level'] = data['avg_glucose_level'].round()
data['bmi'] = data['bmi'].round()
data = data[data['gender'] != 'Other']
# %%
value_counts = data['smoking_status'].value_counts()
print("count of each unique value in 'smoking_status' column:")
print(value_counts)
# %%
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})
data['smoking_status'] = data['smoking_status'].replace(
    {'never smoked': 1, 'Unknown': 2,'formerly smoked': 3, 'smokes': 4 }
)
# %%
imputer = KNNImputer(n_neighbors=5)
data[['bmi']] = imputer.fit_transform(data[['bmi']])
# %%
data_select = data[
    [
        "age",
        "gender",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke"
    ]
]
# %%
y = data[["stroke"]]
X = data_select.drop(columns=["stroke"])
# %%
scaler = StandardScaler() 
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
clf = MLPClassifier(
    hidden_layer_sizes=(10,100,100,),
    max_iter=1000,
    random_state=42
)
clf.fit(X_train,y_train)
# %% [markdown]
"""
### Querying the model from the cloud storage bucket.
"""
# %%
def load_scikit_model(file_name):
    bucket_name = "stroke_prediction123"
    source_blob = "stroke/" + file_name
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Gcredentials.json"
    client = storage.Client()
    
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob)
    
    model_data = blob.download_as_string()
    
    model = joblib.load(BytesIO(model_data))
    return(model)
# %%
model = load_scikit_model("stroke_NN.sav")
# %%
preproc = load_scikit_model("stroke_scaler.sav")
# %% [markdown]
"""
### Deployed Cloud Function source code
"""
# %% [markdown]
"""

The entry point is the `stroke_presence` function, which receives a JSON object with the following keys: `age`, `gender`, `heart_disease`, `avg_glucose_level`, `bmi`, and `smoking_status`. The function loads the model and the preprocessor from the cloud storage bucket, preprocesses the input data, and returns the prediction and the probability of a stroke.

```python
import warnings
import google
import joblib
import pandas as pd
import requests
import sklearn
from urllib.parse import parse_qs
from google.cloud import storage
import os
from io import StringIO
from joblib import load
from io import BytesIO
from flask import jsonify

def stroke_presence(request):
    print("Models")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = load_scikit_model("stroke_NN.sav")
            preproc = load_scikit_model("stroke_scaler.sav")
            print("Models Loaded!")
            
            
            print(request)
            dictionary = request.get_json()
            print(dictionary)
            
           
            required_keys = ['age', 'gender', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
            missing_keys = [key for key in required_keys if key not in dictionary]
            if missing_keys:
                raise ValueError(f"Missing required parameter(s): {', '.join(missing_keys)}")
            
            age = float(dictionary['age'])
            gender = int(dictionary['gender']) 
            heart_disease = int(dictionary['heart_disease'])
            avg_glucose_level = float(dictionary['avg_glucose_level'])
            bmi = float(dictionary['bmi'])
            smoking_status = int(dictionary['smoking_status'])  

            print("Variables Set")
            
            
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

def load_scikit_model(file_name):
    bucket_name = "stroke_prediction123"
    source_blob = "stroke/" + file_name
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Gcredentials.json"
    client = storage.Client()
    print("Client Created")
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob)
    
    model_data = blob.download_as_bytes()
    model = joblib.load(BytesIO(model_data))
    return model

```
"""
# %% [markdown]
"""
### Testing the API for the model
"""
# %%
url = "https://us-central1-spring-cloud-econ-446.cloudfunctions.net/stroke_function"
# %%
r  = requests.post(url,
    json={ 
        "age":90,
        "gender":1,
        "heart_disease":1,
        "avg_glucose_level":90,
        "bmi":10, 
        "smoking_status":4                           
    }
)
# %%
print(r.content.decode('utf-8'))
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        b.) 
#     </div>
#     <div style="padding: 10px;">
#         Make a user-friendly input page that takes the inputs to your ML model via widgets and displays the output. Upload a seperate .ipynb that makes this easy to use. (Next Assignment you will have tot urn this into a shareable webpage).
#     </div>
# %% [markdown]
"""
The widget is included in the file `widget.ipynb`.
"""
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        c.) 
#     </div>
#     <div style="padding: 10px;">
#         Think of a company that would use the ML app you just built. What employees could use this app what would they use it for? Write a short paragraph.
#     </div>
# %% [markdown]
"""
Americans. Our machine learning model is designed to predict the probability of a patient experiencing a stroke based on input parameters such as age, gender, presence of heart disease, BMI, glucose levels, and smoking status. Healthcare providers, including doctors, nurses, and medical researchers, could use this app to identify high-risk patients early on. By leveraging larger datasets, the app can improve the accuracy of predictions, facilitating the development of targeted prevention strategies and personalized treatment plans. Also, insurance companies and public health officials could utilize this model to assess population health risks, allocate resources more effectively, and implement preventive health measures on a broader scale.
"""
