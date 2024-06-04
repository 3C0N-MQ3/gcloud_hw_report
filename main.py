# %% [markdown]
"""
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
# %% [markdown]
"""
<!--*!{misc/title.html}-->
"""
# %%
import requests
import os 
import joblib
import warnings
import pandas as pd
import numpy as np

from google.cloud import storage
from io import StringIO
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
"""
<!--*!{sections/q01-a.html}-->
"""
# %% [markdown]
"""
The source code is available in the following GitHub repository:  

[ECON446-ORG/first-cloud-function](https://github.com/ECON446-ORG/first-cloud-function)

The cloud functions is deployed directly from the GitHub repository to Google Cloud Platform using Cloud Build and a trigger tracking commits to the `main` branch.
"""
# %% [markdown]
"""
<!--*!{sections/q01-b.html}-->
"""
# %%
url = 'https://us-west2-my-first-function-422520.cloudfunctions.net/first_cloud_function'
# %%
print(requests.post(url, data='012937').content.decode('utf-8'))
# %%
print(requests.post(url, data='2').content.decode('utf-8'))
# %%
print(requests.post(url, data='9999999999999').content.decode('utf-8'))
# %% [markdown]
"""
<!--*!{sections/q01-c.html}-->
"""
# %% [markdown]
"""
User added in `iam` with `Viewer` role.
"""
# %% [markdown]
"""
## Automated Webscraping
"""
# %% [markdown]
"""
<!--*!{sections/q02-a.html}-->
"""
# %% [markdown]
"""
The source code is available in the following GitHub repository:  

[ECON446-ORG/automated-webscraping](https://github.com/ECON446-ORG/automated-webscraping)

The cloud functions is deployed directly from the GitHub repository to Google Cloud Platform using Cloud Build and a trigger tracking commits to the `main` branch.

The webscraped data is directly stored in a Google Cloud Storage bucket, and a scheduler is used to trigger the cloud function every day at 12:00 UTC.
"""
# %% [markdown]
"""
<!--*!{sections/q02-b.html}-->
"""
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
"""
<!--*!{sections/q02-c.html}-->
"""
# %% [markdown]
"""
The RuneScape is a free-to-play massively multiplayer online role-playing game. We web scraped the price of OSRS (Old School RuneScape) Gold, which serves as the primary means to facilitate trades, elevate skill levels, obtain powerful equipment, and indulge in a multitude of entertaining activities in Old School RuneScape. According to Jagex, the developer of RuneScape, selling gold and account names is illegal, but the demand and supply for these goods still exist. On the supply side, Venezuelans, who are facing a severe socioeconomic and political crisis that has led to hyperinflation, are playing RuneScape for up to 10 hours a day. This is because mining gold in the game for 10 hours can be more profitable than working two weeks in their local economy. For gold sellers, staying updated on the daily prices is crucial. This information can help them maximize profits by timing their sales when prices are high. Additionally, businesses that operate in the secondary market for virtual goods can use this data to make informed decisions about buying, selling, and trading OSRS gold. Knowing the trends and fluctuations in gold prices can help them optimize their inventory, set competitive prices, and anticipate market movements. 
"""
# %% [markdown]
"""
## Machine Learning Model
"""
# %% [markdown]
"""
<!--*!{sections/q03-a.html}-->
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
<!--*!{sections/q03-b.html}-->
"""
# %% [markdown]
"""
<!--*!{sections/q03-c.html}-->
"""
# %% [markdown]
"""
Americans. Our machine learning model is designed to predict the probability of a patient experiencing a stroke based on input parameters such as age, gender, presence of heart disease, BMI, glucose levels, and smoking status. Healthcare providers, including doctors, nurses, and medical researchers, could use this app to identify high-risk patients early on. By leveraging larger datasets, the app can improve the accuracy of predictions, facilitating the development of targeted prevention strategies and personalized treatment plans. Also, insurance companies and public health officials could utilize this model to assess population health risks, allocate resources more effectively, and implement preventive health measures on a broader scale.
"""