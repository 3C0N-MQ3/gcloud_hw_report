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
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        b.) 
#     </div>
#     <div style="padding: 10px;">
#         Query your cloud function using requests for example input "012937", "2" and "9999999999999".
# </div>
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
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        b.) 
#     </div>
#     <div style="padding: 10px;">
#         Query your stored files.
# </div>
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
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        b.) 
#     </div>
#     <div style="padding: 10px;">
#         Make a user-friendly input page that takes the inputs to your ML model via widgets and displays the output. Upload a seperate .ipynb that makes this easy to use. (Next Assignment you will have tot urn this into a shareable webpage).
#     </div>
# %% [markdown]
# <div style="border: 1px solid black; border-radius: 5px; overflow: hidden;">
#     <div style="background-color: black; color: white; padding: 5px; text-align: left;">
#        c.) 
#     </div>
#     <div style="padding: 10px;">
#         Think of a company that would use the ML app you just built. What employees could use this app what would they use it for? Write a short paragraph.
#     </div>
