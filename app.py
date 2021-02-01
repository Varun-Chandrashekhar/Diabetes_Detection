
# Importing Libraries
import sklearn
import streamlit as st # Web application
import numpy as np # linear algebra
import pandas as pd # data processing

# Import the Libraries

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


#Create a title and a sub-title
st.write("""
Diabetes Detection
Helps detect Diabetes using Machine Learning
""")

#Open and Display an Image
image = Image.open('Diabetes Detection.png')
st.image(image, use_column_width=True) # caption = 'ML',

# COMPUTATION
#Get the Data
df = pd.read_csv('diabetes.csv')

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
#split the dataset into 80 percent training and 20 percent test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)

#Glucose
#insulin
#BP
#BMI
#Age
#Pregancy
#DPF
#
#Get the feature input from the user
def get_user_input():
    Glucose = st.sidebar.slider('Glucose (mM)', 0, 200, 115)
    Insulin = st.sidebar.slider('Insulin (pmol/L)', 0.0, 850.0, 30.0)
    Blood_Pressure = st.sidebar.slider('Blood Pressure (mmHg)', 0, 125, 75)
    BMI = st.sidebar.slider('Body Mass Index (BMI)', 0.0, 70.0, 30.0)
    Age = st.sidebar.slider('Age (years)', 20, 90, 30)
    DPF= st.sidebar.slider('Diabetes Prediction Factor (DPF)', 0.078, 2.500, 0.3725)
    Skin_Thickness = st.sidebar.slider('Skin Thickness (μm)', 0, 100, 25)
    Pregnancies = st.sidebar.slider('Pregnancies (Number of Children)', 0, 20, 0)



    #Store a dictionary into a variable
    user_data = {
              'Glucose (mM)': Glucose,
              'Insulin (pmol/L)': Insulin,
              'Blood Pressure (mmHg)': Blood_Pressure,
              'Body Mass Index (BMI)': BMI,
              'Age (years)': Age,
              'Diabetes Prediction Factor (DPF)': DPF,
              'Skin Thickness (μm)': Skin_Thickness,
              'Pregnancies (Number of Children)': Pregnancies
              }

    #Transform the data into a dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

# store the user input into a variable
user_input = get_user_input()

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

# store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# DISPLAY
#Set a subheader and display the users input
st.subheader('User Input: ')
st.write(user_input)

#set a subheader and display the classification
st.subheader('Prediction:')
classification_result = str(prediction)
if classification_result == "[0]":
  classification_result = "Not Diabetic"
else:
  classification_result = "Diabetic"
st.success(classification_result)

#show the model_metrics
accuracy = str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 10000)
accuracy = accuracy[0:4]
accuracy = int(accuracy)
accuracy = accuracy/100
accuracy = str(accuracy)
st.subheader("Accuracy: ")
st.success(accuracy+"%")

#Set a subheader
st.subheader('Data Information: ')
# Show  the data as a table
st.dataframe(df)
#show statistics on the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)
