#  Importing libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#  Loading the dataset
df=pd.read_csv('diabetes.csv')
# App Heading 
st.header('Lets make a Dashboard')
st.title('Diabetes Prediction App')
st.sidebar.title('Input Features Data :')
st.sidebar.header('Patient Data')

# Showing our data
st.write('Showing our **General** data using HEAD command.')
st.write(df.head())

# Descriptive Statistical Analysis 
st.subheader('Description stats of data')
st.write(df.describe())

# Making x and y form our data
st.subheader('Making our features/labels and predictions or outcomes ')
X=df.drop('Outcome',axis=1)
st.write('X values')
st.write(X)

y=df['Outcome']
st.write('y values',y)

# Splitting our data into Train & Test data
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=1)

# Making sliders for all input features on the left side so that we can change them manually.
def user_input():
    pregnancies=st.sidebar.slider('Pregnancies',0,17,1) # (min,max,slider_strt_value)
    Glucose=st.sidebar.slider('Glucose',0,199,110) # (min,max,slider_strt_value)
    Bp=st.sidebar.slider('BloodPressure',0,122,80) # (min,max,slider_strt_value)
    Skin_thick=st.sidebar.slider('SkinThickness',0,100,12) # (min,max,slider_strt_value)
    Insulin=st.sidebar.slider('Insulin',0,850,80) # (min,max,slider_strt_value)
    BMI=st.sidebar.slider('BMI',0,67,3) # (min,max,slider_strt_value)
    Diabtes_Fun=st.sidebar.slider('DiabetesPedigreeFunction',0.07,2.5,0.31) # (min,max,slider_strt_value)
    Age=st.sidebar.slider('Age',10,81,2) # (min,max,slider_strt_value)

    user_data={ "Pregnancies":pregnancies ,
             "Glucose": Glucose ,
            "Blood Pressure": Bp ,
            "Skin Thickness": Skin_thick ,
            "Insulin" : Insulin ,
            "BMI" : BMI,
            "DiabetePedigreeFunction": Diabtes_Fun ,
            "Age": Age}
    report_data=pd.DataFrame(user_data,index=[0])
    return report_data

user_data=user_input()

# Training the model
model=RandomForestClassifier()
model.fit(X_train,y_train)

# Prediction of the model 
patient_result=model.predict(user_data)

# Visualization
st.title('Visualized Patients Data')


# color function:
if patient_result[0]==0:
    color='blue'
else :
    color='red'
 
 
#  Age vs Pregnancies
st.header('Pregnancy count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='Pregnancies',data=df,hue='Outcome',palette='Greens')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['Pregnancies'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

#  Age vs Glucose
st.header('Glucose count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='Glucose',data=df,hue='Outcome',palette='Blues')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['Glucose'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

#  Age vs BloodPressure
st.header('BP count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='BloodPressure',data=df,hue='Outcome',palette='Reds')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['Blood Pressure'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

#  Age vs SkinThickness
st.header('Skin_Thickness count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='SkinThickness',data=df,hue='Outcome',palette='Oranges')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['Skin Thickness'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.xticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

#  Age vs Insulin
st.header('Insulin count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='Insulin',data=df,hue='Outcome',palette='GnBu')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['Insulin'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.xticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

#  Age vs BMI
st.header('BMI count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='BMI',data=df,hue='Outcome',palette='Purples')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['BMI'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.xticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

#  Age vs DiabetesPedigreeFunction
st.header('DiabetePedigreeFunction count graph others vs your')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='DiabetesPedigreeFunction',data=df,hue='Outcome',palette='PuBu')
ax2=sns.scatterplot(x=user_data["Age"],y=user_data['DiabetePedigreeFunction'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.xticks(np.arange(0,20,2))
plt.title("0 Healthy ,1 Diabetic")
st.pyplot(fig_preg)

# Making heatmap of the dataset
fig_preg2=plt.figure()
ax1=sns.heatmap(data=df,cmap='OrRd')
st.pyplot(fig_preg2)

# Output of user input data
st.header('Your Report Result :')
output=''
if patient_result==0:
    color='blue'
    output='Congratulations! You are safe 🤗'
    st.balloons()
else :
    color='red'
    output='Warning! Your are in danger☠️.'
    st.warning('Alert!')
st.title(output)


