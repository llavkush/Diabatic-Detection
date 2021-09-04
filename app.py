import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib  

#from sklearn.ensemble import RandomForestClassifier

st.write("""
# Diabates Detection App
This app predicts the onset of diabetes based on diagnostic measures!
Data obtained from the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/llavkush/Diabatic-Detection/Master/diabetes.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        pregnancies = st.sidebar.slider('No of Pregnancies',0,20,2)
        #sex = st.sidebar.selectbox('Sex',('male','female'))
        Glucose = st.sidebar.slider('Glucose', 0.0,400.0,120.0)
        BloodPressure = st.sidebar.slider('BloodPressure (mm Hg)', 0.00,150.00,69.10)
        SkinThickness = st.sidebar.slider('SkinThickness (mm)', 0.00,150.00,20.53)
        Insulin= st.sidebar.slider('Insulin (mu U/ml)', 0.00,1000.00,79.79)
        DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction (g)', -3.00,3.00,-1.01)
        Age = st.sidebar.slider('Age (Years)', 0,110,22)
        BMI = st.sidebar.slider('Body Mass Index ', 0.00,100.00,31.00)
        data = {'Pregnancy': pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
    
# Combines user input features with entire Diabates dataset
diabates_raw = pd.read_csv('https://raw.githubusercontent.com/llavkush/Diabatic-Detection/Master/diabetes.csv')
diabates = diabates_raw.drop(columns='Outcome', axis=1)
df = pd.concat([input_df,diabates],axis=0)    
    
df = df[:1] # Selects only the first row (the user input data)
df.loc[:].values.tolist() # Converting df into lists
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = scaler.transform(df)

# Encoding of ordinal features


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
disease = np.array(['Non - Diabatic','Diabatic'])
st.write(disease[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
