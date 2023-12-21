import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Advertising")
st.write("This app predicts the **Sales** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('T.V', 4.3, 7.9, 5.4)
    RADIO = st.sidebar.slider('Radio', 2.0, 4.4, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 6.9, 1.3)
   
    data = {'TV': TV,
            'RADIO': RADIO,
            'Newspaper': Newspaper,
            
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('iris')
X = data.drop(['species'],axis=1)
Y = data.species.copy()

modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
