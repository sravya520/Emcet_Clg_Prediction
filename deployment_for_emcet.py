import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Load the dataset from GitHub
url = "modified_2022.xlsx"
df = pd.read_excel(url)

# Load the trained model from GitHub

with open('rfr_model.sav', 'wb') as f:
    f.write(response.content)
loaded_model = pickle.load(open('rfr_model.sav', 'rb'))

# Streamlit app
st.title('College Admission Predictor')

# Input parameters
input_rank = st.number_input('Enter your rank', min_value=0, step=1)
input_gender = st.selectbox('Select your gender', ['girls', 'boys'])
input_caste = st.selectbox('Select your caste', ['OC', 'SC', 'ST', 'BCA', 'BCB', 'BCC', 'BCD', 'BCE', 'OC_EWS'])

if st.button('Predict'):
    # Filter the dataset based on gender
    df_filtered = df.copy()
    if input_gender == 'boys':
        df_filtered = df_filtered[df_filtered['type_of_college'] != 0]

    # Extract relevant columns based on caste
    rank_columns = {
        'OC': ['OC_BOYS', 'OC_GIRLS'],
        'SC': ['SC_BOYS', 'SC_GIRLS'],
        'ST': ['ST_BOYS', 'ST_GIRLS'],
        'BCA': ['BCA_BOYS', 'BCA_GIRLS'],
        'BCB': ['BCB_BOYS', 'BCB_GIRLS'],
        'BCC': ['BCC_BOYS', 'BCC_GIRLS'],
        'BCD': ['BCD_BOYS', 'BCD_GIRLS'],
        'BCE': ['BCE_BOYS', 'BCE_GIRLS'],
        'OC_EWS': ['OC_EWS_BOYS', 'OC_EWS_GIRLS']
    }

    relevant_columns = rank_columns[input_caste]

    df_filtered = df_filtered.dropna(subset=relevant_columns, how='all')

    df_filtered[relevant_columns] = df_filtered[relevant_columns].fillna(999999)

    # Features and target
    features = df_filtered[relevant_columns].min(axis=1).values.reshape(-1, 1)
    target = df_filtered[relevant_columns].min(axis=1)

    # Predict the rank
    predicted_ranks = loaded_model.predict([[input_rank]])

    # Find the closest colleges
    distances = np.abs(df_filtered[relevant_columns].min(axis=1) - input_rank)
    top_5_indices = distances.nsmallest(5).index
    top_5_colleges = df_filtered.loc[top_5_indices]

    # Display results
    st.subheader('Predicted Rank')
    st.write(predicted_ranks[0])


    st.subheader('Top 5 Closest Colleges')
    st.write(top_5_colleges)
