import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Function to load data
@st.cache
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Load the dataset
df = load_data('modified_2022.xlsx')

# Streamlit inputs
st.title('College Rank Prediction')
input_rank = st.number_input('Enter your rank:', min_value=1, value=6171254)
input_gender = st.selectbox('Select gender:', ['boys', 'girls'])
input_caste = st.selectbox('Select caste:', ['OC', 'SC', 'ST', 'BCA', 'BCB', 'BCC', 'BCD', 'BCE', 'OC_EWS'])

# Filter the dataset based on gender
if input_gender == 'boys':
    df = df[df['type_of_college'] != 0]

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

# Select the relevant columns for the given caste
relevant_columns = rank_columns[input_caste]

# Drop rows where both relevant columns are NaN
df_filtered = df.dropna(subset=relevant_columns, how='all')

# Replace NaNs with a high value (indicating less suitable)
df_filtered[relevant_columns] = df_filtered[relevant_columns].fillna(999999)

# Prepare the features and target
features = df_filtered[relevant_columns].min(axis=1).values.reshape(-1, 1)
target = df_filtered[relevant_columns].min(axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

# Predict the input rank
predicted_ranks = rfr.predict([[input_rank]])

# Get the top 5 colleges
distances = np.abs(df_filtered[relevant_columns].min(axis=1) - input_rank)
top_5_indices = distances.nsmallest(5).index
top_5_colleges = df_filtered.loc[top_5_indices]

# Display results
st.write("Predicted Rank for the input:", predicted_ranks[0])
st.write("Top 5 colleges:")
st.write(top_5_colleges[['inst_name', 'branch_code']])
