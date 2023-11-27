import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from preprocessing import Cleaning
from pycaret.classification import *


# Load the model
model = load_model('model/LightGBM')

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def predict(model, input_df):
    prediciton_df = predict_model(estimator=model, data=input_df)
    predictions = prediciton_df['prediction_label']
    return predictions

def main():
    # Load picture
    image_side = Image.open('img/bank_campaign.jpg')

    # Add option to select online or offline prediction
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
        )

    # Add explanatory text and picture in the sidebar
    st.sidebar.info('This application is used to classify customers who open a term-deposit account and those who do not.')    
    st.sidebar.image(image_side)

    # Add title
    st.title("Bank Campaign - Customers Classification")

    if add_selectbox == 'Online':

        # Set up the form to fill in the required data 

        age = st.number_input(
            'Age', min_value=0, max_value=110, value=0)
        
        job = st.selectbox(
            "Job", ['unemployed', 'management', 'admin.', 'technician', 'self-employed', 'services', 'blue-collar'])

        marital = st.selectbox(
            "Marital Status", ['married', 'single', 'divorced'])

        education = st.selectbox(
            "Education", ['unknown', 'primary', 'secondary', 'tertiary'])

        large_negative_value = -1000000000
        large_positive_value = 1000000000 

        balance = st.number_input(
            'Balance', min_value=large_negative_value, max_value=large_positive_value, value=0)

        housing = st.selectbox(
            "Mortgage Loan", ['Yes', 'No'])
        
        loan = st.selectbox(
            "Personal Loan", ['Yes', 'No'])
        
        default = st.selectbox(
            "Default", ['Yes', 'No'])
        
        contact = st.selectbox(
            "Contact", ['cellular', 'telephone', 'unknown'])

        month = st.selectbox(
            "Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec'])
        
        poutcome = st.selectbox(
            "Previous Campaign Outcome", ['unknown', 'other', 'failure', 'success'])

        duration = st.number_input(
            'Contact Duration', min_value=large_negative_value, max_value=large_positive_value, value=0)
        
        previous = st.number_input(
            'Number of Contact Before Campaign', min_value=large_negative_value, max_value=large_positive_value, value=0)
        
        campaign = st.number_input(
            'Number of Contact During Campaign', min_value=large_negative_value, max_value=large_positive_value, value=0)

        pdays = st.number_input(
            'Number of Days From the Latest Contact', min_value=-2, max_value=1000, value=0)
        
        day = st.number_input(
            'Date of The Latest Contact', min_value=0, max_value=31, value=0)
        
        # Set a variabel to store the output
        output = ""

        input_df = pd.DataFrame([
                {
                    'age': age,
                    'job': job,
                    'marital': marital,
                    'education': education,
                    'balance': balance,
                    'housing': housing,
                    'loan': loan,
                    'default': default,
                    'contact': contact,
                    'month': month,
                    'poutcome': poutcome,
                    'duration': duration,
                    'previous': previous,
                    'campaign': campaign,
                    'pdays': pdays,
                    'day' : day
                }
            ]
        )

        # Make a prediction 
        if st.button("Predict"):
            # st.write(input_df)
            output = model.predict(input_df)
            if (output[0] == 0):
                output = 'The customer will not invest in term-deposit'
            else: 
                output = 'The customer will invest in term-deposit'

        # Show prediction result
        st.success(output)    


    if add_selectbox == 'Batch':

        # Add a feature to upload the file to be predicted
        file_upload = st.file_uploader("Upload csv file for classification", type=["csv"])

        if file_upload is not None:
            # Convert the file to data frame
            data = pd.read_csv(file_upload)

            # Select only columns required by the model
            data = data[[
                'age',
                'job',
                'marital',
                'education',
                'balance',
                'housing',
                'loan',
                'default',
                'contact',
                'month',
                'poutcome', 
                'duration',
                'previous',
                'campaign',
                'pdays',
                'day'
                ]
            ]

            # Make predictions
            data['CustomerClassification'] = np.where(model.predict(data)==1, 'Deposit', 'Not Deposit')

            # Show the result on page
            st.write(data)

            # Add a button to download the prediction result file 
            st.download_button(
                "Press to Download",
                convert_df(data),
                "Classification Result.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == '__main__':
    main()