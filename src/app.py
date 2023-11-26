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

        age = st.selectbox(
            "Age", ['early working age', 'prime working age', 'mature working age.', 'elderly'])
        if age == 'early working age':
            age = 1
        elif age == 'prime working age':
            age = 2
        elif age == 'mature working age':
            age = 3
        elif age == 'elderly':
            age = 4
        
        job = st.selectbox(
            "Job", ['unemployed', 'management', 'admin.', 'technician', 'self-employed', 'services', 'blue-collar'])
        if job == 'unemployed':
            job = 1
        elif job == 'management':
            job = 2
        elif job == 'admin.':
            job = 3
        elif job == 'technician':
            job = 4
        elif job == 'self-employed':
            job = 5
        elif job == 'services':
            job = 6
        elif job == 'blue-collar':
            job = 7

        marital = st.selectbox(
            "Marital Status", ['married', 'single', 'divorced'])
        if marital == 'married':
            marital = 1
        elif marital == 'single':
            marital = 2
        elif marital == 'divorced':
            marital = 3

        education = st.selectbox(
            "Education", ['unknown', 'primary', 'secondary', 'tertiary'])
        if education == 'unknown':
            education = 1
        elif education == 'primary':
            education = 2
        elif education == 'secondary':
            education = 3
        elif education == 'tertiary':
            education = 4

        large_negative_value = -1000000000
        large_positive_value = 1000000000 

        balance = st.number_input(
            'Balance', min_value=large_negative_value, max_value=large_positive_value, value=0)
        
        housing_choice = {
            0: 'No', 
            1: 'Yes',
        }
        housing = st.selectbox(
            "Mortgage Loan", 
            housing_choice.keys(), 
            format_func=lambda x: housing_choice[x],
            )
        
        loan_choice = {
            0: 'No', 
            1: 'Yes',
        }
        loan = st.selectbox(
            "Personal Loan", 
            loan_choice.keys(), 
            format_func=lambda x: loan_choice[x],
            )
        
        default_choice = {
            0: 'No', 
            1: 'Yes',
        }
        default = st.selectbox(
            "Default", 
            default_choice.keys(), 
            format_func=lambda x: default_choice[x],
            )
        
        contact = st.selectbox(
            "Contact", ['cellular', 'telephone', 'unknown'])
        if contact == 'cellular':
            contact = 1
        elif contact == 'telephone':
            contact = 2
        elif contact == 'unknown':
            contact = 3

        month_choice = {
            1: 'jan', 
            2: 'feb',
            3: 'mar',
            4: 'apr',
            5: 'may',
            6: 'jun',
            7: 'jul',
            8: 'aug',
            9: 'sep',
            10: 'okt',
            11: 'nov',
            12: 'dec'
        }
        month = st.selectbox(
            "Last Contact Month", 
            month_choice.keys(), 
            format_func=lambda x: month_choice[x],
            )
        
        poutcome = st.selectbox(
            "Previous Campaign Outcome", ['unknown', 'other', 'failure', 'success'])
        if poutcome == 'unknown':
            poutcome = 0
        elif poutcome == 'other':
            poutcome = 1
        elif poutcome == 'failure':
            poutcome = 3
        elif poutcome == 'success':
            poutcome = 4

        duration = st.number_input(
            'Contact Duration', min_value=large_negative_value, max_value=large_positive_value, value=0)
        
        previous = st.number_input(
            'Number of Contact Before Campaign', min_value=large_negative_value, max_value=large_positive_value, value=0)
        
        campaign = st.number_input(
            'Number of Contact During Campaign', min_value=large_negative_value, max_value=large_positive_value, value=0)
        
        pdays = st.selectbox(
            "Number of Days From the Latest Contact", ['no prior contact', '0-3 months', '3-6 months', '6-9 months', '9-12 months', 'over a year'])
        if pdays == 'no prior contact':
            pdays = 0
        elif pdays == '0-3 months':
            pdays = 1
        elif pdays == '3-6 months':
            pdays = 3
        elif pdays == '6-9 months':
            pdays = 4
        elif pdays == '9-12 months':
            pdays = 5
        elif pdays == 'over a year':
            pdays = 6
        
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

        input_df = Cleaning().transform(input_df)

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