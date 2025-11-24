# app.py
import streamlit as st
import pandas as pd
from models import predict_lead_score
from chatbot import get_answer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Sales Assistant", layout="wide", initial_sidebar_state="expanded")
st.title("Smart Sales Assistant — Demo")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/mock_crm.csv', parse_dates=['created_at'])

df = load_data()

st.sidebar.header("Actions")
mode = st.sidebar.radio("Choose demo", ["Lead Scoring", "Chatbot", "Forecast & Analytics"])

if mode == "Lead Scoring":
    st.header("Lead Scoring Demo")
    st.markdown("Input a lead and get a conversion probability (0-1).")
    with st.form("lead_form"):
        visits = st.number_input("Visits", value=2, min_value=0)
        email_opened = st.checkbox("Email opened?")
        last_contact_days = st.number_input("Days since last contact", value=10, min_value=0)
        expected_amount = st.number_input("Expected Amount", value=1000.0, step=100.0)
        source = st.selectbox("Source", options=['email','ads','referral','organic'])
        submitted = st.form_submit_button("Score Lead")
    if submitted:
        from datetime import datetime
        lead = {
            'visits': visits,
            'email_opened': int(email_opened),
            'last_contact_days': last_contact_days,
            'expected_amount': expected_amount,
            'source': source,
            'created_at': datetime.now()
        }
        try:
            score = predict_lead_score(lead)
            st.metric("Conversion Probability", f"{score:.2%}")
        except Exception as e:
            st.error("Model not trained or artifacts missing. Run training scripts.")
            st.write(e)
    st.subheader("Sample leads")
    st.dataframe(df.sample(10))

elif mode == "Chatbot":
    st.header("Customer Support Chatbot")
    user_q = st.text_input("Ask a question about product/payment/support")
    if st.button("Get Answer"):
        try:
            ans = get_answer(user_q)
            st.success(ans)
        except Exception as e:
            st.error("Chatbot artifacts missing. Run chatbot training.")
            st.write(e)

elif mode == "Forecast & Analytics":
    st.header("Monthly Conversions Forecast")
    if st.button("Load Forecast"):
        try:
            forecast = joblib.load('models/forecast.joblib')
            st.write(forecast.tail(12))
            plt.figure(figsize=(10,4))
            sns.lineplot(data=forecast, x='ds', y='yhat', label='yhat')
            sns.lineplot(data=forecast, x='ds', y='yhat_lower', label='lower', linestyle='--')
            sns.lineplot(data=forecast, x='ds', y='yhat_upper', label='upper', linestyle='--')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        except Exception as e:
            st.error("Forecast missing. Run analytics.py to train forecast.")
            st.write(e)
