# generate_mock_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_crm(n=1500):
    np.random.seed(42)
    names = [f"Lead_{i}" for i in range(n)]
    created = [random_date(datetime(2023,1,1), datetime(2024,11,1)) for _ in range(n)]
    visits = np.random.poisson(3, n)
    email_opens = np.random.binomial(1, 0.6, n)
    last_contact_days = np.random.randint(0, 120, n)
    source = np.random.choice(['email','ads','referral','organic'], n, p=[0.4,0.2,0.2,0.2])
    amount_expected = np.round(np.random.exponential(3000, n), 2)
    converted = (np.random.rand(n) < (0.05 + 0.0001*visits + 0.1*email_opens)).astype(int)

    df = pd.DataFrame({
        'lead_id': names,
        'created_at': created,
        'visits': visits,
        'email_opened': email_opens,
        'last_contact_days': last_contact_days,
        'source': source,
        'expected_amount': amount_expected,
        'converted': converted
    })
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/mock_crm.csv', index=False)
    print("mock_crm.csv created.")

def generate_faq():
    faqs = [
        ("How to reset my password?", "Go to profile > reset password and follow the link."),
        ("What are the payment options?", "We accept UPI, cards, and netbanking."),
        ("How to get an invoice?", "Invoices are emailed to your registered address after payment."),
        ("How do refunds work?", "Refunds are processed within 7-10 business days."),
        ("What is the trial period?", "You get a 14-day free trial for the pro plan.")
    ]
    df = pd.DataFrame(faqs, columns=['question', 'answer'])
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/faq.csv', index=False)
    print("faq.csv created.")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    generate_crm(n=1500)
    generate_faq()
