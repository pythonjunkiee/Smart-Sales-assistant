# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def load_data(path='data/mock_crm.csv'):
    df = pd.read_csv(path, parse_dates=['created_at'])
    return df

def feature_engineer(df):
    df = df.copy()
    df['created_month'] = df['created_at'].dt.month
    df['created_dayofweek'] = df['created_at'].dt.dayofweek
    X = df[['visits','email_opened','last_contact_days','expected_amount','source','created_month','created_dayofweek']]
    y = df['converted']
    # One-hot source
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    src = ohe.fit_transform(X[['source']])
    src_cols = [f"src_{c}" for c in ohe.categories_[0]]
    src_df = pd.DataFrame(src, columns=src_cols, index=X.index)
    X_num = X.drop(columns=['source'])
    X_final = pd.concat([X_num.reset_index(drop=True), src_df.reset_index(drop=True)], axis=1)
    os.makedirs('models', exist_ok=True)
    joblib.dump(ohe, 'models/ohe_source.joblib')
    return X_final, y

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    df = load_data()
    X, y = feature_engineer(df)
    print("Features shape:", X.shape)
