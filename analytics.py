# analytics.py
import pandas as pd
from prophet import Prophet
import joblib
import os

def prepare_timeseries(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['created_at']).dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('date')['converted'].sum().reset_index()
    monthly.columns = ['ds','y']
    return monthly

def train_forecast(df):
    ts = prepare_timeseries(df)
    m = Prophet()
    m.fit(ts)
    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)
    os.makedirs('models', exist_ok=True)
    joblib.dump(m, 'models/prophet_model.joblib')
    joblib.dump(forecast[['ds','yhat','yhat_lower','yhat_upper']], 'models/forecast.joblib')
    return forecast

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/mock_crm.csv', parse_dates=['created_at'])
    os.makedirs('models', exist_ok=True)
    f = train_forecast(df)
    print("Forecast trained and saved.")
