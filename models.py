# models.py
import joblib
import pandas as pd

def load_artifacts():
    clf = joblib.load('models/lead_scoring_rf.joblib')
    ohe = joblib.load('models/ohe_source.joblib')
    return clf, ohe

def prepare_single_lead(row_dict, ohe):
    # row_dict contains visits,email_opened,last_contact_days,expected_amount,source,created_at (datetime)
    import pandas as pd
    created = pd.to_datetime(row_dict.get('created_at'))
    data = {
        'visits': [row_dict.get('visits',0)],
        'email_opened': [int(row_dict.get('email_opened',0))],
        'last_contact_days': [row_dict.get('last_contact_days',0)],
        'expected_amount': [row_dict.get('expected_amount',0.0)],
        'created_month': [created.month],
        'created_dayofweek': [created.dayofweek]
    }
    df = pd.DataFrame(data)
    src = ohe.transform([[row_dict.get('source','email')]])
    src_cols = [f"src_{c}" for c in ohe.categories_[0]]
    src_df = pd.DataFrame(src, columns=src_cols)
    X_final = pd.concat([df.reset_index(drop=True), src_df.reset_index(drop=True)], axis=1)
    return X_final

def predict_lead_score(row_dict):
    clf, ohe = load_artifacts()
    X = prepare_single_lead(row_dict, ohe)
    score = clf.predict_proba(X)[:,1][0]
    return float(score)
