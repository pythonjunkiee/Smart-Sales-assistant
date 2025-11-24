# train_models.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing import load_data, feature_engineer
import os

os.makedirs('models', exist_ok=True)

if __name__ == "__main__":
    df = load_data()
    X, y = feature_engineer(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    joblib.dump(clf, 'models/lead_scoring_rf.joblib')
    print("Model saved to models/lead_scoring_rf.joblib")
