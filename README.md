# Smart Sales Assistant 

## Overview
A compact end-to-end project:
- Lead scoring model (Random Forest)
- Simple retrieval-based chatbot (TF-IDF)
- Monthly conversions forecasting (Prophet)
- Streamlit dashboard to demo all components

## Project layout
```
smart-sales-assistant/
├─ data/
│  ├─ mock_crm.csv
│  └─ faq.csv
├─ generate_mock_data.py
├─ preprocessing.py
├─ train_models.py
├─ models.py
├─ chatbot.py
├─ analytics.py
├─ app.py
├─ requirements.txt
└─ README.md
```

## Setup
1. Create virtualenv:
```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

2. Install:
```
pip install -r requirements.txt
```

## Generate data and train
```
python generate_mock_data.py
python train_models.py         # trains lead scoring model
python chatbot.py              # prepares chatbot artifacts
python analytics.py            # trains forecast (Prophet)
```

## Run demo UI
```
streamlit run app.py
```

