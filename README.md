# Data-Science-Job-Salary-Predictor

# Data Science Salary Predictor 💵

A machine learning web application that predicts salaries for data science roles in the **US and Canada** market. Built with a Ridge Regression model trained on historical compensation data and deployed as an interactive Streamlit app.

---

## Live App

> Deploy locally or via Streamlit Cloud using the instructions below.

---

## Project Overview

Salary expectations in data science vary widely based on job title, experience, company size, and location — making it difficult for both job seekers and employers to make informed decisions. This project builds a regression model that predicts annual salary based on role attributes, and wraps it in a production-ready web app.

**Problem Type:** Supervised Regression  
**Market:** United States and Canada only  
**Dataset:** [Data Science Job Salaries — Kaggle](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)

---

## Model Performance

| Metric | Value |
|---|---|
| Model | Ridge Regression |
| R² Score | 0.3653 |
| RMSE | ~$54,000 |
| MAE | ~$35,876 |
| Training Samples | US/CA filtered subset |
| Features | 31 engineered features |

> **Note:** The relatively low R² reflects structural limitations in the dataset — it lacks direct signals like years of experience, individual skills, or education level, which are the primary salary drivers. The project prioritizes building a complete, production-ready ML pipeline over raw predictive accuracy.

---

## Features Used

The model uses **31 engineered features** derived from 6 raw inputs:

| Feature Group | Details |
|---|---|
| Experience Level | Binary indicators + ordinal encoding (EN / MI / SE / EX) |
| Employment Type | Full-time, Part-time, Contract, Freelance |
| Remote Work | Fully remote, Hybrid, Onsite + normalized ratio |
| Company Size | Small, Medium, Large + ordinal encoding |
| Location | US-based vs Canada-based binary flags |
| Job Title | Keyword extraction (Data Scientist, Engineer, Analyst, ML Engineer, Manager, seniority signals) |
| Interaction Terms | Experience × Company Size, Experience × Remote, Senior × Large Company |

---

## Project Structure

```
.
├── app.py                              # Streamlit web application
├── salary_prediction_model.pkl         # Serialized Ridge Regression model
├── feature_scaler.pkl                  # Serialized StandardScaler
├── model_metadata.json                 # Model performance stats and training info
├── requirements.txt                    # App dependencies
├── ds_salaries_us_canada.csv           # Filtered US/Canada dataset
├── ds_salaries_usca_engineered.csv     # Feature-engineered dataset
├── INSY-674_Assignment_1_...ipynb      # Notebook 1: EDA and baseline model
├── INSY-684_Assignment_1&2_...ipynb    # Notebook 2: Feature engineering + serialization
└── README.md
```

---

## Pipeline

1. **Data Filtering** — Raw Kaggle data filtered to rows where both employee residence and company location are US or Canada
2. **EDA** — Univariate, categorical, and correlation analysis; log transformation identified for skewed salary distribution
3. **Feature Engineering** — 31 features created from 6 raw inputs including binary indicators, ordinal encodings, and interaction terms
4. **Model Selection** — Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting compared; Ridge selected for best R² and regularization of correlated engineered features
5. **Serialization** — Model and scaler saved with Pickle; metadata saved as JSON for app consumption
6. **Deployment** — Streamlit app loads serialized model, engineers features from user input, and returns a predicted salary with a confidence range

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/shaheryar-haider/ds-salary-predictor.git
cd ds-salary-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (ensure `app.py`, `.pkl` files, `model_metadata.json`, and `requirements.txt` are included)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Select the repository and set `app.py` as the entry point
4. Deploy — Streamlit Cloud will provide a public URL

---

## App Features

- Two-column input form for role and company details
- Predicted salary with **lower and upper confidence bounds** based on model MAE
- Sidebar model card showing R², RMSE, MAE, and training info
- Sanity checks flagging unusually low or high predictions
- Expandable input summary and engineered features inspector
- Example job profiles for reference

---

## Limitations

- R² of 0.37 means the model explains ~37% of salary variance — the dataset lacks key salary drivers (years of experience, skills, education)
- Predictions are based on historical data and may not reflect current market conditions
- Scoped to US and Canada only
- Average prediction error of ~$35,876 — use the confidence range, not the point estimate alone

---

## Dependencies

```
streamlit>=1.28.0
pandas>=2.0.3
numpy>=1.24.3
scikit-learn>=1.3.0
```

---

## Author

**Syed Shaheryar Haider**  
