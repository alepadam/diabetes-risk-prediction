# Diabetes Risk Prediction Dashboard

A 4-page Streamlit dashboard built on top of the `diabetes-risk-prediction`
notebooks pipeline (notebooks 01-06). Reuses the exact trained model,
preprocessor, and feature engineering from that pipeline — nothing here is
reimplemented or approximated.

## Pages

1. **Overview** (`app.py`) — project summary, headline model metrics
2. **🎯 Risk Calculator** — enter your own health/lifestyle info, get a
   personalized risk score with a SHAP waterfall explanation of *why*
3. **📈 Model Performance** — how the best model was selected (CV,
   validation, test) and evaluated, with confusion matrix / ROC curve /
   SHAP importance plots pulled from the notebooks' saved figures
4. **🔎 Data Insights** — the statistical findings from notebook 02 and the
   health-equity risk-segmentation finding from notebook 06

## Prerequisites

**Run the notebooks first** (in order, 01 through 06) from the project
root — this dashboard reads their saved outputs rather than recomputing
anything:

- `models/best_model.joblib`, `models/preprocessor.joblib`,
  `models/model_metadata.json` — from notebooks 03 & 04
- `reports/figures/*.png` — from notebooks 01, 04, 05, 06
- `reports/statistical_test_results.csv` — from notebook 02
- `reports/high_risk_group_summary.csv` — from notebook 06

If any of these are missing, the relevant page shows a clear warning
telling you which notebook to run — it won't crash with a raw file-not-found
error.

## Running the Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` by default.

## Structure

```
dashboard/
├── app.py                          # Overview page (entry point)
├── pages/
│   ├── 1_🎯_Risk_Calculator.py
│   ├── 2_📈_Model_Performance.py
│   └── 3_🔎_Data_Insights.py
├── utils/
│   ├── styling.py                  # shared color palette + CSS
│   ├── mappings.py                 # real age/BMI -> dataset's coded categories
│   ├── model_loader.py             # cached loading of model/preprocessor/reports
│   └── predict.py                  # single-input prediction pipeline (reuses src/)
├── .streamlit/
│   └── config.toml                 # theme colors
└── requirements.txt
```

## Design Notes

- **No train/serve skew**: `utils/predict.py` imports and calls the *same*
  `src.preprocessing.clean_data`, `src.features.engineer_features`, and the
  fitted `preprocessor.joblib` used during training — it does not
  reimplement any of that logic.
- **Color palette**: deliberately muted (deep teal, warm off-white, earthy
  risk-tier colors) rather than the default bright blue/purple Streamlit
  look — defined once in `utils/styling.py`.
- **Not a diagnostic tool**: every page carries a visible disclaimer. This
  is an educational project, not a medical device.
