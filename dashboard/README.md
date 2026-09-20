# Diabetes Risk Prediction Dashboard

A 4-page Streamlit dashboard built on top of the `diabetes-risk-prediction`
notebooks pipeline (notebooks 01-06). Reuses the exact trained model,
preprocessor, and feature engineering from that pipeline — nothing here is
reimplemented or approximated.

Navigation is a **top nav bar** (via `st.navigation(position="top")`), not
the sidebar page list — a small sidebar remains for project branding.

## Pages

1. **🩺 Overview** (`pages/overview.py`) — project summary, headline metrics
2. **🎯 Risk Calculator** — enter your own health/lifestyle info, get a
   personalized risk score with a SHAP waterfall explanation of *why*
3. **📊 About Us** — two tabs in one page:
   - *Model Performance*: how the best model was selected (CV, validation,
     test), confusion matrix, ROC curve, SHAP global importance
   - *Data Insights*: the statistical findings from notebook 02 and the
     health-equity risk-segmentation finding from notebook 06
4. **🏥 Diabetes Info & Clinics** — plain-language diabetes education
   (types, symptoms, risk factors, prevention) plus a clinic locator that
   embeds a Google Maps search (no API key required)

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
├── app.py                          # navigation controller (st.navigation, top bar)
├── pages/
│   ├── overview.py
│   ├── risk_calculator.py
│   ├── about_us.py                 # Model Performance + Data Insights, via st.tabs
│   └── diabetes_info_clinics.py    # education + clinic locator
├── utils/
│   ├── styling.py                  # shared color palette + CSS
│   ├── mappings.py                 # real age/BMI -> dataset's coded categories
│   ├── model_loader.py             # cached loading of model/preprocessor/reports
│   └── predict.py                  # single-input prediction pipeline (reuses src/)
├── tests/                          # pytest coverage for utils/
├── .streamlit/
│   └── config.toml                 # theme colors
└── requirements.txt
```

## Design Notes

- **Page icons are a parameter, not part of the filename**
  (`st.Page(..., icon="🎯")` in `app.py`) — deliberately, since emoji in
  *filenames* can get mangled by some zip tools (notably Windows' built-in
  "Extract All", which falls back to a non-UTF-8 codepage). Emoji in page
  *titles/content* is unaffected and safe.
- **`st.set_page_config()` and the custom CSS are set once**, in `app.py`
  only — calling them again inside a page file would error or be redundant,
  since `app.py` is the sole entry point under `st.navigation()`.
- **No train/serve skew**: `utils/predict.py` imports and calls the *same*
  `src.preprocessing.clean_data`, `src.features.engineer_features`, and the
  fitted `preprocessor.joblib` used during training — it does not
  reimplement any of that logic. `utils/predict.get_shap_model_type()`
  mirrors the same tree-vs-linear check notebook 05 uses, so the SHAP
  explainer always matches whichever model actually won training.
- **Clinic locator caveat**: the embedded map uses a public, no-API-key
  Google Maps search URL. It's a convenience preview, not a verified
  clinic directory — the "Open in Google Maps" link is the reliable
  fallback if the embed doesn't render.
- **Color palette**: deliberately muted (deep teal, warm off-white, earthy
  risk-tier colors) rather than the default bright blue/purple Streamlit
  look — defined once in `utils/styling.py`.
- **Not a diagnostic tool**: every page carries a visible disclaimer. This
  is an educational project, not a medical device.
