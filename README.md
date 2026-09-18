# Diabetes Risk Prediction

Predicting diabetes/prediabetes risk from lifestyle and health-survey indicators, using the
**CDC Diabetes Health Indicators** dataset (BRFSS 2015, 253,680 respondents, UCI ML Repository ID 891).

This project goes beyond a single model: it includes exploratory analysis, statistical
hypothesis testing, multiple model comparisons, SHAP-based interpretability, and a
risk-segmentation analysis with a health-equity angle.

## Problem Statement

Diabetes is a major chronic health burden. Early identification of at-risk individuals from
easily collected lifestyle/survey data (rather than expensive lab tests) can support cheap,
scalable screening. This project asks: **which lifestyle and health factors most strongly
predict diabetes risk, and how well can we predict it from survey data alone?**

## Dataset

- Source: [CDC Diabetes Health Indicators, UCI ML Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- 253,680 rows, 21 features, binary target (`Diabetes_binary`)
- No missing values; imbalanced target (~86% no diabetes / ~14% prediabetes or diabetes)
- Features span clinical history, lifestyle habits, healthcare access, self-reported
  wellbeing, and demographics

## Pipeline Overview

```
Raw data → Cleaning/Validation → EDA → Statistical Testing → Feature Engineering
   → Model Training (LogReg / Random Forest / XGBoost) → Evaluation (beyond accuracy)
   → Interpretability (SHAP) → Risk Segmentation Analysis
```

## Notebooks

Run in order — each one saves artifacts (processed data, models, figures, CSVs)
that the next notebook loads directly, rather than re-deriving them:

1. **`01_eda.ipynb`** — univariate/bivariate/multivariate exploration, ranked
   feature-target associations using the correct measure per variable type
2. **`02_statistical_analysis.ipynb`** — formal hypothesis tests with
   assumption checks, effect sizes, and multiple-testing correction. Saves
   `reports/statistical_test_results.csv`
3. **`03_feature_engineering.ipynb`** — builds engineered features, splits
   train/val/test, fits the scaling pipeline on train only. Saves
   `data/processed/*.parquet` and `models/preprocessor.joblib`
4. **`04_modeling.ipynb`** — trains all three models, cross-validates on
   train, selects the best on validation, evaluates once on test. Saves
   `models/best_model.joblib` and `models/model_metadata.json`
5. **`05_interpretability_shap.ipynb`** — SHAP global + individual
   explanations for whichever model notebook 04 selected
6. **`06_risk_segmentation.ipynb`** — buckets predictions into risk tiers,
   checks distribution across income/education/healthcare-access subgroups.
   Saves `reports/high_risk_group_summary.csv`

## Dashboard

An interactive Streamlit dashboard lives in `dashboard/` — a Risk Calculator
with personalized SHAP explanations, Model Performance, and Data Insights
pages, built on the exact trained model/preprocessor from notebooks 03-04
(no reimplemented logic, no train/serve skew). **Requires notebooks 01-06 to
have been run first** — see `dashboard/README.md` for setup and details.

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure

```
diabetes-risk-prediction/
├── config/            # config.yaml - paths, seed, model params
├── data/
│   ├── raw/            # original untouched data (gitignored)
│   └── processed/       # train/val/test splits as parquet (gitignored)
├── notebooks/          # 01-06, analysis narrative, calls into src/
├── src/
│   ├── data_loader.py    # fetch from UCI, cache, validate schema
│   ├── preprocessing.py  # cleaning, train/val/test split, scaling pipeline, SMOTE
│   ├── features.py       # BMI category, composite health score, risk factor count
│   ├── eda.py             # point-biserial, Cramer's V, correlation ratio, chi-square
│   ├── hypothesis_testing.py  # normality/variance checks, effect sizes, FDR correction
│   ├── train.py            # model training, cross-validation, save/load
│   ├── evaluate.py          # metrics beyond accuracy, ROC/confusion matrix plots
│   ├── interpret.py          # SHAP explainer wrappers (tree + linear)
│   ├── segmentation.py        # risk tiering, subgroup crosstabs
│   ├── plotting.py             # reusable univariate/bivariate plot grids
│   └── utils.py                 # config loading, seeding, logging
├── dashboard/           # Streamlit app — see dashboard/README.md
│   ├── app.py
│   ├── pages/            # Risk Calculator, Model Performance, Data Insights
│   └── utils/             # styling, input mappings, cached loaders, prediction pipeline
├── models/             # saved trained models + preprocessor + metadata (gitignored)
├── reports/
│   ├── figures/         # exported plots (23+ across all notebooks)
│   ├── statistical_test_results.csv
│   └── high_risk_group_summary.csv
├── tests/               # unit tests for every src/ module (45+ tests)
└── .github/workflows/    # CI: lint + test on push
```

## Key Findings

_To be filled in after running the full pipeline — headline insights with
supporting plots go here. Suggested structure: top 3-5 predictive features
(from notebook 02 + 05), model performance summary, and the health-equity
finding from notebook 06._

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

## How to Reproduce

```bash
git clone <repo-url>
cd diabetes-risk-prediction
pip install -r requirements.txt
jupyter lab notebooks/01_eda.ipynb
```

Run the notebooks **in order, 01 through 06** — each depends on files saved by
the previous one (processed data, trained models, statistical results). The
raw CSV downloads and caches automatically from UCI on first run of
`01_eda.ipynb`; no manual download needed unless you're offline (see
`data/raw/README.md`).

## Future Improvements

- ~~Deploy the best model as an interactive risk-checking demo~~ ✅ done — see `dashboard/`
- ~~Add SHAP-based per-individual explanation to the demo~~ ✅ done — Risk Calculator page shows a waterfall plot per prediction
- Experiment with the 3-class target (`Diabetes_012`: none / pre / diabetic)
- Fairness/bias audit of predictions across income and education subgroups
- Add experiment tracking (MLflow) as more models/configs are tried
- Automate the pipeline end-to-end with a `Makefile` and data versioning (DVC)
- Deploy the dashboard itself (Streamlit Community Cloud, or containerize + host)

## License

MIT — see `LICENSE`.
