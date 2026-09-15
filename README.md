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

See `notebooks/` for the narrative walkthrough and `src/` for the reusable pipeline code
that the notebooks call into.

## Repository Structure

```
diabetes-risk-prediction/
├── config/            # config.yaml - paths, seed, model params
├── data/
│   ├── raw/            # original untouched data (gitignored)
│   └── processed/       # cleaned / feature-engineered data
├── notebooks/          # analysis narrative, calls into src/
├── src/                # reusable pipeline code
├── models/             # saved trained models
├── reports/figures/     # exported plots
├── tests/               # unit tests for src/ functions
└── .github/workflows/    # CI: lint + test on push
```

## Key Findings

_To be filled in after analysis — headline insights with supporting plots go here._

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
# place the raw CSV in data/raw/ (see data/raw/README.md for source link)
jupyter lab notebooks/01_eda.ipynb
```

## Future Improvements

- Deploy the best model as a FastAPI or Streamlit interactive risk-checking demo
- Add SHAP-based per-individual explanation to the demo ("why was I flagged high-risk?")
- Experiment with the 3-class target (`Diabetes_012`: none / pre / diabetic)
- Fairness/bias audit of predictions across income and education subgroups
- Add experiment tracking (MLflow) as more models/configs are tried
- Automate the pipeline end-to-end with a `Makefile` and data versioning (DVC)

## License

MIT — see `LICENSE`.
