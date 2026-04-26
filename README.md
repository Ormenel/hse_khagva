

## Repo structure

`preprocessing/` - Raw data clean and panel construction (on server Spark)

`modelling/` - Final data preprocessing, feature engineering and model fit (on server Spark + sklearn)

`models/` - Stored fitted models + features + inference of models from raw input from frontend

`backend/` - FAST API backend for OAS service (Docker)

`frontend/` - Streamlit frontend for OAS service (Docker)

`eda/` - EDA for the project

## How to run service

If you have PyCharm just play in `docker-compose.yml`

- Run following commands:
`docker compose build`
`docker compose up`

- Open: `http://localhost:8501`
- At the end run: `docker compose down`


## Models

### Prepayment Prediction

| Model                   | Params                                                                                   |
|-------------------------|------------------------------------------------------------------------------------------|
| **XGBoost**             | Primary production model. 400 estimators, max_depth=4, learning_rate=0.01, subsample=0.8 |
| **LightGBM**            | 300 estimators, max_depth=4, learning_rate=0.05                                          |
| **Random Forest**       | 300 estimators, max_depth=10, min_samples_leaf=50                                        |
| **Logistic Regression** | Baseline. L2 penalty, C=1000, SAGA solver, balanced class weights                        |
| **SGD Classifier**      | Log-loss, L2 penalty                                                                     |
| **Stacking Ensemble**   | SGD + XGBoost + LightGBM stacked with Logistic Regression meta-learner                   |
| **Voting Ensemble**     | Soft-voting average of the XGBoost + LightGBM + SGD + Random Forest + LR                 |

### Interest Rate Model

**Hull-White 1-Factor** short-rate model is used to simulate interest rate paths for Monte Carlo OAS calculation.

## Data

### Source

**FSingle-Family Historical Loan Performance Dataset** — dataset contains a subset of Fannie Mae's 30-year and less, fully amortizing, full documentation, single-family, conventional fixed-rate mortgages.
(https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)

**FRED GS10** - 10-year Treasury par yield is joined as an exogenous feature and used for computing refinancing incentive signals.
(https://fred.stlouisfed.org/series/GS10)

All info about columns is in `eda/crt-file-layout-and-glossary.pdf`

