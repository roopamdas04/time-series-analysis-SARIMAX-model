# Walmart Sales Forecasting — SARIMAX Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-lightgrey)
![pmdarima](https://img.shields.io/badge/pmdarima-2.0+-lightgrey)
![MAPE](https://img.shields.io/badge/MAPE-2.50%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents

1. [Overview](#overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Tools and Technology](#tools-and-technology)
5. [Project Structure](#project-structure)
6. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Key Findings](#key-findings)
9. [Final Conclusion](#Conclusion)
10. [Author & Contact](#author--contact)

---

## Overview

This project builds a **SARIMAX time series model** to forecast weekly sales for Walmart Store 1. Using 143 weeks of historical sales data (February 2010 – October 2012), the model captures annual seasonality, incorporates Consumer Price Index (CPI) as an exogenous predictor, and produces a 20-week forward forecast with 95% confidence intervals.

The final model — **SARIMAX(1,0,1)(0,1,0)[52]** — achieves a **MAPE of 2.50%** and an **RMSE of $53,736** on a held-out test set, making it suitable for practical retail forecasting.

---

## Business Problem

Walmart operates at a scale where even small forecasting errors translate into significant operational costs. Inaccurate sales predictions lead to:

- **Overstocking** — excess inventory ties up capital and increases storage costs
- **Understocking** — lost sales and poor customer experience during peak periods
- **Misaligned budgets** — inaccurate revenue forecasts affect staffing, procurement, and planning

This project answers the question: ***Can we reliably predict weekly sales for a Walmart store to support both inventory management and revenue planning?***

---

## Dataset

| Property | Detail |
|---|---|
| Source | Walmart Sales Forecasting (public dataset) |
| Scope | 45 stores, February 2010 – October 2012 |
| This project | Store 1 only — 143 weekly observations |
| Frequency | Weekly (anchored to Friday) |
| Target variable | `Weekly_Sales` |

**Features in the dataset:**

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Week ending date |
| `Weekly_Sales` | float | Total sales for the week ($) |
| `Holiday_Flag` | binary | 1 if the week contains a public holiday |
| `Temperature` | float | Average regional temperature (°F) |
| `Fuel_Price` | float | Regional fuel price ($/gallon) |
| `CPI` | float | Consumer Price Index |
| `Unemployment` | float | Regional unemployment rate (%) |

---

## Tools and Technology

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Time series modelling | statsmodels (SARIMAX), pmdarima |
| Model evaluation | scikit-learn (MSE), manual MAPE |
| Model persistence | joblib |
| Environment | Jupyter Notebook |

---

## Project Structure

```
walmart-sales-forecasting/
│
├── data
│   ├── Walmart_Data_Analysis_and_Forecasting.csv  # Raw dataset
├── notebook
│   ├── Walmart_Sales_DA1.ipynb       # Main analysis notebook
└── README.md                     # Project documentation
```

---

## Data Cleaning and Preparation

The raw dataset required the following preparation steps before modelling:

- **No missing values** confirmed across all 8 columns (`df.isnull().sum()`)
- **No duplicate rows** confirmed (`df.duplicated().sum()`)
- **Date parsing** — converted `Date` column from string to `datetime` using `dayfirst=True` (format: DD-MM-YYYY)
- **Indexing** — set `Date` as the DatetimeIndex and sorted chronologically
- **Store filtering** — isolated Store 1 (143 rows) from the full 45-store dataset
- **Frequency setting** — applied `asfreq('W-FRI')` to explicitly define weekly frequency, preventing silent misalignment in seasonal modelling
- **Outlier check** — box plots confirmed high-value outliers in `Weekly_Sales` during holiday weeks; these were retained as genuine sales spikes, not data errors

---

## Exploratory Data Analysis

### Seasonality Check
Plotted ACF and PACF on the **raw series** to detect seasonality before any differencing. A statistically significant spike at **lag 52** confirmed annual seasonality — today's sales are correlated with sales from the same week one year ago.

### Stationarity Testing
| Test | Result | Verdict |
|---|---|---|
| ADF (Augmented Dickey-Fuller) | p = 0.013 | Stationary |
| KPSS | p = 0.047 | Non-stationary |

Conflicting results indicate **trend-stationarity**. Tested both d=0 and d=1. The d=1 models produced non-invertible MA parameters (a sign of over-differencing), so **d=0** was selected.

`nsdiffs()` confirmed **D=1** (one seasonal difference required).

### Seasonal Decomposition
Decomposed the series into trend, seasonal, and residual components using `seasonal_decompose(period=52)`. The decomposition confirmed:
- A **gradual upward trend** in sales over time
- A **consistent annual seasonal pattern** with peaks in November–December and a trough in January

### Correlation Analysis
Heatmap of all features against `Weekly_Sales`:

| Feature | Correlation |
|---|---|
| CPI | +0.225 |
| Holiday_Flag | +0.195 |
| Temperature | -0.223 |
| Fuel_Price | +0.125 |
| Unemployment | -0.098 |

**CPI** was selected as the exogenous variable. Holiday_Flag was tested but found to be redundant — with D=1, the model already compares each week to the same week one year prior, absorbing the holiday effect through the seasonal structure.

---

## Key Findings

### Model Selection
After testing multiple SARIMAX configurations (varying p, d, q, P, D, Q), the final model was selected based on:
- Lowest AIC / BIC score
- Prob(Q) > 0.05 (residuals are white noise — Ljung-Box test)
- Prob(JB) > 0.05 (residuals are approximately normal — Jarque-Bera test)
- Lowest RMSE and MAPE on the held-out test set

**Final model: SARIMAX(1,0,1)(0,1,0)[52] with CPI as exogenous variable**

### Diagnostic Results

| Diagnostic | Value | Requirement | Status |
|---|---|---|---|
| Prob(Q) — Ljung-Box | 0.84 | > 0.05 | ✅ Pass |
| Prob(JB) — Jarque-Bera | 0.17 | > 0.05 | ✅ Pass |
| CPI p-value | 0.000 | < 0.05 | ✅ Significant |

### Forecast Accuracy (Test Set)

| Metric | Value |
|---|---|
| RMSE | $53,736 |
| MAPE | 2.50% |
| nRMSE | 3.41% of mean sales |

### CPI Coefficient Interpretation
The CPI coefficient of **+$9,985** means that for every 1-unit increase in CPI, weekly sales increase by approximately $9,985. This reflects the expected relationship between inflation (rising prices) and nominal sales revenue.

### Forward Forecast
The model was retrained on the full dataset (143 weeks) and used to produce a **20-week forward forecast** with 95% confidence intervals. The widening confidence bands over time correctly reflect growing uncertainty, which is expected and desirable behaviour in a well-specified forecasting model.

---

## Conclusion

The **SARIMAX(1,0,1)(0,1,0)[52] model with CPI** is recommended for production use in short-to-medium term sales forecasting for Walmart Store 1, based on the following:

1. **Strong accuracy** — 2.50% MAPE is well within the industry benchmark of under 10% for retail forecasting
2. **Clean residuals** — both Ljung-Box and Jarque-Bera tests confirm the model has fully extracted the signal from the data
3. **Economically meaningful predictor** — CPI is statistically significant and directionally sensible
4. **Operationally applicable** — forecasts can directly inform weekly inventory orders and monthly revenue budgets

**Limitations to be aware of:**
- The model is trained on only 2.8 years of data (~2.8 seasonal cycles). More historical data would improve seasonal parameter estimates and model robustness
- The forward forecast holds CPI constant at its last observed value. In practice, CPI projections from an external economic source should be used
- The model is store-specific. Scaling to all 45 stores would require individual model fitting or a hierarchical forecasting approach

---

## Author & Contact

**Name:** `Roopam Das`

**GitHub:** `github.com/roopamdas04`

**LinkedIn:** `linkedin.com/in/roopamdas`

**Email:** `roopamdas12@gmail.com`

---

*This project was built as part of a data analytics portfolio to demonstrate practical skills in time series analysis, statistical modelling, and forecasting.*
