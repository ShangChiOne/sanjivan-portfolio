# ForecastIQ — AI-Based Food Demand Forecasting System

**Student:** Sanjivan Thiyageswaran | **ID:** TP070073  
**Programme:** BSc (Hons) Computer Science (Artificial Intelligence), APU  
**FYP Title:** Developing an AI-Based Demand Forecasting System to Reduce Food Waste in Small Restaurants

---

## Overview

ForecastIQ is a Flask web application that lets restaurant operators upload historical sales data and receive ML-powered weekly demand forecasts. It uses XGBoost and Random Forest models trained on a Kaggle food delivery dataset (~456K records across 77 fulfillment centers and 51 menu items).

---

## Project Structure

```
FYP Project - TP070073/
├── app/
│   ├── app.py              # Flask backend
│   └── templates/
│       └── index.html      # Single-page frontend
├── data/
│   └── forecasts.db        # SQLite database (auto-created on first run)
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── meal_info.csv
│   └── fulfilment_center_info.csv
├── models/
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.json
│   └── model_metrics.json
├── notebooks/
│   └── demand_forecasting_training.ipynb
└── requirements.txt
```

---

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the app

```bash
python app/app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000
```

---

## Usage

1. Upload a historical sales CSV or Excel file (`.csv`, `.xlsx`, `.xls`)
2. Select a forecast horizon (1, 4, 8, or 12 weeks)
3. View the demand trend chart, model comparison, and per-item forecast table
4. Adjust the **Prep Buffer** (0–20%) to set a safety margin on recommended preparation quantities
5. Filter by **Center** or **Meal** to drill into individual items
6. Edit the **Override** column to manually adjust any forecast
7. Click **Download CSV** to export the filtered results

---

## Models

| Model | MAE | RMSE |
|---|---|---|
| XGBoost | 73.22 | 165.99 |
| Random Forest | 73.23 | 174.57 |

XGBoost is used as the primary forecasting model. Both models are compared visually in the results view.

To retrain the models, run all cells in `notebooks/demand_forecasting_training.ipynb`.

---

## Requirements Covered

| ID | Requirement |
|---|---|
| FR1 | Weekly demand forecast per menu item |
| FR2 | Recommended preparation quantities with adjustable safety buffer |
| FR3 | Upload historical sales data (CSV and Excel) |
| FR4 | Historical sales trend dashboards |
| FR5 | Weekly demand forecasts |
| FR6 | Manual override of individual forecasts |
| FR7 | Filter and drill down to individual menu items |
| FR8 | Downloadable forecast output |
| FR9 | Multiple forecast horizons (1, 4, 8, 12 weeks) |
