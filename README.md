# 🌸 Perfume Sales Forecasting Pipeline using AWS, Databricks & Power BI

## 📌 Project Overview

This project builds an end-to-end sales forecasting pipeline for a perfume retail business. The pipeline ingests sales data from Amazon S3, processes it using Apache Spark on Databricks, engineers time-series features, trains an XGBoost model for demand forecasting, stores the predictions back in Amazon S3, and visualizes business insights through a Power BI dashboard.

---

## 🚀 Architecture

Amazon S3
↓
Databricks (PySpark ETL)
↓
Feature Engineering
↓
XGBoost Forecast Model
↓
Forecast Output CSV
↓
Amazon S3
↓
Power BI Dashboard

---

## 🛠 Technologies Used

- Amazon S3
- Databricks
- Apache Spark (PySpark)
- Pandas
- Scikit-Learn
- XGBoost
- Boto3
- Power BI
- Python

---

## 📂 Project Structure

```
Perfume-Sales-Forecasting/
│
├── Data/
│   ├── Perfumes_Sales_Data.csv
│   └── forecast_predictions_output.csv
│
├── Notebook/
│   └── perfume_sales_forecasting_notebook.py
│
├── Dashboard/
│   ├── dashboard.png
│   └── Perfume_Sales_Dashboard.pbix
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙ Pipeline Workflow

### Phase 1 – Data Ingestion
- Read perfume sales dataset from Amazon S3.
- Load data into Databricks using Pandas and PySpark.

### Phase 2 – ETL
- Convert columns to proper data types.
- Create Spark temporary views.

### Phase 3 – Feature Engineering
- Date features
- Rolling averages
- Lag features
- Target variable generation

### Phase 4 – Machine Learning
- Train XGBoost Regressor.
- Forecast future product demand.
- Evaluate using Mean Absolute Error (MAE).

### Phase 5 – Export
- Export prediction results to Amazon S3 using Boto3.

### Phase 6 – Dashboard
- Build interactive Power BI dashboard showing:
  - Revenue
  - Units Sold
  - Marketing Spend
  - Forecast Demand
  - Product Category Analysis
  - Forecast vs Actual Demand

---

## 📊 Dashboard Preview

(Add dashboard screenshot here)

---

## 📈 Skills Demonstrated

- Data Engineering
- ETL Pipeline
- AWS S3
- Databricks
- Apache Spark
- Feature Engineering
- Machine Learning
- XGBoost
- Power BI Dashboarding

---

## 👨‍💻 Author

Ankit Mehra
