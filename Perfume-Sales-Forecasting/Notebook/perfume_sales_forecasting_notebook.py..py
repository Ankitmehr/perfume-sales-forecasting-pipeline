# Databricks notebook source
# MAGIC %pip install pandas numpy scikit-learn xgboost s3fs boto3 
# MAGIC
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, to_date

# COMMAND ----------

aws_access_key = "your access id"
aws_secret_key = "your seceret key"

bucket_name = "etl-pipeline-17"
bucket_region = "ap-south-1"

file_path = "s3://etl-pipeline-17/uploads/Perfumes_Sales_Data.csv"

# COMMAND ----------

print("Pulling raw data from AWS S3...")

df_pandas = pd.read_csv(
    file_path,
    storage_options={
        "key": aws_access_key,
        "secret": aws_secret_key,
        "client_kwargs": {
            "region_name": bucket_region
        }
    }
)

print(df_pandas.head())

spark_df = spark.createDataFrame(df_pandas)

print("✅ Phase 1 Complete: Data ingested into Spark.")

# COMMAND ----------

spark_df.show(5)
spark_df.printSchema()
print(spark_df.count())

# COMMAND ----------

from pyspark.sql.functions import col, to_date

# Convert data types
spark_df_cleaned = (
    spark_df
    .withColumn("Date", to_date(col("Date")))
    .withColumn("Units_Sold", col("Units_Sold").cast("int"))
    .withColumn("Revenue", col("Revenue").cast("float"))
    .withColumn("Marketing_Spend", col("Marketing_Spend").cast("float"))
)

# Create Temporary View
spark_df_cleaned.createOrReplaceTempView("v_complex_sales")

print("✅ Phase 2 Complete: ETL applied and data staged in memory.")

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F

# Load cleaned data
df = spark.table("v_complex_sales")

# Extract Date Features
df_features = (
    df.withColumn("Year", F.year("Date"))
      .withColumn("Month", F.month("Date"))
      .withColumn("DayOfWeek", F.dayofweek("Date"))
)
# Rolling Windows
window_spec_7 = (
    Window.partitionBy("Product_Category")
          .orderBy("Date")
          .rowsBetween(-7, -1)
)

window_spec_30 = (
    Window.partitionBy("Product_Category")
          .orderBy("Date")
          .rowsBetween(-30, -1)
)

df_features = (
    df_features
    .withColumn("Sales_Lag_7D_Avg",
                F.avg("Units_Sold").over(window_spec_7))
    .withColumn("Sales_Lag_30D_Avg",
                F.avg("Units_Sold").over(window_spec_30))
)

# COMMAND ----------

# Target Label (30 days ahead)
window_lead = (
    Window.partitionBy("Product_Category")
          .orderBy("Date")
)

df_features = df_features.withColumn(
    "Target_Next_Month_Units",
    F.lead("Units_Sold", 30).over(window_lead)
)

# Remove rows with insufficient history
df_final_features = df_features.dropna(
    subset=["Sales_Lag_30D_Avg", "Target_Next_Month_Units"]
)

# Fill missing Marketing Spend values
df_final_features = df_final_features.fillna(
    {"Marketing_Spend": 0.0}
)

# Save as temporary view
df_final_features.createOrReplaceTempView("v_engineered_features")

print("✅ Phase 3 Complete: Features engineered.")

# COMMAND ----------

import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load engineered features
df_ml = spark.table("v_engineered_features").toPandas()

# Sort by Date
df_ml = df_ml.sort_values("Date").reset_index(drop=True)

# One-Hot Encoding
df_encoded = pd.get_dummies(
    df_ml,
    columns=["Product_Category"],
    drop_first=True
)

print(df_encoded.head())

# COMMAND ----------

# Features and Target
features = [
    col for col in df_encoded.columns
    if col not in [
        "Date",
        "Units_Sold",
        "Revenue",
        "Final_Price",
        "Target_Next_Month_Units"
    ]
]

X = df_encoded[features]
y = df_encoded["Target_Next_Month_Units"]

# Chronological Train/Test Split
split_idx = int(len(df_encoded) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Train Model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Prediction
xgb_preds = xgb_model.predict(X_test)

# Evaluation
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"✅ XGBoost MAE: {xgb_mae:.2f} units")

# COMMAND ----------

test_dates = df_ml["Date"].iloc[split_idx:].values
product_cats = df_ml["Product_Category"].iloc[split_idx:].values

df_forecast = pd.DataFrame({
    "Date": test_dates,
    "Product_Category": product_cats,
    "Actual_Future_Demand": y_test.values,
    "Forecasted_Demand": np.round(xgb_preds)
})

spark.createDataFrame(df_forecast).createOrReplaceTempView(
    "v_forecast_predictions"
)

print("✅ Phase 4 Complete: Model trained and predictions generated.")

# COMMAND ----------

import boto3
import io

# Output file name
output_file_name = "forecast_predictions_output.csv"

# Convert Spark DataFrame to Pandas
df_output = spark.table("v_forecast_predictions").toPandas()

# Store CSV in memory
csv_buffer = io.StringIO()

df_output.to_csv(csv_buffer, index=False)

# COMMAND ----------

# Create S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=bucket_region
)

# Upload CSV to S3
s3_client.put_object(
    Bucket=bucket_name,
    Key="forecast_predictions_output.csv",
    Body=csv_buffer.getvalue()
)

print("🎉 Phase 5 Complete: Forecast written to AWS S3!")

# COMMAND ----------

