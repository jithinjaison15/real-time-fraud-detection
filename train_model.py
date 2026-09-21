"""
Trains a RandomForest fraud-detection model on the Kaggle `creditcard.csv`
dataset and saves it as a PySpark PipelineModel that stream_consumer.py
loads to score the live Kafka stream.

Run once before starting stream_consumer.py:
    python train_model.py
"""

import os
import platform

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Cross-platform base directory: the folder this script lives in, instead of
# a hardcoded Windows path that breaks on Linux/macOS/other machines.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# HADOOP_HOME (winutils.exe) is only needed on Windows for local Spark I/O.
# Only set it if we're actually on Windows AND the path exists, otherwise
# leave it alone so Linux/macOS/Docker runs aren't broken by a bogus path.
if platform.system() == "Windows":
    _hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
    if os.path.exists(_hadoop_home):
        os.environ["HADOOP_HOME"] = _hadoop_home
    else:
        print(
            f"Warning: HADOOP_HOME path '{_hadoop_home}' not found. "
            "If Spark fails with a winutils.exe error, install it and set "
            "HADOOP_HOME before rerunning."
        )

# Initialize PySpark Session
spark = SparkSession.builder \
    .appName("FraudDetectionModelTraining") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load Credit Card Dataset (expected alongside this script)
data_path = os.path.join(BASE_DIR, "creditcard.csv")
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found! Please place creditcard.csv next to this script.")
    spark.stop()
    raise SystemExit(1)

df = spark.read.csv(data_path, header=True, inferSchema=True)

required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class", "Time"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    spark.stop()
    raise SystemExit(f"Error: dataset is missing expected columns: {missing}")

# Drop any fully-null rows so VectorAssembler doesn't choke on them
df = df.na.drop(subset=required_cols)

# Features array (V1 to V28 + Amount) -- matches what kafka_producer.py sends
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestClassifier(labelCol="Class", featuresCol="features", numTrees=20, seed=42)

pipeline = Pipeline(stages=[assembler, rf])

# Simple train/test split so we can sanity-check the model before saving it
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Training Random Forest model...")
model = pipeline.fit(train_df)

predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Validation AUC: {auc:.4f}")

# Save model using a path relative to this script, so it's found by
# stream_consumer.py regardless of which machine/OS this runs on.
model_path = os.path.join(BASE_DIR, "saved_fraud_rf_model")
model.write().overwrite().save(model_path)
print(f"Model successfully saved to {model_path}!")

spark.stop()
