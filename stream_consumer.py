"""
Structured Streaming consumer: reads transactions from the Kafka
`fraud_transactions` topic, scores them with the RandomForest model
trained by train_model.py, and prints predicted transactions to the console.
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_fraud_rf_model")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "_checkpoints", "fraud_stream")

# Define input schema - matches the JSON fields kafka_producer.py sends
schema = StructType(
    [StructField("Time", DoubleType())]
    + [StructField(f"V{i}", DoubleType()) for i in range(1, 29)]
    + [StructField("Amount", DoubleType())]
)

spark = SparkSession.builder \
    .appName("FraudDetectionStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"Error: model not found at {MODEL_PATH}. Run `python train_model.py` first."
    )
model = PipelineModel.load(MODEL_PATH)

# Read stream from Kafka
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fraud_transactions") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse JSON payloads - removed .na.drop() to prevent silent row drops
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Fill missing V1-V28 feature values with default 0.0 if producer doesn't send them
feature_cols = [f"V{i}" for i in range(1, 29)]
filled_stream = parsed_stream.fillna(0.0, subset=feature_cols)

# Apply PySpark ML Model
predictions = model.transform(filled_stream)

# Select output columns without dropping non-fraud rows
# (Change to filter(col("prediction") == 1.0) if you only want alerts)
output_stream = predictions.select("Time", "Amount", "prediction")

query = output_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .option("truncate", "false") \
    .start()

print("Streaming query started. Processing transaction stream... (Ctrl+C to stop)")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\nStopping streaming query...")
    query.stop()
    spark.stop()