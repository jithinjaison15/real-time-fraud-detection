# Real-Time Fraud Detection Pipeline

Kafka + Spark Structured Streaming pipeline that scores synthetic credit-card
transactions against a RandomForest model trained on `creditcard.csv`.

## What was fixed

- **docker-compose.yml**: now runs a single Kafka container in **KRaft
  mode** (no Zookeeper). Two earlier problems are both resolved by this:
  1. The original two-container Zookeeper+Kafka setup used a health check
     (`nc`) not installed in the `cp-zookeeper` image, so Zookeeper never
     reported healthy and Kafka waited on it forever.
  2. An interim fix used `bitnami/kafka`, but Bitnami moved most of their
     versioned tags behind a paid "Secure Images" subscription, so
     `bitnami/kafka:3.6` is no longer pullable for free.

  The current setup uses the free `confluentinc/cp-kafka:7.5.0` image
  configured to run both the broker and controller roles itself (KRaft),
  so there's no second container and no dependency race. Health check uses
  `kafka-topics --list`, which is a real CLI tool built into this image.
- **train_model.py**: replaced hardcoded `C:\FraudDetectionProject` paths
  (which only worked on one specific Windows machine) with a path relative
  to the script, made `HADOOP_HOME` only apply on Windows, added a
  train/test split with an AUC check, and validated the CSV has the
  expected columns before training.
- **stream_consumer.py**: model path is now relative (matches
  train_model.py's output), added `startingOffsets=earliest` and
  `failOnDataLoss=false` so it doesn't silently skip data, added a
  `checkpointLocation` (required for a reliable streaming query), and it
  now fails fast with a clear message if the model hasn't been trained yet.
- **kafka_producer.py**: added a connection-retry loop (the original failed
  immediately with `NoBrokersAvailable` if Kafka wasn't fully up yet),
  per-message error handling, and a clean `flush()`/`close()` on exit.
- **kafka_producer.py / requirements.txt**: switched from `kafka-python` to
  `confluent-kafka`. `kafka-python` is unmaintained and fails on Python
  3.12+ with `ModuleNotFoundError: No module named 'kafka.vendor.six.moves'`.
  `confluent-kafka` wraps librdkafka, ships prebuilt wheels for current
  Python versions on Windows/macOS/Linux, and is actively maintained.
- **requirements.txt**: dropped `pandas`/`pyarrow` — neither is actually
  imported by any script in this project, and their old pinned versions
  have no prebuilt wheel for newer Python releases, which forced pip to
  compile from source and fail (`Failed to build 'pandas'`) without a full
  C toolchain. `numpy` is pinned to `1.26.4` (<2.0) because PySpark 3.5.0
  has a known incompatibility with NumPy 2.0+
  ([SPARK-48710](https://issues.apache.org/jira/browse/SPARK-48710)).
- **Python version requirement**: NumPy 1.26.x has no Python 3.13 wheel,
  and PySpark 3.5.0 doesn't work with the NumPy 2.0+ that 3.13 would
  require — so this project needs **Python 3.10, 3.11, or 3.12**, not
  whatever the newest installed Python is. `launch.ps1` now auto-detects a
  compatible interpreter via the Windows `py` launcher instead of blindly
  using `python` (which may resolve to 3.13+) and creates the virtual
  environment with that version.
- **launch.ps1**: the consumer/producer windows were launched by
  dot-sourcing `Activate.ps1` inside a quoted `-Command` string, which broke
  (`& ''` errors) if that path didn't resolve cleanly. Now resolves the venv
  path to an absolute path up front and calls the venv's `python.exe`
  directly, skipping activation entirely.

## Prerequisites

- Docker + Docker Compose
- **Python 3.10, 3.11, or 3.12** — NOT 3.13+. PySpark 3.5.0 is incompatible
  with NumPy 2.0+, and NumPy only ships Python 3.13 wheels starting at
  2.1+, so 3.13 can't satisfy both at once. If you only have Python 3.13
  installed, install 3.12 from python.org alongside it (`launch.ps1` will
  find it automatically via the Windows `py` launcher — no need to change
  your default `python`).
- Java 8/11/17 (required by PySpark)

## Setup

```bash
pip install -r requirements.txt
```

## Run order

1. **Start Kafka/Zookeeper:**
   ```bash
   docker compose up -d
   docker compose ps   # wait until both show healthy
   ```

2. **Train the model** (creates `saved_fraud_rf_model/` next to the script):
   ```bash
   python train_model.py
   ```

3. **Start the consumer** (in its own terminal — it will wait for data):
   ```bash
   python stream_consumer.py
   ```

4. **Start the producer** (in another terminal):
   ```bash
   python kafka_producer.py
   ```

You should see transactions logged by the producer, and any transactions
the model scores as fraud (`prediction == 1.0`) printed by the consumer.

## Shutting down

`Ctrl+C` the producer and consumer, then:
```bash
docker compose down
```
