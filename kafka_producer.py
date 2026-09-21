"""
Kafka producer that streams synthetic credit-card-transaction records
onto the `fraud_transactions` topic for the fraud-detection demo pipeline.

Uses the `confluent-kafka` library (a librdkafka binding) rather than
`kafka-python`, which is unmaintained and breaks on Python 3.12+ with
`ModuleNotFoundError: No module named 'kafka.vendor.six.moves'`.

Run this AFTER `docker compose up -d` has brought Kafka up, and after
`train_model.py` has produced `saved_fraud_rf_model/` (the consumer needs
that model to exist).
"""

import time
import json
import random

from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "fraud_transactions"
SEND_INTERVAL_SECONDS = 0.5
MAX_CONNECT_RETRIES = 10
RETRY_BACKOFF_SECONDS = 5


def wait_for_broker():
    """Poll the broker's metadata until it responds, since Kafka can take a
    few seconds to become ready after `docker compose up`, especially on
    first boot when it's still creating its log directories."""
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        try:
            cluster_metadata = admin.list_topics(timeout=5)
            if cluster_metadata.brokers:
                print("Connected to Kafka.")
                return
        except KafkaException:
            pass
        print(
            f"Kafka not reachable yet (attempt {attempt}/{MAX_CONNECT_RETRIES}). "
            f"Retrying in {RETRY_BACKOFF_SECONDS}s..."
        )
        time.sleep(RETRY_BACKOFF_SECONDS)
    raise ConnectionError(
        f"Could not reach Kafka at {BOOTSTRAP_SERVERS} after "
        f"{MAX_CONNECT_RETRIES} attempts. Is `docker compose up -d` running "
        f"and healthy (`docker compose ps`)?"
    )


def make_transaction():
    """Generate one synthetic transaction shaped like the V1-V28/Amount/Time
    schema the model was trained on."""
    is_fraud = random.random() < 0.05  # 5% chance of simulating fraud

    transaction = {
        "Time": time.time(),
        "Amount": round(
            random.uniform(500.0, 5000.0) if is_fraud else random.uniform(1.0, 200.0),
            2,
        ),
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = (
            random.uniform(-5.0, 5.0) if is_fraud else random.uniform(-1.0, 1.0)
        )

    return transaction, is_fraud


def delivery_callback(err, msg):
    if err is not None:
        print(f"Delivery failed for record: {err}")


def main():
    wait_for_broker()
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    print(f"Starting Kafka Producer -> topic '{TOPIC}'. Press Ctrl+C to stop.")

    try:
        while True:
            transaction, is_fraud = make_transaction()
            producer.produce(
                TOPIC,
                value=json.dumps(transaction).encode("utf-8"),
                callback=delivery_callback,
            )
            # Serve delivery callbacks / trigger internal queue housekeeping
            # without blocking for acks.
            producer.poll(0)

            print(
                f"Sent Transaction | Amount: ${transaction['Amount']:.2f} | "
                f"Sim Fraud: {is_fraud}"
            )
            time.sleep(SEND_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
    finally:
        producer.flush(10)
        print("Producer connection closed cleanly.")


if __name__ == "__main__":
    main()
