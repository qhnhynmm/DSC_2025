import os
import yaml
import logging
from typing import Text

from src.task.train import NLI_Task
from src.task.inference import Predict

def main(config_path: Text) -> None:
    # tắt logging verbose của transformers
    import transformers
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.info("Starting training...")

    # Train model
    task = NLI_Task(config)
    task.train()

    logging.info("Training complete.")
    logging.info("Starting prediction on test set...")

    # Predict
    predictor = Predict(config)
    predictor.predict_submission()

    logging.info("Prediction complete. Submission saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and predict NLI Task")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    main(args.config)
