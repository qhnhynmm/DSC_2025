from src.model.triple_classifier import create_model
from typing import Dict

def init_model(config: Dict):
    if config["model_type"] == "triple_classifier":
    model = create_model(config)
    return model