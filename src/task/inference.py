import os
import logging
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
import transformers

from src.data_utils.load_data import DataModule
from src.model.triple_classifier import TripleClassifier

# id2label trực tiếp
ID2LABEL = {0: "no", 1: "intrinsic", 2: "extrinsic"}

class Predict:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model + checkpoint
        self.model = TripleClassifier(config).to(self.device)
        self.checkpoint_path = os.path.join(config["training"]["output_dir"], "best_model.pth")

        # data
        self.data_module = DataModule(config)
        self.test_loader = self.data_module.get_test_loader()

    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading the best model...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logging.info("Model loaded successfully")

        self.model.eval()
        ids: List[int] = []
        submits: List[str] = []

        logging.info("Obtaining predictions...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                # batch = (context, prompt, response, id_batch)
                context, prompt, response, id_batch = batch
                context, prompt, response = list(context), list(prompt), list(response)

                # Forward
                logits, idx = self.model((context, prompt, response, id_batch))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                # map id2label
                answers = [ID2LABEL[i] for i in preds]
                submits.extend(answers)

                # Xử lý id
                ids.extend(id_batch if isinstance(id_batch, list) else id_batch.tolist())

        # Lưu file submission
        save_path = os.path.join(self.config["training"]["output_dir"], "submission.csv")
        df = pd.DataFrame({'id': ids, 'label': submits})
        df.to_csv(save_path, index=False)
        logging.info(f"Submission saved to {save_path}")


if __name__ == "__main__":
    import yaml

    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    predictor = Predict(config)
    predictor.predict_submission()
