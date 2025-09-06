import os
import logging
from typing import Dict, List, Union

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model + checkpoint
        self.model = TripleClassifier(config).to(self.device)
        self.checkpoint_path = os.path.join(config["training"]["output_dir"], "best_model.pth")

        # data
        self.data_module = DataModule(config)
        self.test_loader = self.data_module.get_test_loader()

    def _process_ids(self, id_batch: Union[torch.Tensor, List, tuple, str, int]) -> List[str]:
        """Chuyển id_batch về list string để lưu CSV"""
        if isinstance(id_batch, torch.Tensor):
            return [str(i) for i in id_batch.cpu().tolist()]
        elif isinstance(id_batch, (list, tuple)):
            return [str(i) for i in id_batch]
        else:
            return [str(id_batch)]

    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Loading the best model...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logging.info("Model loaded successfully")

        self.model.eval()
        ids: List[str] = []
        submits: List[str] = []

        logging.info("Obtaining predictions...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Batches"):
                # batch = (context, prompt, response, id_batch)
                context, prompt, response, id_batch = batch
                context, prompt, response = list(context), list(prompt), list(response)

                # Forward
                logits, _ = self.model((context, prompt, response, id_batch))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                # map id2label
                answers = [ID2LABEL[i] for i in preds]
                submits.extend(answers)

                # xử lý id
                ids.extend(self._process_ids(id_batch))

        # Lưu file submission
        save_path = os.path.join(self.config["training"]["output_dir"], "submission.csv")
        df = pd.DataFrame({"id": ids, "label": submits})
        df.to_csv(save_path, index=False)
        logging.info(f"Submission saved to {save_path}")
        logging.info(f"Sample predictions:\n{df.head()}")


if __name__ == "__main__":
    import yaml

    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    predictor = Predict(config)
    predictor.predict_submission()
    
