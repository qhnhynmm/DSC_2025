from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional
import pandas as pd
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
import yaml
from src.data_utils.data_preprocessing import VietnamesePreprocessor

class Load_csv:
    def __init__(self, config):
        self.data_train = config['data']['train_path']
        self.data_test = config['data']['test_path']
        self.dev_ratio = config['data']['dev_ratio']
        self.processing_data = config['data']['processing_data']
        self.preprocessor = VietnamesePreprocessor()

    def load_data_train_dev(self):
        data_train = pd.read_csv(self.data_train)
        if self.processing_data:
            data_train = self.preprocessor.preprocess_dataframe(data_train, columns=['context'])
        train_data, dev_data = train_test_split(data_train, test_size=self.dev_ratio)
        id2label = {0: "no", 1: "intrinsic", 2: "extrinsic"}
        label2id = {"no": 0, "intrinsic": 1, "extrinsic": 2}
        train_data['label'] = train_data['label'].map(label2id)
        dev_data['label'] = dev_data['label'].map(label2id)

        return train_data, dev_data
    
    def load_data_test(self):
        data_test = pd.read_csv(self.data_test)
        if self.processing_data:
            data_test = self.preprocessor.preprocess_dataframe(data_test, columns=['context'])
        return data_test


# # ---- Load YAML ----
# with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)

# load_config = Load_csv(config)

# train_data, dev_data = load_config.load_data_train_dev()
# test_data = load_config.load_data_test()

# print(train_data.head())
# print(dev_data.head())
# print(test_data.head())
