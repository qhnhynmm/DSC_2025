import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from src.data_utils.load_data_csv import Load_csv
import yaml

class TextDataset(Dataset):
    def __init__(self, dataframe, with_labels=True):
        self.data = dataframe 
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx = self.data.loc[index, 'id']
        context = str(self.data.loc[index, 'context'])
        prompt = str(self.data.loc[index, 'prompt'])
        response = str(self.data.loc[index, 'response'])
        if self.with_labels: 
            label = self.data.loc[index, 'label']
            return context, prompt, response, label, idx
        else:
            return context, prompt, response, idx

class DataModule:
    def __init__(self, config):
        self.train_batch_size = config['training']['per_device_train_batch_size']
        self.val_batch_size = config['training']['per_device_eval_batch_size']
        self.test_batch_size = config['testing']['per_device_eval_batch_size']
        self.num_workers = config['training']['num_workers']
        
        self.train_df, self.val_df = Load_csv(config).load_data_train_dev()
        self.test_df = Load_csv(config).load_data_test()

    def get_train_val_loaders(self):
        train_dataset = TextDataset(self.train_df, with_labels=True)
        val_dataset = TextDataset(self.val_df)
        train_loader = TorchDataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = TorchDataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader
        
    def get_test_loader(self):
        test_dataset = TextDataset(self.test_df, with_labels=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)
        return test_loader

if __name__ == "__main__":
    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = DataModule(config).get_train_val_loaders()
    test_loader = DataModule(config).get_test_loader()

    for batch in train_loader:
        print(batch)
        break
