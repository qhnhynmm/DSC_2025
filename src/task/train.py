import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from src.data_utils.load_data import DataModule
from src.model.triple_classifier import TripleClassifier
from src.eval_metric.evaluate import ScoreCalculator


class NLI_Task:
    def __init__(self, config):
        # Load config
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training config
        self.num_epochs = int(config['training']['num_train_epochs'])
        self.patience = int(config['training']['patience'])
        self.lr = float(config['training']['learning_rate'])
        self.weight_decay = float(config['training']['weight_decay'])
        self.save_dir = str(config['training']['output_dir'])
        self.best_metric = str(config['training']['metric_for_best_model'])

        # Data
        self.data_module = DataModule(config)
        self.train_loader, self.val_loader = self.data_module.get_train_val_loaders()

        # Model
        self.model = TripleClassifier(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # Optimizer + Scheduler + AMP
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        # Metric
        self.metric = ScoreCalculator()

        # Make save directory
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        best_score = 0
        patience_counter = 0
        loss_fn = nn.CrossEntropyLoss()

        # Resume last checkpoint if exists
        last_ckpt = os.path.join(self.save_dir, "last_model.pth")
        start_epoch = 0
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            train_loss = 0

            # tqdm cho training loop
            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False)
            for step, batch in enumerate(train_bar):
                logits, labels, _ = self.model(batch)
                labels = labels.to(self.device)
                loss = loss_fn(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            train_loss /= len(self.train_loader)
            self.scheduler.step()

            # Validation
            self.model.eval()
            valid_acc, valid_f1 = 0, 0
            val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Valid]", leave=False)
            with torch.no_grad():
                for batch in val_bar:
                    logits, labels, _ = self.model(batch)
                    labels = labels.to(self.device)
                    preds = torch.argmax(logits, dim=-1)
                    valid_acc += self.metric.acc(labels, preds)
                    valid_f1 += self.metric.f1(labels, preds)

            valid_acc /= len(self.val_loader)
            valid_f1 /= len(self.val_loader)

            # In kết quả epoch
            tqdm.write(f"Epoch {epoch+1}: "
                       f"Train Loss={train_loss:.4f}, "
                       f"Valid Acc={valid_acc:.4f}, "
                       f"Valid F1={valid_f1:.4f}")

            # Save last checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, last_ckpt)

            # Save best checkpoint
            score = valid_acc if self.best_metric.lower() == "accuracy" else valid_f1
            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score': best_score
                }, os.path.join(self.save_dir, "best_model.pth"))
                tqdm.write(f"✅ Saved best model with {self.best_metric}: {best_score:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                tqdm.write(f"⏹️ Early stopping at epoch {epoch+1}")
                break

        tqdm.write("🎉 Training complete.")



if __name__ == "__main__":
    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = NLI_Task(config)
    task.train()
