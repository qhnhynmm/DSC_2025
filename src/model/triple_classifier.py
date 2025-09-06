import torch
from torch import nn
from typing import Dict
from src.text_module.sbert_embedding import SbertEmbedding  
import sys
import os

class FusionLayer(nn.Module):
    """Fusion triple embedding bằng MultiheadAttention"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TripleClassifier(nn.Module):
    """Triple input classifier dùng SBERT chuẩn"""
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder = SbertEmbedding(config)  
        d_model = config["text_embedding"]["d_model"]
        num_labels = config["model"]["num_labels"]

        self.fusion = FusionLayer(d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_labels)
        )

    def forward(self, batch):
        embeddings, *rest = self.encoder(batch)  # embeddings: [batch, 3, d_model]

        fused = self.fusion(embeddings)
        pooled = self.pooling(fused.transpose(1, 2)).squeeze(-1)  # [batch, d_model]

        logits = self.classifier(pooled)  # [batch, num_labels]

        if len(rest) == 2:  # train
            labels, idx = rest
            return logits, labels, idx
        else:  # val/test
            idx = rest[0]
            return logits, idx


def create_model(config: Dict) -> nn.Module:
    return TripleClassifier(config)


# --- Demo ---
if __name__ == "__main__":
    import yaml
    from data_utils.load_data import DataModule

    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_loader, _ = DataModule(config).get_train_val_loaders()
    model = TripleClassifier(config)

    for batch in train_loader:
        logits, labels, idx = model(batch)
        print("Logits shape:", logits.shape)  # [batch, num_labels]
        break
