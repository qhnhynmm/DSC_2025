import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict
import yaml
from src.text_module.sbert_embedding import SentenceTextEmbedding
from src.data_utils.load_data import DataModule


class FusionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TripleClassifier(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder = SentenceTextEmbedding(config)
        d_model = config["text_embedding"]["d_model"]
        num_labels = config["model"]["num_labels"]

        self.fusion = FusionLayer(d_model, num_heads=4, dropout=0.1)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_labels)
        )

    def forward(self, batch):
        if len(batch) == 5:  # train
            context, prompt, response, labels, idx = batch
        else:  # val/test
            context, prompt, response, idx = batch
            labels = None

        emb_context = self.encoder(list(context))
        emb_prompt = self.encoder(list(prompt))
        emb_response = self.encoder(list(response))

        embeddings = torch.stack([emb_context, emb_prompt, emb_response], dim=1)
        fused = self.fusion(embeddings)
        pooled = self.pooling(fused.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled)

        if labels is not None:
            return logits, labels, idx
        return logits, idx

def create_model(config: Dict) -> nn.Module:
    return TripleClassifier(config)

    
if __name__ == "__main__":
    import yaml
    from src.data_utils.load_data import DataModule

    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_loader, _ = DataModule(config).get_train_val_loaders()
    model = TripleClassifier(config)

    for batch in train_loader:
        logits, labels, idx = model(batch)
        print("Logits shape:", logits.shape)  # [batch, num_labels]
        break