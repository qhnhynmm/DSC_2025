import torch
from torch import nn
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import sys
from src.data_utils.load_data import DataModule
import yaml


class SbertEmbedding(nn.Module):
    """Dùng SBERT chuẩn để embed context, prompt, response -> [batch, 3, d_model]"""
    def __init__(self, config: Dict):
        super().__init__()
        model_name = config["text_embedding"]["text_encoder"] 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(model_name)
        self.encoder = self.encoder.to(device)
        self.device = device

    def forward(self, batch):
        if len(batch) == 5:  # train
            context, prompt, response, label, idx = batch
        else:  # val/test
            context, prompt, response, idx = batch

        # Encode từng input
        with torch.no_grad():
            emb_context = self.encoder.encode(list(context), convert_to_tensor=True, device=self.device)
            emb_prompt = self.encoder.encode(list(prompt), convert_to_tensor=True, device=self.device)
            emb_response = self.encoder.encode(list(response), convert_to_tensor=True, device=self.device)

        # Stack lại [batch, 3, d_model]
        embeddings = torch.stack([emb_context, emb_prompt, emb_response], dim=1)

        if len(batch) == 5:
            return embeddings, label, idx
        else:
            return embeddings, idx


# --- Demo ---
if __name__ == "__main__":
    with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = DataModule(config).get_train_val_loaders()
    model = SbertEmbedding(config)

    for batch in train_loader:
        embeddings, labels, idx = model(batch)
        print("Embedding shape:", embeddings.shape)  # [batch, 3, d_model]
        break

# import torch
# from torch import nn
# from typing import List, Dict, Optional
# from transformers import AutoModel, AutoTokenizer
# from src.data_utils.vocab import create_vocab
# from peft import LoraConfig, get_peft_model, TaskType
# import yaml
# from src.data_utils.load_data import DataModule


# class MeanPooling(nn.Module):
#     def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#         return sum_embeddings / sum_mask


# class SentenceTextEmbedding(nn.Module):
#     def __init__(self, config: Dict, max_len: Optional[int] = None):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.max_length = config["tokenizer"]["max_length"]
#         self.padding = config["tokenizer"]["padding"]
#         self.truncation = config["tokenizer"]["truncation"]
#         self.return_attention_mask = config["tokenizer"]["return_attention_mask"]

#         # Tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
#         if config["text_embedding"]["add_new_token"]:
#             new_tokens, _ = create_vocab(config)
#             new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
#             tokenizer.add_tokens(list(new_tokens))
#         self.tokenizer = tokenizer

#         # Encoder
#         self.encoder = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"]).to(self.device)
#         if config["text_embedding"]["add_new_token"]:
#             self.encoder.resize_token_embeddings(len(self.tokenizer))

#         # Freeze encoder nếu cần
#         if config["text_embedding"]["freeze"]:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#         # LoRA nếu cần
#         if not config["text_embedding"]["freeze"] and config["text_embedding"]["use_lora"]:
#             lora_cfg = LoraConfig(
#                 r=config['text_embedding']['lora_r'],
#                 lora_alpha=config['text_embedding']['lora_alpha'],
#                 lora_dropout=config['text_embedding']['lora_dropout'],
#                 target_modules=config['text_embedding']['lora_target_modules'],
#                 bias="none",
#                 task_type=TaskType.FEATURE_EXTRACTION,
#             )
#             self.encoder = get_peft_model(self.encoder, lora_cfg)

#         self.pooling = MeanPooling()
#         self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(config["text_embedding"]['dropout'])

#     def forward(self, texts: List[str]) -> torch.Tensor:
#         enc = self.tokenizer(
#             texts,
#             padding=self.padding,
#             truncation=self.truncation,
#             max_length=self.max_length,
#             return_tensors="pt",
#             return_attention_mask=self.return_attention_mask,
#         )
#         input_ids = enc["input_ids"].to(self.device)
#         attention_mask = enc["attention_mask"].to(self.device)

#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         token_embeddings = outputs.last_hidden_state
#         sentence_embeddings = self.pooling(token_embeddings, attention_mask)
#         out = self.dropout(self.activation(self.proj(sentence_embeddings)))
#         return out  # [batch, d_model]


# class TripleTextEmbedding(nn.Module):
#     """Embed context, prompt, response -> stack thành [batch, 3, d_model]"""
#     def __init__(self, config: Dict):
#         super().__init__()
#         self.encoder = SentenceTextEmbedding(config)

#     def forward(self, batch):
#         if len(batch) == 5:  # train
#             context, prompt, response, label, idx = batch
#         else:  # val/test
#             context, prompt, response, idx = batch

#         emb_context = self.encoder(list(context))     # [batch, d_model]
#         emb_prompt = self.encoder(list(prompt))       # [batch, d_model]
#         emb_response = self.encoder(list(response))   # [batch, d_model]

#         # stack lại [batch, 3, d_model]
#         embeddings = torch.stack([emb_context, emb_prompt, emb_response], dim=1)

#         if len(batch) == 5:
#             return embeddings, label, idx
#         else:
#             return embeddings, idx


# # --- Demo ---
# if __name__ == "__main__":
#     with open("C:\\Users\\cbnn7\\OneDrive\\Desktop\\DSC_2025\\src\\config\\test.yaml", "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)

#     train_loader, val_loader = DataModule(config).get_train_val_loaders()
#     model = TripleTextEmbedding(config)

#     for batch in train_loader:
#         with torch.no_grad():
#             embeddings, labels, idx = model(batch)
#         print("Embedding shape:", embeddings.shape)  # [batch, 3, d_model]
#         break
