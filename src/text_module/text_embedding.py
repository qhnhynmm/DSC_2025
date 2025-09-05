import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from mask.masking import generate_padding_mask
from data_utils.vocab import create_vocab
from peft import LoraConfig, get_peft_model, TaskType


class TextEmbedding(nn.Module):
    def __init__(self, config: Dict, max_len: Optional[int] = None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # tokenizer config
        self.max_length = max_len or config["tokenizer"]["max_length"]
        self.padding = config["tokenizer"]["padding"]
        self.truncation = config["tokenizer"]["truncation"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"]

        # ---- Load tokenizer ----
        tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        if config["text_embedding"]["add_new_token"]:
            new_tokens, _ = create_vocab(config)
            new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
            tokenizer.add_tokens(list(new_tokens))
        self.tokenizer = tokenizer

        # ---- Load encoder ----
        self.encoder = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"]).to(self.device)
        if config["text_embedding"]["add_new_token"]:
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        # ---- Freeze nếu cần ----
        if config["text_embedding"]["freeze"]:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ---- LoRA nếu cần ----
        if not config["text_embedding"]["freeze"] and config["text_embedding"]["use_lora"]:
            lora_cfg = LoraConfig(
                r=config['text_embedding']['lora_r'],
                lora_alpha=config['text_embedding']['lora_alpha'],
                lora_dropout=config['text_embedding']['lora_dropout'],
                target_modules=config['text_embedding']['lora_target_modules'],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        # ---- Projection ----
        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])

    def _tokenize_single(self, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
        """Tokenize một list text."""
        return self.tokenizer(
            texts,
            padding=self.padding if max_len == self.max_length else "max_length",
            truncation=self.truncation,
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=self.return_attention_mask,
        )

    def _combine_two_texts(self, text1: List[str], text2: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ghép text1 + text2 lại thành một chuỗi input_ids và attention_mask."""
        half_len = (self.max_length - 1) // 2  # chia đôi max_length, trừ 1 cho [CLS]

        enc1 = self._tokenize_single(text1, half_len)
        enc2 = self._tokenize_single(text2, half_len)

        input_ids1, mask1 = enc1["input_ids"], enc1["attention_mask"]
        input_ids2, mask2 = enc2["input_ids"], enc2["attention_mask"]

        # Ghép: [CLS] text1 [SEP] text2 [SEP]
        input_ids = torch.cat([input_ids1, input_ids2[:, 1:]], dim=1)
        attention_mask = torch.cat([mask1, mask2[:, 1:]], dim=1)

        # Pad thêm nếu chưa đủ max_length
        cur_len = input_ids.shape[1]
        if cur_len < self.max_length:
            pad_len = self.max_length - cur_len
            pad_token = self.tokenizer.pad_token_id

            pad_ids = torch.full((input_ids.shape[0], pad_len), pad_token, dtype=input_ids.dtype)
            pad_mask = torch.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)

            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        return input_ids, attention_mask

    def forward(self, text1: List[str], text2: Optional[List[str]] = None):
        if text2 is not None:
            input_ids, attention_mask = self._combine_two_texts(text1, text2)
        else:
            enc = self._tokenize_single(text1, self.max_length)
            input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        # Đưa lên device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Padding mask
        padding_mask = generate_padding_mask(input_ids, padding_idx=self.tokenizer.pad_token_id)

        # Projection
        out = self.dropout(self.activation(self.proj(features)))
        return out, padding_mask
