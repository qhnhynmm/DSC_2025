import re
import unicodedata
# import py_vncorenlp
import pandas as pd
from typing import List, Optional


class VietnamesePreprocessor:
    """
    Bộ tiền xử lý văn bản tiếng Việt.
    """

    def __init__(self, stopwords: Optional[List[str]] = None, use_word_seg: bool = False):
        # Danh sách dấu câu
        self.punctuations = [
            ',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'",
            '`', '...', '--', '-', '_', '*', '@', '#', '$', '%', '^', '&', '+', '=', '/', '\\',
            '|', '~', '``', "''", '“', '”', '‘', '’', '«', '»', '„', '‟', '‹', '›', '〝', '〞',
            '‒', '–', '—', '―', '•', '·', '⋅', '°', ':3', '<3', ':>', ':v', ':)', '=)', ':(',
            '-.-', '-_-'
        ]

        # Stopwords
        self.stopwords = stopwords if stopwords else []

        # # Word segmentation
        # self.use_word_seg = use_word_seg
        # py_vncorenlp.download_model(save_dir="vncorenlp")
        # self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="vncorenlp")

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa Unicode."""
        return unicodedata.normalize("NFC", text)

    def to_lowercase(self, text: str) -> str:
        """Chuyển chữ thường."""
        return text.lower()

    def remove_duplicate_chars(self, text: str) -> str:
        """Loại bỏ ký tự lặp lại."""
        return re.sub(r"(.)\1+", r"\1", text)

    def remove_punctuation(self, text: str) -> str:
        """Loại bỏ dấu câu."""
        for p in self.punctuations:
            text = text.replace(p, " ")
        return re.sub(r"\s+", " ", text).strip()

    def normalize_whitespace(self, text: str) -> str:
        """Chuẩn hóa khoảng trắng."""
        return re.sub(r"\s+", " ", text).strip()

    # def word_segment(self, text: str) -> List[str]:
    #     """Tách từ tiếng Việt."""
    #     return self.rdrsegmenter.word_segment(text)[0]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Xóa stopwords."""
        return [t for t in tokens if t not in self.stopwords]

    def preprocess(self, text: str) -> List[str] | str:
        """
        Tiền xử lý văn bản:
        - Unicode normalize
        - Lowercase
        - Remove duplicate chars
        - Remove punctuation
        - Normalize whitespace
        - Word segmentation (nếu bật)
        - Remove stopwords (nếu có)
        """
        if not isinstance(text, str):
            return ""

        text = self.normalize_unicode(text)
        text = self.to_lowercase(text)
        text = self.remove_duplicate_chars(text)
        text = self.remove_punctuation(text)
        text = self.normalize_whitespace(text)

        # if self.use_word_seg:
        #     tokens = self.word_segment(text)
        #     if self.stopwords:
        #         tokens = self.remove_stopwords(tokens)
        #     return tokens
        return text

    def preprocess_dataframe(self, df: pd.DataFrame, columns: str | List[str]) -> pd.DataFrame:
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' không tồn tại trong DataFrame.")
            df[col] = df[col].apply(self.preprocess)
        return df
