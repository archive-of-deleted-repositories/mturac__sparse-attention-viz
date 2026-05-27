"""
Utilities: Data Loading, Tokenization, Metrics
===============================================
Handles IMDB/AG News loading, simple word-level tokenizer,
batching, and evaluation metrics. All CPU-friendly.
"""

import re
import time
import tracemalloc
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Simple word-level tokenizer (no external tokenizer dependency)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Word-level tokenizer with vocabulary built from training data.
    Keeps things simple and CPU-friendly — no BPE, no sentencepiece.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, vocab_size: int = 20_000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD, 1: self.UNK}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r"<[^>]+>", " ", text)        # strip HTML
        text = re.sub(r"[^a-z0-9\s]", " ", text)    # keep alphanumeric
        return text.split()

    def build_vocab(self, texts: List[str]) -> "SimpleTokenizer":
        counter: Counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))

        most_common = counter.most_common(self.vocab_size - 2)  # -2 for PAD/UNK
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        tokens = self._tokenize(text)[:max_len]
        return [self.word2idx.get(w, self.UNK_IDX) for w in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.UNK) for i in ids if i != self.PAD_IDX]

    def vocab_len(self) -> int:
        return len(self.word2idx)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: SimpleTokenizer,
        max_len: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(self.texts[idx], self.max_len)
        # Pad
        pad_len = self.max_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [SimpleTokenizer.PAD_IDX] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_imdb(n_samples: Optional[int] = None) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load IMDB dataset. Falls back to synthetic data if HF datasets unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset("imdb")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]

        if n_samples:
            train_texts = train_texts[:n_samples]
            train_labels = train_labels[:n_samples]
            test_texts = test_texts[: n_samples // 4]
            test_labels = test_labels[: n_samples // 4]

        return train_texts, train_labels, test_texts, test_labels

    except Exception:
        print("[WARN] Could not load IMDB from HuggingFace. Using synthetic data.")
        return _synthetic_sentiment(n_samples or 500)


def load_agnews(n_samples: Optional[int] = None) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load AG News dataset (4-class). Falls back to synthetic."""
    try:
        from datasets import load_dataset

        ds = load_dataset("ag_news")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]

        if n_samples:
            train_texts = train_texts[:n_samples]
            train_labels = train_labels[:n_samples]
            test_texts = test_texts[: n_samples // 4]
            test_labels = test_labels[: n_samples // 4]

        return train_texts, train_labels, test_texts, test_labels

    except Exception:
        print("[WARN] Could not load AG News from HuggingFace. Using synthetic data.")
        return _synthetic_news(n_samples or 500)


def _synthetic_sentiment(n: int):
    """Generate simple synthetic sentiment data for demo purposes."""
    positive_words = [
        "great", "amazing", "fantastic", "wonderful", "excellent", "love",
        "best", "beautiful", "perfect", "brilliant", "superb", "outstanding",
    ]
    negative_words = [
        "terrible", "awful", "horrible", "bad", "worst", "hate",
        "boring", "waste", "poor", "disappointing", "dreadful", "ugly",
    ]
    filler = [
        "movie", "film", "acting", "story", "plot", "director",
        "scene", "character", "performance", "script", "cinema", "show",
    ]

    rng = np.random.RandomState(42)
    texts, labels = [], []
    for _ in range(n):
        label = rng.randint(2)
        words = rng.choice(positive_words if label == 1 else negative_words, size=3).tolist()
        words += rng.choice(filler, size=rng.randint(5, 15)).tolist()
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(label)

    split = int(0.8 * n)
    return texts[:split], labels[:split], texts[split:], labels[split:]


def _synthetic_news(n: int):
    """Generate simple synthetic 4-class news data."""
    categories = {
        0: ["stocks", "market", "economy", "trade", "profit", "revenue", "nasdaq", "dow"],
        1: ["game", "team", "player", "score", "season", "championship", "coach", "win"],
        2: ["research", "technology", "software", "AI", "data", "computer", "innovation"],
        3: ["president", "government", "policy", "election", "congress", "senate", "vote"],
    }

    rng = np.random.RandomState(42)
    texts, labels = [], []
    for _ in range(n):
        label = rng.randint(4)
        words = rng.choice(categories[label], size=rng.randint(5, 12)).tolist()
        texts.append(" ".join(words))
        labels.append(label)

    split = int(0.8 * n)
    return texts[:split], labels[:split], texts[split:], labels[split:]


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str = "cpu",
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask)
        loss = criterion(out["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    all_attn_weights = []

    # Measure inference time and memory
    tracemalloc.start()
    start_time = time.perf_counter()

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, attention_mask)
        preds = out["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Keep last batch attention for visualization
        all_attn_weights = out["attention_weights"]

    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, output_dict=True)

    return {
        "accuracy": acc,
        "f1": f1,
        "report": report,
        "inference_time_ms": elapsed * 1000,
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "predictions": all_preds,
        "labels": all_labels,
        "last_attention_weights": all_attn_weights,
    }


def get_single_example_attention(
    model: torch.nn.Module,
    text: str,
    tokenizer: SimpleTokenizer,
    max_len: int = 64,
) -> Dict:
    """Run a single text through the model and return attention maps + tokens."""
    model.eval()
    ids = tokenizer.encode(text, max_len)
    tokens = tokenizer.decode(ids)
    pad_len = max_len - len(ids)
    attention_mask = [1] * len(ids) + [0] * pad_len
    ids_padded = ids + [SimpleTokenizer.PAD_IDX] * pad_len

    input_ids = torch.tensor([ids_padded], dtype=torch.long)
    attn_mask = torch.tensor([attention_mask], dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids, attn_mask)

    return {
        "tokens": tokens,
        "attention_weights": [w[0].numpy() for w in out["attention_weights"]],  # remove batch dim
        "logits": out["logits"][0].numpy(),
        "seq_len": len(tokens),
    }
