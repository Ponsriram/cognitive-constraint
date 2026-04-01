from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class Sample:
    id: int
    space: str
    initial: str
    new: str
    distractor: str
    query: str
    answer: str


class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.itos: list[str] = [self.PAD, self.UNK]

    def add_token(self, token: str) -> None:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def add_text(self, text: str) -> None:
        for tok in tokenize(text):
            self.add_token(tok)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, self.stoi[self.UNK]) for tok in tokenize(text)]

    @property
    def size(self) -> int:
        return len(self.itos)


class CIRALabDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        vocab: Vocab,
        answer_to_id: dict[str, int],
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.answer_to_id = answer_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        return {
            "id": s.id,
            "space": s.space,
            "query_ids": self.vocab.encode(s.query),
            "wm_ids": self.vocab.encode(s.new),
            "lm_initial_ids": self.vocab.encode(s.initial),
            "lm_distractor_ids": self.vocab.encode(s.distractor),
            "label": self.answer_to_id[s.answer],
            "query": s.query,
            "answer": s.answer,
            "initial": s.initial,
            "distractor": s.distractor,
        }


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def read_samples(path: str | Path) -> list[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Sample(**item) for item in raw]


def build_vocab(samples: list[Sample]) -> Vocab:
    vocab = Vocab()
    for s in samples:
        vocab.add_text(s.initial)
        vocab.add_text(s.new)
        vocab.add_text(s.distractor)
        vocab.add_text(s.query)
    return vocab


def build_answer_space(samples: list[Sample]) -> tuple[dict[str, int], dict[int, str]]:
    answers = sorted({s.answer for s in samples})
    a2i = {a: i for i, a in enumerate(answers)}
    i2a = {i: a for a, i in a2i.items()}
    return a2i, i2a


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    assert val_ratio >= 0 and test_ratio >= 0 and (val_ratio + test_ratio) < 1.0
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)

    test_size = int(round(len(samples) * test_ratio))
    val_size = int(round(len(samples) * val_ratio))

    test_idx = idx[:test_size]
    val_idx = idx[test_size : test_size + val_size]
    train_idx = idx[test_size + val_size :]

    to_list = lambda indices: [samples[int(i)] for i in indices]
    return to_list(train_idx), to_list(val_idx), to_list(test_idx)


def _pad(sequences: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return ids, mask


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    query_ids, query_mask = _pad([x["query_ids"] for x in batch])
    wm_ids, wm_mask = _pad([x["wm_ids"] for x in batch])
    lm_i_ids, lm_i_mask = _pad([x["lm_initial_ids"] for x in batch])
    lm_d_ids, lm_d_mask = _pad([x["lm_distractor_ids"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    return {
        "id": [x["id"] for x in batch],
        "space": [x["space"] for x in batch],
        "query_ids": query_ids,
        "query_mask": query_mask,
        "wm_ids": wm_ids,
        "wm_mask": wm_mask,
        "lm_initial_ids": lm_i_ids,
        "lm_initial_mask": lm_i_mask,
        "lm_distractor_ids": lm_d_ids,
        "lm_distractor_mask": lm_d_mask,
        "labels": labels,
        "answer": [x["answer"] for x in batch],
        "initial": [x["initial"] for x in batch],
        "distractor": [x["distractor"] for x in batch],
    }
