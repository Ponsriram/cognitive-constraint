from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from cclm_cira.metrics import accuracy, stale_or_distractor_hits
from cclm_cira.model import CIRAClassifier


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    stale_hits: int
    distractor_hits: int


def evaluate_model(
    model: CIRAClassifier,
    data_loader: DataLoader,
    id_to_answer: dict[int, str],
    device: torch.device,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    all_preds: list[int] = []
    all_labels: list[int] = []
    pred_answers: list[str] = []
    all_initial: list[str] = []
    all_distractor: list[str] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = model(
                query_ids=batch["query_ids"],
                query_mask=batch["query_mask"],
                wm_ids=batch["wm_ids"],
                wm_mask=batch["wm_mask"],
                lm_initial_ids=batch["lm_initial_ids"],
                lm_initial_mask=batch["lm_initial_mask"],
                lm_distractor_ids=batch["lm_distractor_ids"],
                lm_distractor_mask=batch["lm_distractor_mask"],
            )
            loss = criterion(out.logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)

            pred = torch.argmax(out.logits, dim=-1)
            all_preds.extend(pred.detach().cpu().tolist())
            all_labels.extend(batch["labels"].detach().cpu().tolist())
            pred_answers.extend([id_to_answer[i] for i in pred.detach().cpu().tolist()])
            all_initial.extend(batch["initial"])
            all_distractor.extend(batch["distractor"])

    stale_hits, distractor_hits = stale_or_distractor_hits(pred_answers, all_initial, all_distractor)
    acc = accuracy(all_preds, all_labels)
    denom = max(1, len(all_labels))
    return EvalResult(
        loss=total_loss / denom,
        accuracy=acc,
        stale_hits=stale_hits,
        distractor_hits=distractor_hits,
    )
