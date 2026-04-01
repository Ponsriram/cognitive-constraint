from __future__ import annotations

from typing import Iterable


def accuracy(preds: Iterable[int], labels: Iterable[int]) -> float:
    preds_l = list(preds)
    labels_l = list(labels)
    if not labels_l:
        return 0.0
    hits = sum(int(p == y) for p, y in zip(preds_l, labels_l))
    return hits / len(labels_l)


def error_rate(preds: Iterable[int], labels: Iterable[int]) -> float:
    return 1.0 - accuracy(preds, labels)


def stale_or_distractor_hits(
    pred_answers: list[str],
    initial_texts: list[str],
    distractor_texts: list[str],
) -> tuple[int, int]:
    stale_hits = 0
    distractor_hits = 0
    for ans, initial, distractor in zip(pred_answers, initial_texts, distractor_texts):
        a = ans.lower()
        if a in initial.lower():
            stale_hits += 1
        if a in distractor.lower():
            distractor_hits += 1
    return stale_hits, distractor_hits
