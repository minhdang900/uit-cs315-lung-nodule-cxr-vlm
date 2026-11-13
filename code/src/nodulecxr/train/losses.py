"""Sample-weighted token cross-entropy."""
from __future__ import annotations

import torch.nn.functional as F


def weighted_token_ce(logits, labels, weights):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    b, t, v = shift_logits.shape
    ce = F.cross_entropy(shift_logits.reshape(-1, v), shift_labels.reshape(-1),
                         ignore_index=-100, reduction="none").view(b, t)
    w = weights.view(-1, 1).to(ce.dtype) * (shift_labels != -100).to(ce.dtype)
    return (ce * w).sum() / w.sum().clamp_min(1e-8)
