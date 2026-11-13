"""Turn two teacher-forced continuations into P(yes)."""
from __future__ import annotations

import math


def first_divergence(a, b) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def two_way_probability(logp_yes: float, logp_no: float) -> float:
    m = max(logp_yes, logp_no)
    ey, en = math.exp(logp_yes - m), math.exp(logp_no - m)
    return ey / (ey + en)


def token_logprob_sum(logits, ids, start: int) -> float:
    import torch
    logp = torch.log_softmax(logits[start - 1:-1].float(), dim=-1)
    return float(logp.gather(1, ids[start:].unsqueeze(1)).sum())
