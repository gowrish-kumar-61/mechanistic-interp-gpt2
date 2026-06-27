"""
Metrics for mechanistic interpretability experiments.

Primary metric for IOI: Logit Difference
─────────────────────────────────────────
    LD = logit(IO_token) − logit(S_token)  at final token position

    Clean run:      LD >> 0  (model correctly predicts IO)
    Corrupted run:  LD << 0  (after name swap, model wants S)

    Patching recovery:
        LD_patch − LD_corrupt
        ────────────────────────  × 100%
        LD_clean − LD_corrupt

    = 100% → patch fully restores clean behavior
    = 0%   → patch has no effect
    < 0%   → patch makes things worse
"""

import torch
from typing import Dict, List, Optional


def logit_diff(
    logits:    torch.Tensor,    # [B, S, V]
    io_token:  int,             # token id of indirect object  e.g. " Mary"
    s_token:   int,             # token id of subject          e.g. " John"
    position:  int = -1,        # which sequence position to read (default: last)
) -> torch.Tensor:
    """
    LD = logit(IO) − logit(S)  at `position` across batch.
    Returns scalar tensor (mean over batch).
    """
    lo = logits[:, position, :]          # [B, V]
    return (lo[:, io_token] - lo[:, s_token]).mean()


def patching_metric(
    ld_patched:   float,
    ld_clean:     float,
    ld_corrupted: float,
) -> float:
    """
    Normalised recovery:
        (LD_patch − LD_corr) / (LD_clean − LD_corr)

    1.0 = full recovery,  0.0 = no effect,  <0 = harmful patch.
    """
    denom = ld_clean - ld_corrupted
    if abs(denom) < 1e-9:
        return 0.0
    return (ld_patched - ld_corrupted) / denom


def top_k_logits(
    logits:    torch.Tensor,    # [B, S, V]  or  [B, V]
    k:         int = 10,
    position:  int = -1,
    batch_idx: int = 0,
) -> List[tuple]:
    """
    Return top-k (token_id, logit_value) at `position` for `batch_idx`.
    """
    lo = logits[batch_idx, position, :] if logits.dim() == 3 else logits[batch_idx]
    vals, ids = lo.topk(k)
    return list(zip(ids.tolist(), vals.tolist()))


def softmax_prob(
    logits:   torch.Tensor,
    token_id: int,
    position: int = -1,
    batch_idx: int = 0,
) -> float:
    """P(token_id) from logits at position."""
    lo    = logits[batch_idx, position, :] if logits.dim() == 3 else logits[batch_idx]
    probs = torch.softmax(lo, dim=-1)
    return probs[token_id].item()


def ablate_head_metric(
    logits_normal:  torch.Tensor,   # [1, S, V]
    logits_ablated: torch.Tensor,   # [1, S, V]
    io_token: int,
    s_token:  int,
) -> Dict[str, float]:
    """
    Measure how much ablating a head hurts performance.
    Returns dict with LD_normal, LD_ablated, delta.
    """
    ld_n = logit_diff(logits_normal,  io_token, s_token).item()
    ld_a = logit_diff(logits_ablated, io_token, s_token).item()
    return {
        "ld_normal":  ld_n,
        "ld_ablated": ld_a,
        "delta":      ld_n - ld_a,   # positive = ablation hurt performance
    }
