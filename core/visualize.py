"""
Visualisation utilities for mechanistic interpretability.
All functions save PNGs to disk and optionally call plt.show().
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Optional

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(name: str, show: bool = False) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


# ── Attention pattern grid ─────────────────────────────────────────────────────

def plot_attn_patterns(
    patterns:   torch.Tensor,          # [L, H, S, S]  or  [H, S, S]
    tokens:     List[str],             # token strings (length S)
    layer:      Optional[int] = None,  # if None, `patterns` is [L, H, S, S]
    title:      str  = "Attention Patterns",
    fname:      str  = "attn_patterns.png",
    show:       bool = False,
) -> None:
    """
    Grid of attention heatmaps.
    If layer is given, patterns should be [H, S, S].
    Otherwise patterns is [L, H, S, S] and we show all layers × heads.
    """
    if patterns.dim() == 4:
        L, H, S, _ = patterns.shape
        fig, axes = plt.subplots(L, H, figsize=(H * 2, L * 2))
        for l in range(L):
            for h in range(H):
                ax = axes[l, h]
                data = patterns[l, h].float().cpu().numpy()
                im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                ax.set_title(f"L{l}H{h}", fontsize=6)
                ax.axis("off")
    else:
        H, S, _ = patterns.shape
        cols = min(H, 6)
        rows = math.ceil(H / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).flatten()
        for h in range(H):
            ax = axes[h]
            data = patterns[h].float().cpu().numpy()
            ax.imshow(data, vmin=0, vmax=1, cmap="Blues")
            ax.set_title(f"L{layer}H{h}", fontsize=8)
            if len(tokens) <= 30:
                ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=5)
                ax.set_yticks(range(S)); ax.set_yticklabels(tokens, fontsize=5)
            else:
                ax.axis("off")
        for h in range(H, len(axes)):
            axes[h].axis("off")

    fig.suptitle(title, fontsize=10)
    _save(fname, show)


# ── Induction score heatmap ────────────────────────────────────────────────────

def plot_induction_scores(
    scores: torch.Tensor,      # [L, H]
    fname:  str  = "induction_scores.png",
    show:   bool = False,
) -> None:
    """
    Heatmap of induction scores.  Red = high (likely induction head).
    """
    data = scores.float().cpu().numpy()
    L, H = data.shape
    fig, ax = plt.subplots(figsize=(H * 0.7, L * 0.7))
    sns.heatmap(
        data, ax=ax,
        xticklabels=[f"H{h}" for h in range(H)],
        yticklabels=[f"L{l}" for l in range(L)],
        cmap="RdYlBu_r",
        annot=True, fmt=".2f", annot_kws={"size": 6},
        linewidths=0.4, linecolor="white",
        vmin=0, vmax=data.max(),
    )
    ax.set_title("Induction Score per Head\n(higher = more induction-like)", fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    _save(fname, show)


# ── Activation patching heatmap ───────────────────────────────────────────────

def plot_patching_heatmap(
    results:  torch.Tensor,     # [L, H]  patching effect (0..1 normalised)
    title:    str  = "Activation Patching",
    fname:    str  = "patching_heatmap.png",
    show:     bool = False,
) -> None:
    """
    Heatmap of patching recovery per (layer, head).
    Green = restores clean performance.  Red = corrupts further.
    """
    data = results.float().cpu().numpy()
    L, H = data.shape
    fig, ax = plt.subplots(figsize=(H * 0.8, L * 0.7))
    sns.heatmap(
        data, ax=ax,
        xticklabels=[f"H{h}" for h in range(H)],
        yticklabels=[f"L{l}" for l in range(L)],
        cmap="RdYlGn",
        annot=True, fmt=".2f", annot_kws={"size": 6},
        linewidths=0.4, linecolor="white",
        center=0, vmin=-0.5, vmax=1.0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    _save(fname, show)


# ── Single attention head ──────────────────────────────────────────────────────

def plot_single_head(
    pattern:  torch.Tensor,     # [S, S]
    tokens:   List[str],
    layer:    int,
    head:     int,
    fname:    Optional[str] = None,
    show:     bool = False,
) -> None:
    S = len(tokens)
    fig, ax = plt.subplots(figsize=(max(6, S * 0.4), max(5, S * 0.4)))
    data = pattern.float().cpu().numpy()
    im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(S)); ax.set_yticklabels(tokens, fontsize=7)
    ax.set_title(f"Layer {layer} Head {head} — Attention Pattern", fontsize=9)
    ax.set_xlabel("Key (source)");  ax.set_ylabel("Query (dest)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fname = fname or f"attn_L{layer}H{head}.png"
    _save(fname, show)


# ── Logit difference bar chart ────────────────────────────────────────────────

def plot_logit_diff_by_head(
    deltas:     torch.Tensor,    # [L, H]  change in LD from ablation
    fname:      str  = "logit_diff_by_head.png",
    show:       bool = False,
    top_n:      int  = 15,
) -> None:
    """
    Bar chart of top-N heads ranked by their logit-difference contribution.
    """
    L, H = deltas.shape
    flat  = deltas.float().cpu().numpy().flatten()
    labels = [f"L{l}H{h}" for l in range(L) for h in range(H)]
    pairs  = sorted(zip(labels, flat), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔLogit Difference (ablation effect)")
    ax.set_title(f"Top-{top_n} Heads by IOI Importance")
    ax.tick_params(axis="x", rotation=45)
    _save(fname, show)
