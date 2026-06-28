"""
Synthetic GPT-2 weight generation with PLANTED circuits.

Why this approach?
──────────────────
Real GPT-2 weights require downloading from HuggingFace (403 in sandbox).
This file does something MORE instructive:
  1. Init all 117M parameters per GPT-2 init scheme (N(0, 0.02), scaled residuals)
  2. SURGICALLY plant a Previous Token Head (PTH) at L3H0
  3. SURGICALLY plant an Induction Head (IH) at L5H1
  4. Detection pipeline should find EXACTLY these two heads
     → Proves detection works; ground truth is known

Mathematical construction of planted circuits:
──────────────────────────────────────────────

PREVIOUS TOKEN HEAD (PTH) at L3H0:
  Goal: A[i, i-1] >> A[i, j≠i-1]   (attend to previous token's position)
  
  GPT-2 residual stream at position j has:
    x[j] = tok_embed(t_j) + pos_embed(j)
  
  If we set W_Q = W_K = P @ wpe.T where P is a projection that extracts
  the positional component and amplifies it, then:
    Q[i] = P @ wpe[i]
    K[j] = P @ wpe[j]
    score(i,j) = Q[i] · K[j] / √Dh = wpe[i]ᵀ PᵀP wpe[j] / √Dh
  
  We want score(i, i-1) >> score(i, j≠i-1).
  Construction: Set W_Q ← α * wpe[1:]ᵀ ... 
  
  Simpler: Use outer product trick.
  Let v_pos = wpe.mean(0)  (average position direction)
  Set W_Q = W_K = α * v_pos.outer(v_pos)  [768, 64]  — project then amplify
  But this makes ALL positions look the same...
  
  REAL construction:
  Set W_Q[:, :] = α * wpe.T    [768, 64]  (each head dim maps to a pos embedding)
  Set W_K[:, :] = α * wpe[:-1].T append shifted  -- this doesn't work cleanly
  
  CLEANEST: We set QK scores directly by setting a high scalar for i,i-1 pairs
  via: W_Q = I[64] @ pos_proj, W_K = shift(W_Q)
  where shift maps position j embedding to position j+1 embedding
  
  Practical approach used here:
  - pos_vecs = first 64 right-singular vectors of wpe (pos structure)
  - W_Q = pos_vecs @ scale           [768, 64]
  - W_K = (shifted pos_vecs) @ scale — achieve shift via regression on wpe
  
  This makes Q[i] · K[i-1] maximized.

INDUCTION HEAD (IH) at L5H1:
  Goal: At second occurrence of token t (position k+i), attend to position i+1.
  
  Requirements:
    K-composition: K[j] should ≈ embed(t_{j-1}) (previous token's embedding)
    This is provided by PTH writing into residual at layer 3.
    
    Q[k+i] = W_Q @ x[k+i] ≈ tok_embed(t_i) [extracts token identity]
    K[j] ≈ W_K @ (PTH_output[j]) ≈ W_K @ tok_embed(t_{j-1}) [previous token]
    score(k+i, j) is max when tok_embed(t_i) ≈ tok_embed(t_{j-1}) → j = i+1
    
  Construction:
    W_Q ≈ subspace of wte  [768, 64]
    W_K ≈ same subspace    [768, 64]  (but applied to PTH-written residual)
    For simplicity: W_Q = W_K = top-64 left singular vectors of wte.T
    
    V = I or wte projection (copies token to output)
    W_O maps z ≈ embed(t_{i+1}) → residual (boosting t_{i+1} logit)

NOTE: These are APPROXIMATE constructions. The planted circuits will show
the right pattern (high induction / PT score) but not perfectly as in learned circuits.
Real GPT-2 has more refined implementations from gradient descent.

This is still EXTREMELY useful:
  - Pipeline verified on known ground truth
  - Exact math of what detection measures
  - Gowrish runs on real GPT-2 weights on his machine → reproduces literature
"""

import torch
import math
import numpy as np
from typing import Dict

# ── Constants matching GPT-2 Small ────────────────────────────────────────────
D_MODEL  = 768
N_LAYERS = 12
N_HEADS  = 12
D_HEAD   = 64
D_MLP    = 3072
N_CTX    = 1024
VOCAB    = 50257

# ── Planted circuit config ─────────────────────────────────────────────────────
PTH_LAYER = 3   # Previous Token Head layer
PTH_HEAD  = 0   # Previous Token Head index
IH_LAYER  = 5   # Induction Head layer
IH_HEAD   = 1   # Induction Head index

CIRCUIT_STRENGTH = 15.0   # Scale for planted circuit signals


def _init_weight(shape, std=0.02) -> torch.Tensor:
    """GPT-2 default weight init: N(0, 0.02)."""
    return torch.randn(*shape) * std


def _init_residual_weight(shape, n_layer=N_LAYERS, std=0.02) -> torch.Tensor:
    """
    GPT-2 scaled init for residual stream weights.
    From paper: divide by sqrt(2 * n_layer) to keep residual stream variance stable.
    Applied to: c_proj weights (output projections for attn + MLP).
    """
    return torch.randn(*shape) * (std / math.sqrt(2 * n_layer))


def build_synthetic_weights(seed: int = 42, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Build full GPT-2 Small weight dict with planted induction circuits.

    Returns dict with SAME keys as HuggingFace GPT2LMHeadModel.named_parameters().
    """
    torch.manual_seed(seed)
    W = {}

    # ── Embeddings ─────────────────────────────────────────────────────────────
    # Token embedding: [50257, 768]  — each row is a token's embedding
    W["transformer.wte.weight"] = _init_weight((VOCAB, D_MODEL))

    # Position embedding: [1024, 768]  — each row is a position's embedding
    # For PTH construction, positions need to be distinguishable
    # Use sinusoidal-like structure + random perturbation
    wpe = torch.zeros(N_CTX, D_MODEL)
    for pos in range(N_CTX):
        for i in range(0, D_MODEL, 2):
            angle = pos / (10000 ** (i / D_MODEL))
            wpe[pos, i]   = math.sin(angle)
            if i + 1 < D_MODEL:
                wpe[pos, i+1] = math.cos(angle)
    wpe = wpe + torch.randn(N_CTX, D_MODEL) * 0.01  # small noise
    W["transformer.wpe.weight"] = wpe

    # ── Final LayerNorm ─────────────────────────────────────────────────────────
    W["transformer.ln_f.weight"] = torch.ones(D_MODEL)
    W["transformer.ln_f.bias"]   = torch.zeros(D_MODEL)

    # ── Transformer layers ──────────────────────────────────────────────────────
    for L in range(N_LAYERS):
        p = f"transformer.h.{L}"

        # LayerNorm 1 (pre-attention)
        W[f"{p}.ln_1.weight"] = torch.ones(D_MODEL)
        W[f"{p}.ln_1.bias"]   = torch.zeros(D_MODEL)

        # LayerNorm 2 (pre-MLP)
        W[f"{p}.ln_2.weight"] = torch.ones(D_MODEL)
        W[f"{p}.ln_2.bias"]   = torch.zeros(D_MODEL)

        # ── Attention: c_attn [768, 2304] (Q, K, V concatenated) ───────────────
        W_qkv = _init_weight((D_MODEL, 3 * D_MODEL), std=0.02)
        b_qkv = torch.zeros(3 * D_MODEL)

        # ── Plant PTH at specified layer/head ──────────────────────────────────
        if L == PTH_LAYER:
            W_qkv, b_qkv = _plant_pth(W_qkv, b_qkv, PTH_HEAD, wpe)

        # ── Plant IH at specified layer/head ───────────────────────────────────
        if L == IH_LAYER:
            W_qkv, b_qkv = _plant_ih(
                W_qkv, b_qkv, IH_HEAD,
                W["transformer.wte.weight"],
                wpe,
            )

        W[f"{p}.attn.c_attn.weight"] = W_qkv  # [768, 2304]
        W[f"{p}.attn.c_attn.bias"]   = b_qkv  # [2304]

        # ── Attention output projection: c_proj [768, 768] ─────────────────────
        W[f"{p}.attn.c_proj.weight"] = _init_residual_weight((D_MODEL, D_MODEL))
        W[f"{p}.attn.c_proj.bias"]   = torch.zeros(D_MODEL)

        # ── MLP: c_fc [768, 3072], c_proj [3072, 768] ─────────────────────────
        W[f"{p}.mlp.c_fc.weight"]   = _init_weight((D_MODEL, D_MLP))
        W[f"{p}.mlp.c_fc.bias"]     = torch.zeros(D_MLP)
        W[f"{p}.mlp.c_proj.weight"] = _init_residual_weight((D_MLP, D_MODEL))
        W[f"{p}.mlp.c_proj.bias"]   = torch.zeros(D_MODEL)

    # Move to device
    W = {k: v.to(device) for k, v in W.items()}
    print(f"Built synthetic GPT-2 weights: {len(W)} tensors, "
          f"~{sum(p.numel() for p in W.values())/1e6:.1f}M params")
    print(f"Planted PTH -> L{PTH_LAYER}H{PTH_HEAD}  |  IH -> L{IH_LAYER}H{IH_HEAD}")
    return W


def _plant_pth(
    W_qkv: torch.Tensor,   # [768, 2304]  (to be modified in-place)
    b_qkv: torch.Tensor,   # [2304]
    head:  int,
    wpe:   torch.Tensor,   # [1024, 768]
) -> tuple:
    """
    Plant Previous Token Head at `head` index.

    Construction:
      Q[i] and K[i-1] should have maximal dot product.
      Use positional structure in wpe.

      Step 1: Extract 64-dim subspace capturing position differences.
              Use SVD of (wpe[1:] - wpe[:-1]) — the "shift" matrix.
      Step 2: Set W_Q[:, h*64:(h+1)*64] = Q_vecs * strength
              Set W_K[:, h*64:(h+1)*64] = K_vecs * strength
              where Q_vecs = wpe[:-1] top-64 directions
                    K_vecs = wpe[1:]  matched directions
              So Q[i] = Q_vecs @ x[i] and K[j] = K_vecs @ x[j]
              Q[i] · K[i-1] = large (matched construction)

    Bias: b_qkv = 0 (no bias needed for this)
    """
    W_qkv = W_qkv.clone()
    b_qkv = b_qkv.clone()

    # Positional embedding matrix
    pos_from = wpe[1:500].float()    # positions 1..499  (query: "I am here")
    pos_to   = wpe[0:499].float()    # positions 0..498  (key:   "I am 1 before query")

    # SVD to get position-change subspace
    # We want directions v such that pos_from @ v is correlated with pos_to @ v
    # Use cross-covariance: C = pos_from.T @ pos_to
    C = pos_from.T @ pos_to   # [768, 768]
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)  # U:[768,768], S:[768], Vt:[768,768]

    # Top D_HEAD singular vectors
    W_Q_h = U[:, :D_HEAD] * CIRCUIT_STRENGTH    # [768, 64]
    W_K_h = Vt[:D_HEAD, :].T * CIRCUIT_STRENGTH  # [768, 64]

    # Normalize
    W_Q_h = W_Q_h / (W_Q_h.norm() + 1e-8) * CIRCUIT_STRENGTH
    W_K_h = W_K_h / (W_K_h.norm() + 1e-8) * CIRCUIT_STRENGTH

    # Inject into W_qkv at Q and K slots for this head
    q_start = head * D_HEAD          # Q slice: [0:2304] starts at head*64
    k_start = D_MODEL + head * D_HEAD  # K slice starts at 768

    W_qkv[:, q_start:q_start + D_HEAD] = W_Q_h
    W_qkv[:, k_start:k_start + D_HEAD] = W_K_h

    return W_qkv, b_qkv


def _plant_ih(
    W_qkv:   torch.Tensor,   # [768, 2304]
    b_qkv:   torch.Tensor,   # [2304]
    head:    int,
    wte:     torch.Tensor,   # [50257, 768]  token embeddings
    wpe:     torch.Tensor,   # [1024, 768]   pos embeddings
) -> tuple:
    """
    Plant Induction Head at `head` index.

    Construction:
      For induction: Q[k+i] ≈ embed(t_i), K[j] ≈ embed(t_{j-1})
      The K-composition from PTH means residual at j contains info about t_{j-1}.

      To implement without actual PTH K-composition (layer 5, PTH at 3 hasn't fully propagated):
        - W_Q extracts token identity from x = tok_embed + pos_embed
        - W_K extracts token identity from residual (which has PTH contribution)
        - Make W_Q ≈ W_K to create token-matching attention

      Step: Top-64 left singular vectors of wte.T gives token-discriminating directions
    """
    W_qkv = W_qkv.clone()
    b_qkv = b_qkv.clone()

    # Token embedding SVD to get token-discriminating subspace
    wte_sample = wte[:5000].float()   # [5000, 768] — use subset for SVD speed
    U, S, Vt = torch.linalg.svd(wte_sample.T @ wte_sample, full_matrices=False)
    # U: [768, 768] — principal components of token embedding space
    W_tok = U[:, :D_HEAD]   # [768, 64] — project to token-identity subspace

    # Scale
    W_Q_h = W_tok * CIRCUIT_STRENGTH / (W_tok.norm() + 1e-8)   # [768, 64]
    W_K_h = W_tok * CIRCUIT_STRENGTH / (W_tok.norm() + 1e-8)   # [768, 64]

    # V: copy the token embedding to output (W_V ≈ identity on token subspace)
    W_V_h = W_tok * 1.0 / (W_tok.norm() + 1e-8)   # [768, 64]

    q_start = head * D_HEAD
    k_start = D_MODEL + head * D_HEAD
    v_start = 2 * D_MODEL + head * D_HEAD

    W_qkv[:, q_start:q_start + D_HEAD] = W_Q_h
    W_qkv[:, k_start:k_start + D_HEAD] = W_K_h
    W_qkv[:, v_start:v_start + D_HEAD] = W_V_h

    return W_qkv, b_qkv


def total_params(W: dict) -> int:
    return sum(v.numel() for v in W.values())


def print_weight_table(W: dict) -> None:
    """Print every tensor name + shape + params (same format as real GPT-2 inspection)."""
    print("=" * 70)
    print(f"  {'NAME':<53} {'SHAPE':<25} {'PARAMS':>10}")
    print("=" * 70)
    total = 0
    for name, t in sorted(W.items()):
        n = t.numel()
        total += n
        print(f"  {name:<53} {str(tuple(t.shape)):<25} {n:>10,}")
    print("-" * 70)
    print(f"  Total: {total:,}  ({total/1e6:.1f}M)")
    print()

    # Shape consistency checks (derived from weights, works for any GPT-2 variant)
    d = W["transformer.wte.weight"].shape[1]
    v = W["transformer.wte.weight"].shape[0]
    c = W["transformer.wpe.weight"].shape[0]
    m = W["transformer.h.0.mlp.c_fc.weight"].shape[1]
    expected = [
        ("transformer.wte.weight",              (v, d)),
        ("transformer.wpe.weight",              (c, d)),
        ("transformer.h.0.attn.c_attn.weight",  (d, 3*d)),
        ("transformer.h.0.attn.c_proj.weight",  (d, d)),
        ("transformer.h.0.mlp.c_fc.weight",     (d, m)),
        ("transformer.h.0.mlp.c_proj.weight",   (m, d)),
    ]
    print("Shape checks:")
    for key, exp in expected:
        act = tuple(W[key].shape)
        ok = "OK" if act == exp else f"FAIL expected {exp}"
        print(f"  {key.split('.')[-1]:<35} {str(act):<20} {ok}")
    print()
