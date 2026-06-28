"""
03 — Induction Head Detection

Theory:
    An induction head implements the pattern:
        [..A B .. A B ..]
                  ↑
        At second 'A', attend to position of first 'B' (= first A + 1)

    Two-head circuit (Olsson et al. 2022):
        ┌─────────────────────────────────────────────────────────┐
        │ Previous Token Head  (PTH) — early layer               │
        │   At position j: attends to j-1                        │
        │   Writes "what came before me" into residual           │
        │                                                         │
        │ Induction Head (IH) — later layer, uses PTH output    │
        │   Q[i] ∝ embedding of t_i                              │
        │   K[j] ∝ embedding of t_{j-1}  (from PTH)             │
        │   → high attention when t_{j-1} = t_i                  │
        │   V[j] ≈ t_j  →  copies "what followed t_i before"    │
        └─────────────────────────────────────────────────────────┘

Detection — Induction Score:
    1. Build repeated sequence: [a₁ a₂ … aₖ  a₁ a₂ … aₖ]   len = 2k
    2. Run forward pass, extract attention patterns
    3. Score(L, H) = mean_{i=0}^{k-2}  A[k+i,  i+1]
                                         ↑        ↑
                                   second copy   first copy + 1

    High score ≈ head is an induction head.

Detection — Previous Token Score:
    Score_PT(L, H) = mean_{i=1}^{S-1}  A[i, i-1]
    High = head attends to previous token (strict subdiagonal).

Known GPT-2 Small induction heads (Nanda et al.):
    L5H1, L5H5, L6H9, L7H10, L7H11

Run:
    python 03_induction_heads.py
"""

import sys, torch, math, numpy as np, argparse
sys.path.insert(0, ".")
from core.model import gpt2_forward, config_from_weights, load_weights, get_tokenizer
from core.synthetic_weights import build_synthetic_weights, PTH_LAYER, PTH_HEAD, IH_LAYER, IH_HEAD
from core.visualize import (
    plot_induction_scores,
    plot_single_head,
    plot_attn_patterns,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD REPEATED TOKEN SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════

def make_repeated_tokens(seq_half: int = 50, seed: int = 42) -> torch.Tensor:
    """
    Random tokens, repeated:  [t₀ t₁ … t_{k-1}  t₀ t₁ … t_{k-1}]
    Shape: [1, 2*seq_half]

    Why random? No priors about what to attend to.
    Use high token IDs to avoid special tokens (BOS, EOS, PAD).
    """
    torch.manual_seed(seed)
    half   = torch.randint(1000, 50000, (1, seq_half))   # [1, k]
    tokens = torch.cat([half, half], dim=1)               # [1, 2k]
    return tokens.to(DEVICE)


def extract_all_patterns(tokens: torch.Tensor, W: dict) -> torch.Tensor:
    """
    Run forward pass, extract attn.{L}.pattern for every layer.
    Returns [L, H, S, S] tensor.
    """
    with torch.no_grad():
        _, cache = gpt2_forward(tokens, W)

    cfg = config_from_weights(W)
    S = tokens.shape[1]
    patterns = torch.zeros(cfg["n_layers"], cfg["n_heads"], S, S)
    for L in range(cfg["n_layers"]):
        patterns[L] = cache[f"attn.{L}.pattern"][0]
    return patterns


# ══════════════════════════════════════════════════════════════════════════════
# INDUCTION SCORE
# ══════════════════════════════════════════════════════════════════════════════

def induction_score(patterns: torch.Tensor, seq_half: int) -> torch.Tensor:
    """
    patterns : [L, H, 2k, 2k]
    seq_half : k

    Score(L, H) = (1/(k-1)) * Σ_{i=0}^{k-2}  A[k+i,  i+1]

    Explanation of indices:
        Destination position: k+i  (token i in the second copy)
        Source position:      i+1  (token AFTER first occurrence of t_i)

        At position k+i, current token = t_i (same as position i in first copy).
        Induction head should look at position i+1 = "what followed t_i before".
    """
    L, H, two_k, _ = patterns.shape
    k     = seq_half
    score = torch.zeros(L, H)

    for i in range(k - 1):
        dst   = k + i      # position in second copy
        src   = i + 1      # one after first occurrence
        score += patterns[:, :, dst, src]   # [L, H]

    score /= (k - 1)
    return score   # [L, H]


# ══════════════════════════════════════════════════════════════════════════════
# PREVIOUS TOKEN SCORE
# ══════════════════════════════════════════════════════════════════════════════

def prev_token_score(patterns: torch.Tensor) -> torch.Tensor:
    """
    patterns : [L, H, S, S]

    Score_PT(L, H) = mean_{i=1}^{S-1} A[i, i-1]

    Subdiagonal attention = head copies previous token's info.
    PTHs exist in early layers and FEED INTO induction heads
    via K-composition (PTH output ≈ shifted embedding in K).
    """
    L, H, S, _ = patterns.shape
    score = torch.zeros(L, H)
    for i in range(1, S):
        score += patterns[:, :, i, i - 1]   # [L, H]
    score /= (S - 1)
    return score


# ══════════════════════════════════════════════════════════════════════════════
# PRINT TOP HEADS
# ══════════════════════════════════════════════════════════════════════════════

def print_top_heads(scores: torch.Tensor, label: str, top_n: int = 10) -> None:
    L, H = scores.shape
    flat  = [(f"L{l}H{h}", scores[l, h].item())
             for l in range(L) for h in range(H)]
    top   = sorted(flat, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nTop-{top_n} {label}:")
    for name, val in top:
        bar = "█" * max(1, int(val * 40))
        print(f"  {name:<7} score={val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISE SPECIFIC HEADS
# ══════════════════════════════════════════════════════════════════════════════

def visualise_head(
    patterns: torch.Tensor,   # [L, H, S, S]
    layer: int,
    head: int,
    seq_half: int,
) -> None:
    """
    Show attention pattern of a single head.
    Mark the expected induction stripe visually.
    """
    # Simple token labels: 'a0'...'a{k-1}' repeated
    k = seq_half
    labels = [f"t{i%k}" for i in range(2 * k)]
    plot_single_head(
        patterns[layer, head],   # [S, S]
        labels,
        layer=layer, head=head,
        fname=f"induction_L{layer}H{head}.png",
    )
    print(f"  Plotted L{layer}H{head} → figures/induction_L{layer}H{head}.png")


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_induction_circuit(W: dict, tok, seq_half: int = 30,
                             ih_scores=None, pt_scores=None) -> None:
    """
    Verify induction heads by ablating top-scoring heads and measuring
    drop in P(correct next token) on repeated sequences.
    """
    print("\nAblation verification ...")
    tokens = make_repeated_tokens(seq_half=seq_half, seed=7)
    B, S   = tokens.shape
    k      = seq_half

    with torch.no_grad():
        logits_normal, _ = gpt2_forward(tokens, W)

    def avg_correct_prob(logits):
        probs = torch.softmax(logits[0], dim=-1)
        total = sum(probs[i, tokens[0, i+1].item()].item() for i in range(k, S-1))
        return total / (S - 1 - k)

    baseline_prob = avg_correct_prob(logits_normal)
    print(f"  Baseline avg P(correct next token, second copy): {baseline_prob:.4f}")

    # Auto-detect top heads to ablate from scores
    if ih_scores is not None and pt_scores is not None:
        nL, nH = ih_scores.shape
        ih_top = sorted([(l, h, ih_scores[l, h].item()) for l in range(nL) for h in range(nH)],
                        key=lambda x: x[2], reverse=True)[:3]
        pt_top = sorted([(l, h, pt_scores[l, h].item()) for l in range(nL) for h in range(nH)],
                        key=lambda x: x[2], reverse=True)[:2]
        ablate_heads = [(l, h) for l, h, _ in ih_top] + [(l, h) for l, h, _ in pt_top]
    else:
        ablate_heads = [(IH_LAYER, IH_HEAD), (PTH_LAYER, PTH_HEAD)]

    with torch.no_grad():
        _, cache_ref = gpt2_forward(tokens, W)

    patches = {
        f"attn.{l}.head.{h}.out": torch.zeros_like(cache_ref[f"attn.{l}.head.{h}.out"])
        for l, h in ablate_heads
    }

    with torch.no_grad():
        logits_ablated, _ = gpt2_forward(tokens, W, patches=patches)

    ablated_prob = avg_correct_prob(logits_ablated)
    print(f"  Ablating {ablate_heads}:")
    print(f"  Ablated  avg P(correct): {ablated_prob:.4f}")
    print(f"  Drop: {baseline_prob - ablated_prob:.4f}")
    if baseline_prob - ablated_prob > 0.01:
        print("  [CONFIRMED] Ablating these heads hurts induction.")
    else:
        print("  [?] Minimal drop. Heads may not be causal for this task.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model name (gpt2, gpt2-xl, etc). Omit for synthetic.")
    args = parser.parse_args()

    if args.model:
        W   = load_weights(args.model, device=DEVICE)
        tok = get_tokenizer(args.model)
    else:
        W   = build_synthetic_weights(device=DEVICE)
        tok = get_tokenizer()

    cfg = config_from_weights(W)
    print(f"Architecture: {cfg['n_layers']}L {cfg['n_heads']}H d={cfg['d_model']}")
    SEQ_HALF = 50

    print(f"Building repeated token sequence (k={SEQ_HALF}, total={2*SEQ_HALF}) …")
    tokens   = make_repeated_tokens(seq_half=SEQ_HALF)
    patterns = extract_all_patterns(tokens, W)   # [12, 12, 100, 100]
    print(f"Patterns shape: {patterns.shape}")

    # ── Scores ─────────────────────────────────────────────────────────────────
    ih_scores  = induction_score(patterns,    seq_half=SEQ_HALF)   # [L, H]
    pt_scores  = prev_token_score(patterns)                         # [L, H]
    torch.save(ih_scores, "figures/induction_scores.pt")
    torch.save(pt_scores, "figures/prev_token_scores.pt")

    print_top_heads(ih_scores,  "Induction Heads")
    print_top_heads(pt_scores,  "Previous Token Heads")

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    plot_induction_scores(ih_scores, fname="induction_scores.png")
    plot_induction_scores(pt_scores, fname="prev_token_scores.png")

    # ── Visualise top heads ────────────────────────────────────────────────────
    nL, nH = ih_scores.shape
    flat_ih = [(l, h, ih_scores[l, h].item()) for l in range(nL) for h in range(nH)]
    top3_ih = sorted(flat_ih, key=lambda x: x[2], reverse=True)[:3]
    print("\nVisualising top-3 induction heads …")
    for l, h, sc in top3_ih:
        visualise_head(patterns, layer=l, head=h, seq_half=SEQ_HALF)

    top3_pt = sorted([(l, h, pt_scores[l, h].item()) for l in range(nL) for h in range(nH)],
                     key=lambda x: x[2], reverse=True)[:3]
    print("Visualising top-3 previous-token heads …")
    for l, h, sc in top3_pt:
        print(f"  L{l}H{h} PT-score={sc:.4f}")

    # ── Verify circuit causally ────────────────────────────────────────────────
    verify_induction_circuit(W, tok, seq_half=30, ih_scores=ih_scores, pt_scores=pt_scores)

    print("\n03 complete. Compare your top-induction heads to literature:")
    print("  Expected: L5H1, L5H5, L6H9, L7H10, L7H11")
    print("  Expected PTH: L3H0, L4H11  (feeds into IH via K-composition)")
    print("Proceed to 04_ioi_circuit.py")
