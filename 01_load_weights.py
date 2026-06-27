"""
01 — Load GPT-2 weights (synthetic with planted circuits).
     Inspect every tensor shape.  Verify forward pass runs.

NOTE ON WEIGHTS
Real GPT-2 requires HuggingFace (network blocked in this env).
We use synthetic GPT-2 weights with PLANTED circuits:
  • PTH planted at L3H0  (previous token head)
  • IH  planted at L5H1  (induction head)
Detection pipeline in 03 should find EXACTLY these two heads.
This is ground-truth validation — more rigorous than post-hoc checking.

On your own machine: replace load_weights() call with:
    from core.model import load_weights  # uses HuggingFace

Expected:
  ✓ All shapes match GPT-2 Small spec
  ✓ Forward pass runs without error
  ✓ Cache contains all intermediate activations
  ✓ Top-k predictions are garbage (random weights) — expected
"""

import sys, math, torch, argparse
sys.path.insert(0, ".")
from core.model import gpt2_forward, get_tokenizer, tokenize, load_weights, config_from_weights
from core.synthetic_weights import (
    build_synthetic_weights, print_weight_table,
    total_params, PTH_LAYER, PTH_HEAD, IH_LAYER, IH_HEAD
)
from core.metrics import top_k_logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# 1. ARCHITECTURE MATH

def print_arch_math():
    d_model, n_heads, d_head = 768, 12, 64
    d_mlp, n_layers, vocab   = 3072, 12, 50257

    print("GPT-2 Small — Architecture Math")
    print("-" * 45)
    print(f"  d_model        = {d_model}")
    print(f"  n_heads        = {n_heads}")
    print(f"  d_head         = d_model/n_heads = {d_model}/{n_heads} = {d_head}")
    print(f"  d_mlp          = 4 × d_model = {d_mlp}")
    print(f"  n_layers       = {n_layers}")
    print(f"  vocab          = {vocab}")
    print()

    # Memory footprint
    params = sum([
        vocab * d_model,            # wte
        1024 * d_model,             # wpe
        n_layers * (
            3 * d_model * d_model + # QKV
            d_model * d_model +     # c_proj attn
            d_model * d_mlp +       # fc
            d_mlp * d_model         # mlp proj
        ),
    ])
    print(f"  Param count (approx):   {params/1e6:.1f}M")
    print(f"  Memory FP32:            {params*4/1024**2:.0f} MB")
    print(f"  Memory FP16:            {params*2/1024**2:.0f} MB")
    print()

    # Residual stream at each position is 768-dim vector
    # Each layer reads from it (via LN) and writes ADDITIVE updates
    # Key insight: superposition — multiple features share same 768-dim space
    print("  Residual stream = 768-dim information highway")
    print("  Each layer writes: Δx = Attn_out + MLP_out")
    print("  Attention reads via Q,K,V; MLP is a lookup table")
    print()

# 2. LOAD AND INSPECT WEIGHTS

def inspect_weights(W):
    print(f"Planted circuits:")
    print(f"  L{PTH_LAYER}H{PTH_HEAD} → Previous Token Head (subdiagonal attention)")
    print(f"  L{IH_LAYER}H{IH_HEAD}  → Induction Head    (attends to pos after first occurrence)")
    print()
    print_weight_table(W)

# 3. FORWARD PASS VERIFICATION

def verify_forward(W, tok=None):
    tok  = tok or get_tokenizer()
    text = "The transformer architecture uses self-attention to"
    ids  = tokenize(text, tok, device=DEVICE)

    print(f"Test: '{text}'")
    print(f"Tokens: {tok.convert_ids_to_tokens(ids[0].tolist())}")
    print()

    with torch.no_grad():
        logits, cache = gpt2_forward(ids, W)

    print(f"  Output logits shape: {logits.shape}")    # [1, S, 50257]
    print(f"  Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
    print()

    print("  Top-5 next-token predictions (random weights → garbage, expected):")
    for tid, val in top_k_logits(logits, k=5):
        word = tok.decode([tid])
        print(f"    '{word}' (id={tid:5d})  logit={val:.3f}")
    print()

    return cache, ids

# 4. CACHE INSPECTION — ALL INTERMEDIATE SHAPES

def inspect_cache(cache, ids, n_layers=12):
    S = ids.shape[1]
    print(f"Cache shapes (seq_len={S}):")
    show = [
        "embed",
        "pos_embed",
        "resid_pre.0",
        "attn.0.q",
        "attn.0.pattern",
        "attn.0.z",
        f"resid_post.{n_layers - 1}",
        "logits",
    ]
    for key in show:
        if key in cache:
            print(f"  {key:<40} {str(tuple(cache[key].shape))}")
    print(f"\n  Total cached tensors: {len(cache)}")

# 5. VERIFY PTH PATTERN ON SIMPLE SEQUENCE

def quick_pth_check(W):
    """
    On a simple sequence [t0 t1 t2 t3 t4], L3H0 should attend each position
    to the previous one: A[i, i-1] should be the largest score.
    """
    tok = get_tokenizer()
    text = "Hello world foo bar baz"
    ids  = tokenize(text, tok, DEVICE)

    with torch.no_grad():
        _, cache = gpt2_forward(ids, W)

    pattern = cache[f"attn.{PTH_LAYER}.pattern"][0, PTH_HEAD]  # [S, S]
    S = pattern.shape[0]
    tokens = tok.convert_ids_to_tokens(ids[0].tolist())

    print(f"\nPTH (L{PTH_LAYER}H{PTH_HEAD}) attention pattern on '{text}':")
    print(f"  {'':12}" + "".join(f"{t:>10}" for t in tokens))
    for i in range(1, S):
        row = pattern[i].tolist()
        prev_rank = sorted(range(len(row)), key=lambda x: row[x], reverse=True).index(i-1)
        bar = "█" * int(row[i-1] * 30)
        print(f"  {tokens[i]:>12}  A[{i},{i-1}]={row[i-1]:.3f} rank={prev_rank+1}  {bar}")

    # Subdiagonal score (= prev token score for PTH)
    pt_score = sum(pattern[i, i-1].item() for i in range(1, S)) / (S - 1)
    print(f"\n  Previous-token score at L{PTH_LAYER}H{PTH_HEAD}: {pt_score:.4f}")
    if pt_score > 0.3:
        print(f"  ✓ PTH planted successfully — strong subdiagonal attention")
    else:
        print(f"  ⚠ PT score lower than expected ({pt_score:.4f}). Check circuit strength.")

# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl). Omit for synthetic.")
    args = parser.parse_args()

    if args.model:
        print(f"Loading {args.model} from HuggingFace …")
        W = load_weights(args.model, device=DEVICE)
        tok = get_tokenizer(args.model)
    else:
        print("Building synthetic GPT-2 weights …")
        W = build_synthetic_weights(device=DEVICE)
        tok = get_tokenizer()
        inspect_weights(W)

    cfg = config_from_weights(W)
    print(f"\nArchitecture: {cfg['n_layers']}L {cfg['n_heads']}H d_model={cfg['d_model']} d_mlp={cfg['d_mlp']}")
    print_weight_table(W)

    if args.model:
        cache, ids = verify_forward(W, tok)
    else:
        cache, ids = verify_forward(W)
    inspect_cache(cache, ids, n_layers=cfg["n_layers"])

    if not args.model:
        quick_pth_check(W)

    print("\n01 complete.")
