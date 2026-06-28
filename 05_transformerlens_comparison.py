"""
05 — TransformerLens API Comparison

TransformerLens (TL) is blocked in this sandbox (needs HuggingFace).
This script serves TWO purposes:

  A. REFERENCE: Show exactly how our manual implementation maps to TL API.
     Every concept is the same; only naming and syntax differ.

  B. VERIFICATION TEST: Use our manual impl to produce the SAME outputs
     TL would produce, then document what to check when you run real TL
     on your own machine (with pip install transformer_lens).

TL vs Manual — Key Differences:
  1. Hook names:
       Manual:  cache["attn.5.pattern"]       → [B, H, S, S]
       TL:      cache["pattern", 5]           → [B, H, S, S]  (via ActivationCache)

  2. z tensor axis:
       Manual:  cache["attn.5.z"]    → [B, H, S, Dh]   (H=axis 1)
       TL:      cache["z", 5]        → [B, S, H, Dh]   (H=axis 2!)
       → GOTCHA: .permute(0,2,1,3) needed to convert between them

  3. Weight centering:
       TL default: center_writing_weights=True
       Folds LayerNorm into weights → logit deviations ≈ 0.01 vs raw HF
       Our manual: no centering → exact match with HuggingFace

  4. Patching API:
       Manual:  gpt2_forward(tokens, W, patches={"attn.5.head.1.out": clean_val})
       TL:      model.run_with_hooks(tokens, fwd_hooks=[("z", partial(patch_fn, layer=5))])

  5. OV circuit:
       Manual:  OV_h = W_qkv[:, 2*D+h*Dh:2*D+(h+1)*Dh] @ W_o[h*Dh:(h+1)*Dh, :]
       TL:      model.OV[layer][head]   → FactoredMatrix object

  6. Weight key prefix:
       HuggingFace:  "transformer.h.5.attn.c_attn.weight"
       TL (loaded):  "blocks.5.attn.W_Q"  (separate Q,K,V matrices, transposed)

Run this ON YOUR OWN MACHINE (with HF access):
    pip install transformer_lens
    python 05_transformerlens_comparison.py --real

When --real flag is set: loads actual GPT-2 via TL, runs both implementations,
computes deviation tables.
"""

import sys, torch, math
import torch.nn.functional as F
sys.path.insert(0, ".")
from core.model import gpt2_forward, config_from_weights
from core.tokenizer import get_tokenizer, tokenize
from core.synthetic_weights import (
    build_synthetic_weights, PTH_LAYER, PTH_HEAD, IH_LAYER, IH_HEAD
)
from core.visualize import plot_induction_scores

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_REAL_TL = "--real" in sys.argv

# SECTION A: API MAPPING REFERENCE
def print_api_mapping():
    print("=" * 65)
    print("  TransformerLens ↔ Manual Implementation API Mapping")
    print("=" * 65)

    mapping = [
        ("LOADING MODEL",
         "GPT2LMHeadModel.from_pretrained('gpt2')\n" +
         "    + manual weight dict",
         "HookedTransformer.from_pretrained('gpt2')"),
        ("FORWARD PASS",
         "logits, cache = gpt2_forward(tokens, W)",
         "logits, cache = model.run_with_cache(tokens)"),
        ("ATTN PATTERN L5",
         "cache['attn.5.pattern'][0]  → [H,S,S]",
         "cache['pattern', 5][0]      → [H,S,S]"),
        ("Z VALUES L5",
         "cache['attn.5.z'][0]        → [H,S,Dh]  H=axis0",
         "cache['z', 5][0]            → [S,H,Dh]  H=axis1 ←!"),
        ("RESIDUAL PRE L5",
         "cache['resid_pre.5']",
         "cache['resid_pre', 5]"),
        ("PATCHING",
         "patches = {'attn.5.head.1.out': clean_val}\n" +
         "    gpt2_forward(tokens, W, patches=patches)",
         "model.run_with_hooks(\n" +
         "    tokens,\n" +
         "    fwd_hooks=[('z', partial(patcher, layer=5, head=1))])"),
        ("OV MATRIX",
         "W_V_h @ W_O_h (manual slice)",
         "model.OV[layer][head]  → FactoredMatrix(W_V, W_O)"),
        ("QK MATRIX",
         "W_Q_h.T @ W_K_h (manual)",
         "model.QK[layer][head]  → FactoredMatrix(W_Q, W_K)"),
        ("WEIGHT KEYS",
         "'transformer.h.5.attn.c_attn.weight' [768,2304]",
         "'blocks.5.attn.W_Q' [H,D,Dh] (split + transposed)"),
    ]

    for section, manual, tl in mapping:
        print(f"\n  ─── {section} ───")
        print(f"  Manual: {manual}")
        print(f"  TL:     {tl}")
    print()
  
# SECTION B: INDUCTION SCORE (OUR IMPL vs WHAT TL WOULD GIVE)
def run_manual_induction(W, tok, k=40):
    """
    Run induction score detection on our planted model.
    Document exactly what TL would compute (same math, different API).
    """
    torch.manual_seed(42)
    good_ids = [262, 290, 284, 257, 11, 198, 1601, 1757,
                5576, 2921, 1816, 3650, 1492, 3241, 2116, 1524,
                5390, 5765, 3362, 4186, 1649, 3241, 2116, 7576,
                5553, 39432, 10959, 3544, 2116, 3241, 1757, 1601,
                284, 262, 290, 257, 3650, 2921, 5576, 1816]
    half = torch.tensor([good_ids[i % len(good_ids)] for i in range(k)]).unsqueeze(0)
    tokens = torch.cat([half, half], dim=1).to(DEVICE)   # [1, 2k]

    with torch.no_grad():
        _, cache = gpt2_forward(tokens, W)

    cfg = config_from_weights(W)
    scores = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    for L in range(cfg["n_layers"]):
        pat = cache[f"attn.{L}.pattern"][0]   # [H, 2k, 2k]
        for i in range(k - 1):
            dst = k + i
            src = i + 1
            scores[L] += pat[:, dst, src]    # [H]
    scores /= (k - 1)
    return scores, tokens

def run_manual_pt_score(W, k=40):
    """Previous token scores — same as 03 but shown in TL-comparison context."""
    torch.manual_seed(42)
    good_ids = [262, 290, 284, 257, 11, 198, 1601, 1757,
                5576, 2921, 1816, 3650, 1492, 3241, 2116, 1524,
                5390, 5765, 3362, 4186, 1649, 3241, 2116, 7576,
                5553, 39432, 10959, 3544, 2116, 3241, 1757, 1601,
                284, 262, 290, 257, 3650, 2921, 5576, 1816]
    half   = torch.tensor([good_ids[i % len(good_ids)] for i in range(k)]).unsqueeze(0)
    tokens = torch.cat([half, half], dim=1).to(DEVICE)
    S = tokens.shape[1]

    with torch.no_grad():
        _, cache = gpt2_forward(tokens, W)

    cfg = config_from_weights(W)
    pt_scores = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    for L in range(cfg["n_layers"]):
        pat = cache[f"attn.{L}.pattern"][0]   # [H, S, S]
        for i in range(1, S):
            pt_scores[L] += pat[:, i, i - 1]
    pt_scores /= (S - 1)
    return pt_scores

# SECTION C: WHAT TO VERIFY ON REAL MACHINE
def print_real_machine_checklist():
    print("=" * 65)
    print("  Checklist: What to verify on your own machine (real GPT-2)")
    print("=" * 65)

    checks = [
        ("01 — Forward pass",
         [
             "max_logit_deviation < 5e-4  (manual vs HF)",
             "cache has 291 tensors (12 layers × residual/attn/mlp × positions)",
             "wte shape = (50257, 768), wpe shape = (1024, 768)",
         ]),
        ("02 — Activation patching (real IOI)",
         [
             "LD clean ≈ +2.5 to +4.0  (model strongly predicts IO)",
             "LD corrupted ≈ -2.5 to -4.0  (after swap, predicts S)",
             "Top recovery heads include L9H6, L9H9, L10H0 (Name Movers)",
             "Resid patching: recovery jumps at L8-L9 (Name Movers write here)",
         ]),
        ("03 — Induction heads (real GPT-2)",
         [
             "L5H1 induction score ≈ 0.94  (top head)",
             "L5H5 induction score ≈ 0.87",
             "L6H9 induction score ≈ 0.82",
             "L3H0 prev-token score ≈ 0.80+  (top PTH)",
             "L4H11 prev-token score ≈ 0.7+",
             "Ablating L5H1+L5H5+L6H9 drops P(correct) on repeated seq by ~0.3",
         ]),
        ("04 — IOI circuit (real GPT-2)",
         [
             "NMH ablation: LD drop > 1.5 (ablating L9H6+L9H9+L10H0)",
             "S-inhib ablation: LD changes direction (±0.5+)",
             "L9H9 OV cosine similarity ≈ 0.35-0.45 (strong copy)",
             "L9H6 OV cosine similarity ≈ 0.30-0.40",
             "L3H0 pattern: strong subdiagonal (PTH detection)",
         ]),
        ("05 — TL comparison",
         [
             "TL induction scores match manual within 0.02 (same math)",
             "TL logits match manual within 0.01 (center_writing_weights effect)",
             "cache['pattern', 5] shape = cache['attn.5.pattern'] shape",
             "cache['z', 5] is TRANSPOSED vs manual (S,H,Dh vs H,S,Dh)",
         ]),
    ]

    for title, items in checks:
        print(f"\n  {title}:")
        for item in items:
            print(f"    □ {item}")
    print()


# SECTION D: OPTIONAL — REAL TL (if --real flag + HF access)
def run_real_tl_comparison(tok):
    """
    Requires: pip install transformer_lens
              HuggingFace access to download gpt2

    Compares:
      1. Our manual induction scores vs TL scores
      2. Logit deviation: manual vs HF vs TL
      3. Attention pattern at L5H1 (planted in synthetic, real in gpt2)
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        print("  transformer_lens not installed. pip install transformer_lens")
        return

    print("Loading real GPT-2 via TransformerLens …")
    model_tl = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
    model_tl.eval()

    # Induction score comparison
    torch.manual_seed(42)
    half   = torch.randint(1000, 50000, (1, 40))
    tokens = torch.cat([half, half], dim=1).to(DEVICE)
    k      = 40

    # TL forward
    _, cache_tl = model_tl.run_with_cache(
        tokens,
        names_filter=lambda n: "pattern" in n,
    )
    n_layers = model_tl.cfg.n_layers
    n_heads = model_tl.cfg.n_heads
    scores_tl = torch.zeros(n_layers, n_heads)
    for L in range(n_layers):
        pat = cache_tl["pattern", L][0]
        for i in range(k - 1):
            scores_tl[L] += pat[:, k + i, i + 1]
    scores_tl /= (k - 1)

    from core.model import load_weights
    W_real = load_weights(device=DEVICE)
    cfg = config_from_weights(W_real)
    with torch.no_grad():
        _, cache_m = gpt2_forward(tokens, W_real)
    scores_manual = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    for L in range(cfg["n_layers"]):
        pat = cache_m[f"attn.{L}.pattern"][0]
        for i in range(k - 1):
            scores_manual[L] += pat[:, k + i, i + 1]
    scores_manual /= (k - 1)

    dev = (scores_manual - scores_tl).abs()
    print(f"\n  Induction score deviation (manual vs TL):")
    print(f"    Max: {dev.max():.6f}")
    print(f"    Mean: {dev.mean():.6f}")
    print(f"  ✓ Match confirmed." if dev.max() < 0.02 else f"  ✗ Mismatch — check center_writing_weights")

    top_manual = sorted(
        [(f"L{l}H{h}", scores_manual[l, h].item()) for l in range(scores_manual.shape[0]) for h in range(scores_manual.shape[1])],
        key=lambda x: x[1], reverse=True
    )[:5]
    top_tl = sorted(
        [(f"L{l}H{h}", scores_tl[l, h].item()) for l in range(scores_tl.shape[0]) for h in range(scores_tl.shape[1])],
        key=lambda x: x[1], reverse=True
    )[:5]

    print(f"\n  Top-5 induction heads (manual): {top_manual}")
    print(f"  Top-5 induction heads (TL):     {top_tl}")
  
# MAIN
if __name__ == "__main__":
    tok = get_tokenizer()
    W   = build_synthetic_weights(device=DEVICE)

    print_api_mapping()

    print("Running induction detection (manual, synthetic weights) …")
    ih_scores, tokens = run_manual_induction(W, tok, k=40)
    pt_scores         = run_manual_pt_score(W, k=40)

    # Results table
    flat_ih = sorted(
        [(f"L{l}H{h}", ih_scores[l, h].item()) for l in range(ih_scores.shape[0]) for h in range(ih_scores.shape[1])],
        key=lambda x: x[1], reverse=True
    )
    flat_pt = sorted(
        [(f"L{l}H{h}", pt_scores[l, h].item()) for l in range(pt_scores.shape[0]) for h in range(pt_scores.shape[1])],
        key=lambda x: x[1], reverse=True
    )

    print("\n  Manual results (synthetic weights with planted circuits):")
    print(f"  {'IH score':>40}  {'PT score'}")
    print("  " + "-"*55)
    for i in range(5):
        ih_name, ih_val = flat_ih[i]
        pt_name, pt_val = flat_pt[i]
        ih_mark = " ← PLANTED" if ih_name == f"L{IH_LAYER}H{IH_HEAD}" else ""
        pt_mark = " ← PLANTED" if pt_name == f"L{PTH_LAYER}H{PTH_HEAD}" else ""
        print(f"  {i+1}. {ih_name} {ih_val:.4f}{ih_mark:<20}   "
              f"{pt_name} {pt_val:.4f}{pt_mark}")

    # Save heatmaps
    plot_induction_scores(ih_scores, fname="induction_scores_manual.png")
    plot_induction_scores(pt_scores, fname="prev_token_scores_manual.png")

    # Axis difference demo
    print("\n  CRITICAL GOTCHA: TL z-tensor axis order")
    print("  Manual: cache['attn.5.z'] → [B, H, S, Dh]  (H = axis 1)")
    print("  TL:     cache['z', 5]     → [B, S, H, Dh]  (H = axis 2)")
    print("  Convert: tl_z.permute(0, 2, 1, 3) → [B, H, S, Dh]")
    print()

    # Verify numerically our manual z is [B,H,S,Dh]
    cfg = config_from_weights(W)
    with torch.no_grad():
        _, cache = gpt2_forward(tokens, W)
    z5 = cache["attn.5.z"]
    S = tokens.shape[1]
    expected = (1, cfg["n_heads"], S, cfg["d_head"])
    print(f"  Our cache['attn.5.z'] shape: {tuple(z5.shape)}")
    print(f"  Expected [B=1, H={cfg['n_heads']}, S={S}, Dh={cfg['d_head']}]: {tuple(z5.shape) == expected}")

    print_real_machine_checklist()

    if USE_REAL_TL:
        print("─" * 65)
        print("REAL TL COMPARISON (--real flag detected)")
        print("─" * 65)
        run_real_tl_comparison(tok)
    else:
        print("Run with --real flag on your machine (with HF access + TL installed)")
        print("  python 05_transformerlens_comparison.py --real")

    print("\n05 complete.")
    print("\nNEXT STEPS:")
    print("  1. python 01_load_weights.py --model gpt2-xl")
    print("  2. python 03_induction_heads.py --model gpt2-xl")
    print("  3. python 04_ioi_circuit.py --model gpt2-xl")
    print("  4. python 05_transformerlens_comparison.py --real")
