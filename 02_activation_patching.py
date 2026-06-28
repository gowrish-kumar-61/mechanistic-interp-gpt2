"""
02 — Activation Patching on Induction Task

WHY NOT IOI?
IOI (Indirect Object Identification) requires a model that learned the task
from training data. Random/synthetic weights have no IOI circuit.
LD_clean ≈ LD_corrupted → no gap → patching recovers noise, not signal.

WHAT WE DO INSTEAD:
Use the INDUCTION TASK, which our planted IH at L5H1 DOES handle.

Protocol:
  Clean prompt:     [A B C D  A B C D]   (repeated)
  Corrupted prompt: [A B C D  X Y Z W]   (second copy replaced with different tokens)

  At position 'A' (second copy), the induction head attends to position 'B'
  (what followed 'A' in the first copy). This boosts P(B).

  Metric: LD = logit(t_{i+1}) − logit(t_{i+1}^corrupted)
          where t_{i+1} = next token in clean (what IH predicts)
                t_{i+1}^corrupted = what would be predicted without induction

  Causal patching:
    Patch attn.L.head.H.out from clean cache into corrupted run.
    Recovery = how much does patching this head restore clean prediction?

  If L=5, H=1 shows high recovery → IH is the causal component.

ACTIVATION PATCHING MATH:
Let:
  LD_clean     = metric on clean run         (induction working)
  LD_corrupted = metric on corrupted run     (induction broken)
  LD_patched   = metric when head(L,H) from clean is spliced in

  Recovery(L,H) = (LD_patched - LD_corr) / (LD_clean - LD_corr)
    = 1.0 → this head ALONE restores induction (causal component!)
    = 0.0 → patching this head does nothing
    < 0.0 → this head makes things worse (inhibitory)

Run:
    python 02_activation_patching.py
"""

import sys, torch, argparse
sys.path.insert(0, ".")
from core.model import gpt2_forward, config_from_weights, load_weights, get_tokenizer
from core.tokenizer import tokenize
from core.synthetic_weights import build_synthetic_weights, PTH_LAYER, PTH_HEAD, IH_LAYER, IH_HEAD
from core.metrics import patching_metric
from core. visualize import plot_patching_heatmap
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_HALF = 8  # small k for speed (total seq = 16)

# INDUCTION PATCHING SETUP
def make_induction_prompts(tok, k=SEQ_HALF, seed=42, device="cpu"):
    """
    Clean:     [t0 t1 t2 ... t_{k-1}  t0 t1 t2 ... t_{k-1}]
    Corrupted: [t0 t1 t2 ... t_{k-1}  s0 s1 s2 ... s_{k-1}]  (different second half)

    Uses actual token IDs from our vocabulary for reproducibility.
    """
    torch.manual_seed(seed)
    # Use high-freq IDs that are in our vocab (stable across runs)
    # Sample from tokens that exist in our fixed vocab
    good_ids = [262, 290, 284, 257, 11, 198, 1601, 1757, 5576, 2921,
                1816, 3650, 1492, 3241, 2116, 1524, 5390, 5765, 3362, 4186]
    # First half
    half_a = torch.tensor([good_ids[i % len(good_ids)] for i in range(k)]).unsqueeze(0)
    # Corrupted second half (different tokens)
    half_b = torch.tensor([good_ids[(i + k) % len(good_ids)] for i in range(k)]).unsqueeze(0)

    clean_ids = torch.cat([half_a, half_a], dim=1)   # [1, 2k]
    corr_ids  = torch.cat([half_a, half_b], dim=1)   # [1, 2k]

    tokens_clean = [tok.decode([t]) for t in clean_ids[0].tolist()]
    tokens_corr  = [tok.decode([t]) for t in corr_ids[0].tolist()]

    print(f"  Seq length: {2*k}  (k={k} repeated)")
    print(f"  Clean[0:k]:      {clean_ids[0, :k].tolist()}")
    print(f"  Clean[k:2k]:     {clean_ids[0, k:].tolist()}")
    print(f"  Corrupted[k:2k]: {corr_ids[0, k:].tolist()}")

    return clean_ids.to(device), corr_ids.to(device), good_ids


def induction_ld(logits: torch.Tensor, clean_ids: torch.Tensor, k: int, batch=0) -> float:
    """
    Induction logit difference at positions k..2k-2.

    At position k+i (second occurrence of t_i), the IH should boost t_{i+1}.
    LD = logit(t_{i+1}) at position k+i, averaged over i=0..k-2.

    Higher = the model more strongly predicts the "inductively correct" next token.
    """
    total = 0.0
    count = 0
    for i in range(k - 1):
        dst_pos = k + i                          # query position
        target_id = clean_ids[batch, i + 1].item()  # inductively correct answer
        lo = logits[batch, dst_pos, :]           # [V]
        total += lo[target_id].item()
        count += 1
    return total / max(count, 1)

# BASELINES
def measure_baselines(clean_ids, corr_ids, W, k):
    with  torch.no_grad():
        logits_clean, cache_clean = gpt2_forward(clean_ids, W)
        logits_corr,  cache_corr  = gpt2_forward(corr_ids,  W)

    ld_clean = induction_ld(logits_clean, clean_ids, k)
    ld_corr  = induction_ld(logits_corr,  clean_ids, k)   # metric vs clean targets

    print(f"\n  LD (clean,  inductive positions): {ld_clean:+.4f}")
    print(f"  LD (corrupted, same positions):   {ld_corr:+.4f}")
    print(f"  Gap (clean − corr):                {ld_clean - ld_corr:+.4f}")
    print()
    if ld_clean > ld_corr:
        print(f"  ✓ Planted IH boosts correct token in clean run (+{ld_clean-ld_corr:.4f})")
    else:
        print(f"  ✗ No gap (IH not helping — check circuit_strength in synthetic_weights.py)")

    return ld_clean, ld_corr, cache_clean, cache_corr

# HEAD PATCHING SWEEP
def run_head_patching(clean_ids, corr_ids, W, ld_clean, ld_corr, cache_clean, k,
                      n_layers=12, n_heads=12):
    """
    Sweep all 144 (layer, head) pairs.
    Patch attn.L.head.H.out from CLEAN into CORRUPTED run.
    Measure recovery of induction LD.
    Returns [L, H] recovery tensor.
    """
    recovery = torch.zeros(n_layers, n_heads)
    print("Patching heads (clean → corrupted)  …")
    for L in tqdm(range(n_layers), desc="Layer"):
        for H in range(n_heads):
            hook_name = f"attn.{L}.head.{H}.out"
            patches   = {hook_name: cache_clean[hook_name]}
            with torch.no_grad():
                logits_p, _ = gpt2_forward(corr_ids, W, patches=patches)
            ld_p = induction_ld(logits_p, clean_ids, k)
            recovery[L, H] = patching_metric(ld_p, ld_clean, ld_corr)
    return recovery

# RESIDUAL STREAM PATCHING (LAYER-LEVEL)
def run_resid_patching(clean_ids, corr_ids, W, ld_clean, ld_corr, cache_clean, k):
    """
    Patch entire residual stream at each point:
        resid_pre.L  (before attention at layer L)
        resid_mid.L  (after attn, before MLP)
        resid_post.L (after MLP)

    This shows WHICH LAYER information flows through to achieve induction.
    """
    results = {"pre": [], "mid": [], "post": []}
    print("Patching residual stream …")
    n_layers = config_from_weights(W)["n_layers"]
    for L in tqdm(range(n_layers), desc="Layer"):
        for key, name in [("pre", f"resid_pre.{L}"),
                          ("mid", f"resid_mid.{L}"),
                          ("post", f"resid_post.{L}")]:
            patches = {name: cache_clean[name]}
            with torch.no_grad():
                logits_p, _ = gpt2_forward(corr_ids, W, patches=patches)
            ld_p = induction_ld(logits_p, clean_ids, k)
            results[key].append(patching_metric(ld_p, ld_clean, ld_corr))
    return results

# MAIN
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
    print("─" * 60)
    print(f"INDUCTION TASK — ACTIVATION PATCHING ({cfg['n_layers']}L {cfg['n_heads']}H)")
    if not args.model:
        print(f"Planted IH: L{IH_LAYER}H{IH_HEAD}  |  Planted PTH: L{PTH_LAYER}H{PTH_HEAD}")
    print("─" * 60)

    clean_ids, corr_ids, good_ids = make_induction_prompts(tok, k=SEQ_HALF, device=DEVICE)

    ld_clean, ld_corr, cache_clean, cache_corr = measure_baselines(
        clean_ids, corr_ids, W, SEQ_HALF
    )

    if abs(ld_clean - ld_corr) < 1e-6:
        print("WARNING: LD gap is zero — model sees no difference between clean/corrupted")
        print("This can happen when IH output is diluted by W_O before affecting logits.")
        print("Proceeding with sweep anyway to show patching infrastructure works.")
        print()

    # Head-level patching
    recovery = run_head_patching(
        clean_ids, corr_ids, W,
        ld_clean, ld_corr, cache_clean, SEQ_HALF,
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
    )
    torch.save(recovery, "figures/recovery_heads.pt")

    nL, nH = recovery.shape
    flat  = [(f"L{l}H{h}", recovery[l, h].item()) for l in range(nL) for h in range(nH)]
    top10 = sorted(flat, key=lambda x: abs(x[1]), reverse=True)[:10]
    print("\nTop-10 heads by |recovery|:")
    for name, val in top10:
        mark = " ← PLANTED IH" if name == f"L{IH_LAYER}H{IH_HEAD}" else ""
        mark = " ← PLANTED PTH" if name == f"L{PTH_LAYER}H{PTH_HEAD}" else mark
        bar = "█" * max(1, int(abs(val) * 30))
        print(f"  {name:<7} {val:+.4f}  {bar}{mark}")

    # Heatmap
    plot_patching_heatmap(
        recovery,
        title=f"Head Output Patching Recovery\n(clean→corrupted, induction task)\nPlanted: L{IH_LAYER}H{IH_HEAD}=IH, L{PTH_LAYER}H{PTH_HEAD}=PTH",
        fname="patching_heads.png",
    )

    # Residual stream patching
    resid = run_resid_patching(
        clean_ids, corr_ids, W, ld_clean, ld_corr, cache_clean, SEQ_HALF
    )
    print("\nResidual stream patching recovery:")
    print("  Layer:  " + "  ".join(f"L{i:2d}" for i in range(cfg["n_layers"])))
    for key, vals in resid.items():
        print(f"  {key:5s}: " + "  ".join(f"{v:+.2f}" for v in vals))
    print()

    # Manual verification: directly check L5H1 recovery
    print(f"\nDirect test: patch ONLY L{IH_LAYER}H{IH_HEAD} (planted IH):")
    hook = f"attn.{IH_LAYER}.head.{IH_HEAD}.out"
    with torch.no_grad():
        logits_patched_ih, _ = gpt2_forward(corr_ids, W, patches={hook: cache_clean[hook]})
    ld_patched_ih = induction_ld(logits_patched_ih, clean_ids, SEQ_HALF)
    rec_ih = patching_metric(ld_patched_ih, ld_clean, ld_corr)
    print(f"  LD (patched L{IH_LAYER}H{IH_HEAD}): {ld_patched_ih:+.4f}")
    print(f"  Recovery:                    {rec_ih:+.4f}")

    print(f"\nDirect test: patch ONLY L{PTH_LAYER}H{PTH_HEAD} (planted PTH):")
    hook = f"attn.{PTH_LAYER}.head.{PTH_HEAD}.out"
    with torch.no_grad():
        logits_patched_pth, _ = gpt2_forward(corr_ids, W, patches={hook: cache_clean[hook]})
    ld_patched_pth = induction_ld(logits_patched_pth, clean_ids, SEQ_HALF)
    rec_pth = patching_metric(ld_patched_pth, ld_clean, ld_corr)
    print(f"  LD (patched L{PTH_LAYER}H{PTH_HEAD}): {ld_patched_pth:+.4f}")
    print(f"  Recovery:                    {rec_pth:+.4f}")

    print("\n02 complete. Proceed → python 03_induction_heads.py")
