"""
04 — IOI Circuit Dissection

Wang et al. (2022) "Interpretability in the Wild: a Circuit for IOI in GPT-2 small"
https://arxiv.org/abs/2211.00593

Full circuit (simplified):

   Layers 0-5: DUPLICATE TOKEN HEADS
         Detect that S appears twice.
         Write "S is duplicated" signal into residual at S2 position.
         Known heads: L0H1, L0H10, L1H8, L3H0.
   Layers 5-7:  INDUCTION HEADS (repurposed for IOI)
         Help identify S2 via induction-like mechanism.
   Layers 7-8:  S-INHIBITION HEADS
         Read "S is duplicated" from residual.
         Suppress writing S to output.
         Known heads: L7H3, L7H9, L8H6, L8H10.
   Layers 9-10: NAME MOVER HEADS
         Copy IO token's embedding to final position.
         Inhibited from copying S (by S-inhibition heads).
         Known heads: L9H6, L9H9, L10H0.
   Layers 9-10: BACKUP NAME MOVERS
         Redundant copies of NMH. Activate when NMH is ablated.
         Known: L9H1, L10H6, L10H10.

This script:
    A. Verifies Name Mover Heads — ablate → LD drops
    B. Verifies S-Inhibition Heads — ablate → NMH now copies S (LD drops further)
    C. OV Circuit analysis — W_V @ W_O for each head → what does it "copy"?
    D. Attention pattern analysis — what token does each head attend to?

Run:
    python 04_ioi_circuit.py
"""

import sys, torch, argparse
sys.path.insert(0, ".")
from core.model import gpt2_forward, config_from_weights, load_weights, get_tokenizer
from core.tokenizer import tokenize
from core.synthetic_weights import build_synthetic_weights, PTH_LAYER, PTH_HEAD, IH_LAYER, IH_HEAD
from core.metrics import logit_diff
from core.visualize import plot_single_head, plot_logit_diff_by_head, plot_patching_heatmap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PROMPT SETUP
CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"
CORR_PROMPT  = "When John and Mary went to the store, Mary gave a drink to"
IO_NAME      = " Mary"
S_NAME       = " John"

def get_token_positions(prompt: str, tok) -> dict:
    """
    Find positions of key tokens in the IOI prompt.
    Returns dict: token_name → [list of positions]
    """
    tokens = tok.encode(prompt)
    strs   = tok.convert_ids_to_tokens(tokens)
    io_id  = tok.encode(IO_NAME)[0]
    s_id   = tok.encode(S_NAME)[0]

    io_pos = [i for i, t in enumerate(tokens) if t == io_id]
    s_pos  = [i for i, t in enumerate(tokens) if t == s_id]
    end_pos = len(tokens) - 1  # "to" position — model predicts from here

    print("  Token breakdown:")
    for i, (tid, s) in enumerate(zip(tokens, strs)):
        tag = ""
        if tid == io_id:  tag = " ← IO"
        if tid == s_id:   tag = " ← S"
        if i == end_pos:  tag = " ← FINAL (predict here)"
        print(f"    [{i:2d}] '{s}' (id={tid}){tag}")

    return {"io": io_pos, "s": s_pos, "end": end_pos}

# HEAD ABLATION SWEEP
def ablate_head(
    tokens:   torch.Tensor,
    layer:    int,
    head:     int,
    W:        dict,
    cache_ref: dict,
    mode:     str = "zero",   # "zero" or "mean"
) -> torch.Tensor:
    """
    Ablate head (L, H) by replacing its output with zeros (or mean).
    Returns patched logits.

    Ablation = causally remove a component.
    If LD drops when we ablate a head → that head contributes positively to IOI.
    If LD rises when we ablate a head → that head was suppressing the answer!
    """
    key = f"attn.{layer}.head.{head}.out"
    if mode == "zero":
        patch_val = torch.zeros_like(cache_ref[key])
    else:
        patch_val = cache_ref[key].mean(dim=1, keepdim=True).expand_as(cache_ref[key])

    with torch.no_grad():
        logits, _ = gpt2_forward(tokens, W, patches={key: patch_val})
    return logits

def sweep_ablations(
    tokens:    torch.Tensor,
    io_token:  int,
    s_token:   int,
    W:         dict,
    ld_base:   float,
) -> torch.Tensor:
    """
    For each (L, H): ablate → measure ΔLD = LD_base − LD_ablated.
    Positive delta = head was HELPFUL (ablating hurts).
    Negative delta = head was HARMFUL (ablating helps). → Inhibitor heads!
    """
    with torch.no_grad():
        _, cache_ref = gpt2_forward(tokens, W)

    cfg = config_from_weights(W)
    deltas = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    print("Sweeping ablations …")
    for L in range(cfg["n_layers"]):
        for H in range(cfg["n_heads"]):
            logits_abl = ablate_head(tokens, L, H, W, cache_ref)
            ld_abl     = logit_diff(logits_abl, io_token, s_token).item()
            deltas[L, H] = ld_base - ld_abl   # +ve = ablating HURTS

    return deltas

# OV CIRCUIT ANALYSIS
def compute_ov_circuit(
    W:     dict,
    layer: int,
    head:  int,
) -> torch.Tensor:
    """
    OV matrix for head H at layer L:

        OV_h = W_V_h @ W_O_h

    where:
        W_V_h = W_qkv[:, 1536 + h*64 : 1536 + (h+1)*64]   [768, 64]
        W_O_h = W_o[h*64 : (h+1)*64, :]                    [64, 768]

    So OV_h : [768, 768]  (rank-64 matrix)

    Interpretation:
        If token embedding e_t is input, OV_h copies it as:
            out ≈ (e_t @ OV_h) in the vocabulary direction

        Full embedding copy circuit:
            copy_logits = e_t @ OV_h @ W_E.T
            where W_E = wte.weight [50257, 768]

        If the top predictions of copy_logits for input token t are t itself,
        The head is a "copy head" (typical for Name Mover Heads in IOI).

    Returns: [768, 768] OV matrix
    """
    cfg = config_from_weights(W)
    D, Dh, H = cfg["d_model"], cfg["d_head"], cfg["n_heads"]

    p     = f"transformer.h.{layer}"
    W_qkv = W[f"{p}.attn.c_attn.weight"]
    W_o   = W[f"{p}.attn.c_proj.weight"]

    v_start = 2 * D + head * Dh
    W_V_h   = W_qkv[:, v_start:v_start + Dh]
    W_O_h   = W_o.reshape(H, Dh, D)[head]
    return W_V_h @ W_O_h

def top_copy_logits(
    W:          dict,
    layer:      int,
    head:       int,
    test_tokens: list,
    tok,
    top_k:      int = 5,
) -> None:
    """
    For each test token, show what the OV circuit predicts.
    Name Mover Heads should predict the SAME token (copy behavior).
    """
    OV_h = compute_ov_circuit(W, layer, head)
    W_E  = W["transformer.wte.weight"]   # [50257, 768]  (unembedding = W_E.T)

    print(f"\n  OV circuit for L{layer}H{head}:")
    for tok_id in test_tokens:
        e_t          = W_E[tok_id]              # [768]
        copy_logits  = e_t @ OV_h @ W_E.T       # [50257]
        top_ids      = copy_logits.topk(top_k).indices.tolist()
        top_words    = [tok.decode([i]) for i in top_ids]
        input_word   = tok.decode([tok_id])
        print(f"    Input: '{input_word}' → OV output top-{top_k}: {top_words}")

# ATTENTION PATTERN ANALYSIS — WHERE DOES EACH HEAD ATTEND?
def analyse_head_attention(
    clean_ids:  torch.Tensor,
    W:          dict,
    tok,
    target_heads: list,  # [(layer, head), ...]
) -> None:
    """
    For each target head, show which positions it attends to
    FROM the final "to" position (where the model predicts next token).
    """
    with torch.no_grad():
        _, cache = gpt2_forward(clean_ids, W)

    S       = clean_ids.shape[1]
    tokens  = tok.convert_ids_to_tokens(clean_ids[0].tolist())

    print("\nAttention FROM final position → all positions:")
    print(f"  {'Head':<8} " + "  ".join(f"{t:>8}" for t in tokens[-12:]))
    print("  " + "-" * 100)

    for L, H in target_heads:
        pattern  = cache[f"attn.{L}.pattern"][0, H]   # [S, S]
        from_end = pattern[-1, :].tolist()             # attention from last pos
        # Show last 12 tokens for brevity
        vals = from_end[-12:]
        bar  = "  ".join(f"{v:>8.3f}" for v in vals)
        print(f"  L{L}H{H:<4} " + bar)

    # Visualise
    for L, H in target_heads:
        pattern = cache[f"attn.{L}.pattern"][0, H]
        plot_single_head(
            pattern, tokens, layer=L, head=H,
            fname=f"ioi_attn_L{L}H{H}.png",
        )
        print(f"  Saved figures/ioi_attn_L{L}H{H}.png")

# CIRCUIT COMPONENT VERIFICATION
def verify_name_movers(
    clean_ids: torch.Tensor,
    io_token:  int,
    s_token:   int,
    W:         dict,
    ld_clean:  float,
) -> None:
    """
    Ablate NAME MOVER HEADS: L9H6, L9H9, L10H0
    Expected: LD drops significantly (these heads WRITE IO to output).
    """
    name_movers = [(9, 6), (9, 9), (10, 0)]
    print("\nAblating NAME MOVER HEADS:")

    with torch.no_grad():
        _, cache = gpt2_forward(clean_ids, W)

    patches = {}
    for L, H in name_movers:
        key = f"attn.{L}.head.{H}.out"
        patches[key] = torch.zeros_like(cache[key])

    with torch.no_grad():
        logits_abl, _ = gpt2_forward(clean_ids, W, patches=patches)

    ld_abl = logit_diff(logits_abl, io_token, s_token).item()
    print(f"  LD clean:           {ld_clean:+.3f}")
    print(f"  LD ablate NMH:      {ld_abl:+.3f}")
    print(f"  Drop:               {ld_clean - ld_abl:.3f}")
    if ld_clean - ld_abl > 1.0:
        print("  ✓ Name Mover Heads confirmed: ablating them hurts IOI.")
    else:
        print("  ? Smaller than expected drop — may differ from literature.")

def verify_s_inhibition(
    clean_ids:  torch.Tensor,
    io_token:   int,
    s_token:    int,
    W:          dict,
    ld_clean:   float,
) -> None:
    """
    Ablate S-INHIBITION HEADS: L7H3, L7H9, L8H6, L8H10
    Expected: LD INCREASES (they were suppressing S, removing them frees NMH to copy S MORE → hurts task).

    Wait—actually ablating S-inhibition heads should make the model predict S more.
    So LD should DROP (less IO, more S) or become very negative.
    """
    s_inhib = [(7, 3), (7, 9), (8, 6), (8, 10)]
    print("\nAblating S-INHIBITION HEADS:")

    with torch.no_grad():
        _, cache = gpt2_forward(clean_ids, W)

    patches = {}
    for L, H in s_inhib:
        key = f"attn.{L}.head.{H}.out"
        patches[key] = torch.zeros_like(cache[key])

    with torch.no_grad():
        logits_abl, _ = gpt2_forward(clean_ids, W, patches=patches)

    ld_abl = logit_diff(logits_abl, io_token, s_token).item()
    print(f"  LD clean:                {ld_clean:+.3f}")
    print(f"  LD ablate S-inhib:       {ld_abl:+.3f}")
    print(f"  Change:                  {ld_abl - ld_clean:.3f}")
    # Ablating S-inhibition → NMH copies S too (less discriminative) → LD can drop
    # OR if S-inhib was writing negative signal for S → removing helps S prediction
    print("  Note: S-inhibition ablation effect direction depends on mechanistic details.")
    print("  See FINDINGS.md for your observed direction.")

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
    print("═" * 60)
    print(f"IOI CIRCUIT ANALYSIS — {cfg['n_layers']}L {cfg['n_heads']}H d={cfg['d_model']}")
    print("═" * 60)
    print(f"\nClean prompt: '{CLEAN_PROMPT}'")

    # Tokenise and find positions
    clean_ids = tokenize(CLEAN_PROMPT, tok, DEVICE)
    corr_ids  = tokenize(CORR_PROMPT,  tok, DEVICE)
    io_token  = tok.encode(IO_NAME)[0]
    s_token   = tok.encode(S_NAME)[0]

    print("\nToken positions (clean):")
    pos = get_token_positions(CLEAN_PROMPT, tok)

    # Baselines
    with torch.no_grad():
        logits_clean, _ = gpt2_forward(clean_ids, W)
        logits_corr,  _ = gpt2_forward(corr_ids,  W)

    ld_clean = logit_diff(logits_clean, io_token, s_token).item()
    ld_corr  = logit_diff(logits_corr,  io_token, s_token).item()
    print(f"\nBaseline LD (clean):     {ld_clean:+.3f}")
    print(f"Baseline LD (corrupted): {ld_corr:+.3f}")

    # Ablation sweep
    deltas = sweep_ablations(clean_ids, io_token, s_token, W, ld_clean)
    torch.save(deltas, "figures/ablation_deltas.pt")
    plot_logit_diff_by_head(deltas, fname="ablation_by_head.png")

    # Component verification
    verify_name_movers(clean_ids, io_token, s_token, W, ld_clean)
    verify_s_inhibition(clean_ids, io_token, s_token, W, ld_clean)

    # Attention patterns of key heads
    key_heads = [(9, 9), (9, 6), (10, 0),   # Name Movers
                 (7, 3), (7, 9),             # S-Inhibition
                 (0, 1), (3, 0)]             # Duplicate Token
    analyse_head_attention(clean_ids, W, tok, key_heads)

    # OV circuit — do Name Movers copy?
    print("\nOV circuit analysis — Name Mover Heads:")
    test_ids = [io_token, s_token]
    for L, H in [(9, 9), (9, 6), (10, 0)]:
        top_copy_logits(W, L, H, test_ids, tok, top_k=5)

    print("\n04 complete. Key heads identified.")
    print("Expected findings:")
    print("  Name Movers:     L9H6, L9H9, L10H0  — ablating hurts LD")
    print("  S-Inhibition:    L7H3, L7H9, L8H6, L8H10")
    print("  Dup Token:       L0H1, L0H10, L3H0")
    print("Fill in FINDINGS.md with your actual experimental results.")
    print("Proceed to 05_transformerlens_comparison.py")
