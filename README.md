# Mechanistic Interpretability on GPT-2

Manual mechanistic interpretability pipeline for GPT-2 (Small through XL). Pure PyTorch — no TransformerLens dependency for the core analysis.

## What This Does

1. **Load & inspect weights** — manual forward pass with full activation caching
2. **Activation patching** — swap activations between clean/corrupted runs to find causal components
3. **Induction head detection** — find heads that implement the [A B ... A → B] pattern
4. **IOI circuit dissection** — identify Name Mover, S-Inhibition, and Duplicate Token heads
5. **TransformerLens comparison** — verify manual implementation matches TL outputs

## Architecture

```
core/
├── model.py              # Manual GPT-2 forward pass, auto-detects architecture from weights
├── synthetic_weights.py   # Planted circuits (PTH @ L3H0, IH @ L5H1) for ground-truth validation
├── tokenizer.py           # Offline tokenizer (no HuggingFace download needed)
├── metrics.py             # Logit difference, patching recovery, top-k
└── visualize.py           # Attention heatmaps, induction scores, patching plots

01_load_weights.py         # Weight loading + shape verification
02_activation_patching.py  # Causal patching on induction task
03_induction_heads.py      # Induction & previous-token head detection
04_ioi_circuit.py          # IOI circuit: NMH, S-inhibition, OV analysis
05_transformerlens_comparison.py  # API mapping + optional real TL comparison
```

## Quick Start

```bash
pip install -r requirements.txt

# Run with synthetic weights (no downloads, planted circuits for validation)
python 01_load_weights.py
python 02_activation_patching.py
python 03_induction_heads.py
python 04_ioi_circuit.py
python 05_transformerlens_comparison.py

# Run on real GPT-2 XL (1.5B params, requires HuggingFace)
python 01_load_weights.py --model gpt2-xl
python 03_induction_heads.py --model gpt2-xl
python 04_ioi_circuit.py --model gpt2-xl
```

## Supported Models

Any GPT-2 variant — architecture is auto-detected from weight shapes:

| Model | Params | Layers | Heads | d_model |
|-------|--------|--------|-------|---------|
| `gpt2` | 117M | 12 | 12 | 768 |
| `gpt2-medium` | 345M | 24 | 16 | 1024 |
| `gpt2-large` | 774M | 36 | 20 | 1280 |
| `gpt2-xl` | 1.5B | 48 | 25 | 1600 |

## Key Findings

See [FINDINGS.md](FINDINGS.md) for detailed experimental results.

**Synthetic weights (ground-truth validation):**
- PTH at L3H0 detected with 19× signal-to-noise ratio (PT-score 0.87 vs 0.045 baseline)
- Induction score requires K-composition — planting Q/K alone is insufficient
- OV copy behavior washed out by random W_O (gradient descent aligns V and O simultaneously)

## How It Works

The core insight: GPT-2's forward pass is just matrix multiplications and softmax. By caching every intermediate activation and allowing targeted replacement (patching), we can identify which components are causally responsible for specific behaviors.

```python
# Manual forward pass with hook infrastructure
logits, cache = gpt2_forward(tokens, W)

# Patch a specific head's output from a clean run into a corrupted run
patches = {"attn.5.head.1.out": cache_clean["attn.5.head.1.out"]}
logits_patched, _ = gpt2_forward(corrupted_tokens, W, patches=patches)
```

## References

- Olsson et al. (2022) — [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)
- Wang et al. (2022) — [Interpretability in the Wild: IOI in GPT-2 Small](https://arxiv.org/abs/2211.00593)
- Neel Nanda's [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
