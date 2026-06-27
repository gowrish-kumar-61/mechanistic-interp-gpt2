"""
GPT-2 manual forward pass with hook infrastructure.
No TransformerLens. Pure PyTorch. Works for any GPT-2 variant (Small/Medium/Large/XL).

Architecture auto-detected from weight shapes — no hardcoded constants.

WEIGHT CONVENTIONS (HuggingFace GPT-2 uses Conv1D, NOT nn.Linear)
  Conv1D stores W as [in, out], computes x @ W + b
  So HF weights do NOT need transposition before matmul.

HOOK PROTOCOL
  cache   : Dict[str, Tensor] — every activation saved here
  patches : Dict[str, Tensor] — if key present, REPLACE activation
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ponytail: keeping manual layer_norm/gelu — pedagogical, shows the math for interp work

def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)
    return w * (x - mean) / (var + eps).sqrt() + b


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


def multi_head_attn(
    x: torch.Tensor, W_qkv: torch.Tensor, b_qkv: torch.Tensor,
    W_o: torch.Tensor, b_o: torch.Tensor, layer: int,
    cache: Dict, patches: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, S, D = x.shape
    D_HEAD = 64
    H = D // D_HEAD
    L = layer

    def hook(name: str, t: torch.Tensor) -> torch.Tensor:
        cache[name] = t.detach()
        return patches.get(name, t)

    qkv = x @ W_qkv + b_qkv
    q, k, v = qkv.split(D, dim=-1)

    def heads(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, S, H, D_HEAD).permute(0, 2, 1, 3)

    q = hook(f"attn.{L}.q", heads(q))
    k = hook(f"attn.{L}.k", heads(k))
    v = hook(f"attn.{L}.v", heads(v))

    scores = q @ k.transpose(-2, -1) / math.sqrt(D_HEAD)
    causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    pattern = F.softmax(scores, dim=-1)
    pattern = hook(f"attn.{L}.pattern", pattern)

    z = pattern @ v
    z = hook(f"attn.{L}.z", z)

    W_o_h = W_o.reshape(H, D_HEAD, D)
    for h in range(H):
        out_h = z[:, h] @ W_o_h[h]
        hook(f"attn.{L}.head.{h}.out", out_h)

    z_flat = z.permute(0, 2, 1, 3).reshape(B, S, D)
    out = z_flat @ W_o + b_o
    out = hook(f"attn.{L}.out", out)

    return out, pattern


def mlp_block(
    x: torch.Tensor, W_fc: torch.Tensor, b_fc: torch.Tensor,
    W_proj: torch.Tensor, b_proj: torch.Tensor, layer: int,
    cache: Dict, patches: Dict,
) -> torch.Tensor:
    L = layer

    def hook(name: str, t: torch.Tensor) -> torch.Tensor:
        cache[name] = t.detach()
        return patches.get(name, t)

    pre  = hook(f"mlp.{L}.pre",  x @ W_fc + b_fc)
    post = hook(f"mlp.{L}.post", gelu(pre))
    out  = hook(f"mlp.{L}.out",  post @ W_proj + b_proj)
    return out


def config_from_weights(W: Dict[str, torch.Tensor]) -> dict:
    """Derive architecture from weight shapes. Works for gpt2 / gpt2-medium / gpt2-large / gpt2-xl."""
    d_model = W["transformer.wte.weight"].shape[1]
    return {
        "d_model":  d_model,
        "n_heads":  d_model // 64,
        "d_head":   64,
        "d_mlp":    W["transformer.h.0.mlp.c_fc.weight"].shape[1],
        "n_ctx":    W["transformer.wpe.weight"].shape[0],
        "vocab":    W["transformer.wte.weight"].shape[0],
        "n_layers": sum(1 for k in W if k.endswith(".ln_1.weight")),
    }


def load_weights(model_name: str = "gpt2", device: str = "cpu") -> Dict[str, torch.Tensor]:
    from transformers import GPT2LMHeadModel
    print(f"Downloading {model_name} weights …")
    hf = GPT2LMHeadModel.from_pretrained(model_name)
    W = {}
    for name, param in hf.named_parameters():
        W[name] = param.detach().to(device)
    cfg = config_from_weights(W)
    print(f"Loaded {len(W)} tensors. {cfg['n_layers']}L {cfg['n_heads']}H d={cfg['d_model']}")
    return W


def gpt2_forward(
    tokens:  torch.Tensor,
    W:       Dict[str, torch.Tensor],
    patches: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict]:
    patches = patches or {}
    cache: Dict[str, torch.Tensor] = {}
    B, S = tokens.shape
    device = tokens.device

    cfg = config_from_weights(W)
    n_layers = cfg["n_layers"]

    def hook(name: str, t: torch.Tensor) -> torch.Tensor:
        cache[name] = t.detach()
        return patches.get(name, t)

    assert S <= cfg["n_ctx"], f"Sequence length {S} > max context {cfg['n_ctx']}"

    pos     = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    tok_emb = hook("embed",     W["transformer.wte.weight"][tokens])
    pos_emb = hook("pos_embed", W["transformer.wpe.weight"][pos])
    x       = tok_emb + pos_emb

    for L in range(n_layers):
        hp = f"transformer.h.{L}"
        x  = hook(f"resid_pre.{L}", x)

        x_ln1 = layer_norm(x, W[f"{hp}.ln_1.weight"], W[f"{hp}.ln_1.bias"])
        attn_out, _ = multi_head_attn(
            x_ln1,
            W[f"{hp}.attn.c_attn.weight"], W[f"{hp}.attn.c_attn.bias"],
            W[f"{hp}.attn.c_proj.weight"], W[f"{hp}.attn.c_proj.bias"],
            layer=L, cache=cache, patches=patches,
        )
        x = x + attn_out
        x = hook(f"resid_mid.{L}", x)

        x_ln2 = layer_norm(x, W[f"{hp}.ln_2.weight"], W[f"{hp}.ln_2.bias"])
        mlp_out = mlp_block(
            x_ln2,
            W[f"{hp}.mlp.c_fc.weight"], W[f"{hp}.mlp.c_fc.bias"],
            W[f"{hp}.mlp.c_proj.weight"], W[f"{hp}.mlp.c_proj.bias"],
            layer=L, cache=cache, patches=patches,
        )
        x = x + mlp_out
        x = hook(f"resid_post.{L}", x)

    x      = layer_norm(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
    logits = x @ W["transformer.wte.weight"].T
    logits = hook("logits", logits)

    return logits, cache


def get_tokenizer(model_name: str = "gpt2"):
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        return tok
    except ImportError:
        from core.tokenizer import get_tokenizer as _get_tok
        return _get_tok()


def tokenize(text: str, tokenizer, device: str = "cpu") -> torch.Tensor:
    ids = tokenizer.encode(text, return_tensors="pt")
    return ids.to(device)
