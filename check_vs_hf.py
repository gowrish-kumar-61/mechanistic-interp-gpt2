"""Check the manual forward pass against HuggingFace's GPT2LMHeadModel on real weights.

Run:  python check_vs_hf.py --model gpt2-medium
"""
import sys, argparse, torch
sys.path.insert(0, ".")
from transformers import GPT2LMHeadModel
from core.model import gpt2_forward, load_weights, get_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2-medium")
name = p.parse_args().model

tok = get_tokenizer(name)
W = load_weights(name)
hf = GPT2LMHeadModel.from_pretrained(name).eval()

text = "When Mary and John went to the store, John gave a drink to"
ids = torch.tensor([tok.encode(text)])
with torch.no_grad():
    ours, _ = gpt2_forward(ids, W)
    ref = hf(ids).logits

dev = (ours - ref).abs().max().item()
same_top = (ours[0, -1].argmax() == ref[0, -1].argmax()).item()
print(f"{name}: max |logit diff| = {dev:.2e}, same top-1 next token: {same_top}")
print(f"top-1 next token: {tok.decode([ref[0, -1].argmax().item()])!r}")
assert dev < 1e-3 and same_top, "manual forward pass does not match HuggingFace"
print("ok: manual forward pass matches HuggingFace")
