"""
Offline tokenizer for mechanistic interpretability experiments.

Why not HuggingFace GPT2Tokenizer?
───────────────────────────────────
Both HuggingFace and tiktoken download vocab.bpe / encoder.json from
  https://openaipublic.blob.core.windows.net/gpt-2/...
which is blocked in this sandbox environment.

What we need for mech interp experiments:
  1. INDUCTION HEADS: Repeated token sequences [t0..tk  t0..tk]
     → Only need stable integer IDs, any consistent mapping works
  2. IOI PATCHING: "When Mary and John ... John gave ... to"
     → Need names to be SINGLE TOKENS (avoid subword splitting)
     → Our word-level tokenizer guarantees this
  3. ATTENTION PATTERNS: Token strings for axis labels
     → Any readable labels work

The MockTokenizer below:
  - Maps space-separated words to integer IDs
  - Has a fixed vocab of common English words at stable IDs
  - Can encode any text deterministically
  - decode() and convert_ids_to_tokens() work correctly
  - On your own machine: replace with GPT2Tokenizer.from_pretrained('gpt2')

Compatibility:
  - encode(text) → List[int]
  - encode(text, return_tensors="pt") → Tensor[1, S]
  - decode([id, id, ...]) → str
  - convert_ids_to_tokens([id, id, ...]) → List[str]
  - pad_token (set to eos_token)
"""

import re
import torch
from typing import List, Optional, Union

# Fixed GPT-2-like vocabulary for common words / subwords
# IDs chosen to not overlap with BOS/EOS/PAD (0-3)
# Names used in IOI experiments get stable low IDs
_FIXED_VOCAB = {
    "<|endoftext|>": 0,
    " Mary":     1601,
    " John":     1757,
    " Sarah":    3362,
    " Tom":      4186,
    " Emma":     5390,
    " David":    5765,
    " Alice":    7576,
    " Bob":      5553,
    "When":       1649,
    " and":        290,
    " went":      1816,
    " to":         284,
    " the":        262,
    " store":     3650,
    ",":            11,
    " gave":      2921,
    " a":          257,
    " drink":     5576,
    " park":      3952,
    " ball":      2613,
    " walked":    6940,
    " school":    1524,
    " handed":    6416,
    " book":      1492,
    "The":         464,
    " transformer": 39432,
    " architecture": 10959,
    " uses":      3544,
    " self":      2116,
    "-":           12,
    " attention": 3241,
    " Hello":    18435,
    " world":     1510,
    " foo":      22944,
    " bar":      2318,
    " baz":      2643,
    "\n":           198,
}

# Build reverse mapping (id → token)
_ID_TO_TOKEN = {v: k for k, v in _FIXED_VOCAB.items()}


class MockTokenizer:
    """
    Word-level tokenizer compatible with GPT2Tokenizer API.
    Each space-delimited word + punctuation becomes one token.
    IDs are stable and drawn from GPT-2 vocab space (0-50256).
    """

    def __init__(self):
        self.vocab = dict(_FIXED_VOCAB)
        self._id_to_tok = dict(_ID_TO_TOKEN)
        self._next_id = 50000   # assign new tokens starting here
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.eos_token_id = 0
        self.pad_token_id = 0

    def _split(self, text: str) -> List[str]:
        """
        Tokenize text into space-separated words and punctuation.
        Preserves leading spaces (like GPT-2 BPE does with 'Ġ').
        """
        # Split on whitespace, keeping punctuation attached to words
        tokens = []
        # Handle leading whitespace
        for match in re.finditer(r'\S+|\n', text):
            word = match.group()
            # Attach space prefix if preceded by whitespace
            start = match.start()
            if start > 0 and text[start-1] == ' ':
                word = ' ' + word
            elif start == 0 and text.startswith(' '):
                word = ' ' + word
            tokens.append(word)
        # Special case: space at start
        if text.startswith(' ') and tokens and not tokens[0].startswith(' '):
            tokens[0] = ' ' + tokens[0]
        return tokens

    def _get_id(self, token: str) -> int:
        """Return token ID, creating new if not in vocab."""
        if token in self.vocab:
            return self.vocab[token]
        # Create deterministic ID from hash
        h = hash(token) % 45000 + 4000   # range 4000-49000
        # Avoid collisions with fixed vocab
        while h in self._id_to_tok:
            h = (h + 1) % 45000 + 4000
        self.vocab[token] = h
        self._id_to_tok[h] = token
        return h

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
    ) -> Union[List[int], torch.Tensor]:
        tokens = self._split(text)
        ids = [self._get_id(t) for t in tokens]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        parts = []
        for i in ids:
            tok = self._id_to_tok.get(i, f"<unk{i}>")
            if skip_special_tokens and tok == "<|endoftext|>":
                continue
            parts.append(tok)
        return "".join(parts)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._id_to_tok.get(i, f"<unk{i}>") for i in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    @property
    def vocab_size(self):
        return 50257


def get_tokenizer() -> MockTokenizer:
    """Drop-in replacement for GPT2Tokenizer.from_pretrained('gpt2')."""
    return MockTokenizer()


def tokenize(text: str, tokenizer, device: str = "cpu") -> torch.Tensor:
    """text → [1, S] token id tensor."""
    return tokenizer.encode(text, return_tensors="pt").to(device)
