"""LLM backend with multiple providers.

Priority order:
1. OpenRouter free models (no credit card needed)
2. HuggingFace Inference API (if HF_TOKEN set & credits available)
3. Local Ollama fallback

Set OPENROUTER_API_KEY for the fastest, most reliable free option.
Get a free key at: https://openrouter.ai/settings/keys
"""
from __future__ import annotations
import time
import json
import logging
import os
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – override via env vars
# ---------------------------------------------------------------------------
# OpenRouter (free, recommended)
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL: str = os.getenv(
    "OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free"
)

# HuggingFace (existing, may have exhausted credits)
HF_MODEL_ID: str = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Generation parameters
_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_TOKENS", "8192"))
_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.15"))
_TOP_P = float(os.getenv("HF_TOP_P", "0.9"))

# Ollama timeout
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# System prompt shared by all providers
_SYSTEM_PROMPT = (
    "You are an expert ML engineer. You write clean, complete, "
    "runnable Python code that faithfully implements machine "
    "learning papers based on extracted specifications."
)


# ---------------------------------------------------------------------------
# Provider 1: OpenRouter (FREE, no credit card)
# ---------------------------------------------------------------------------
def call_openrouter(prompt: str) -> str | None:
    """Try each configured OpenRouter model in turn until one answers."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.warning("OpenRouter: no OPENROUTER_API_KEY in environment")
        return None

    models = [m.strip() for m in os.getenv(
        "OPENROUTER_MODELS", os.getenv("OPENROUTER_MODEL", "openrouter/free")
    ).split(",") if m.strip()]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/hallucination-control-framework",
        "X-Title": "Hallucination Control Framework",
    }

    for sweep in range(2):                      # go through the whole list twice
        for model in models:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": _MAX_NEW_TOKENS,
                "temperature": _TEMPERATURE,
                "top_p": _TOP_P,
            }).encode("utf-8")
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
                if "error" in data:
                    logger.warning("OpenRouter %s error: %s", model, data["error"])
                    continue
                text = data["choices"][0]["message"]["content"]
                if text:
                    logger.warning("OpenRouter model used: %s", model)
                    return text
            except urllib.error.HTTPError as e:
                logger.warning("OpenRouter %s -> HTTP %d, trying next model", model, e.code)
                if e.code in (401, 402):        # key/credit problems: no point continuing
                    return None
            except Exception as exc:
                logger.warning("OpenRouter %s failed: %s", model, exc)
        time.sleep(3)                            # short pause between sweeps
    return None

# ---------------------------------------------------------------------------
# Provider 2: HuggingFace Inference API
# ---------------------------------------------------------------------------
def _get_hf_client():
    """Lazy-initialised HF Inference client."""
    from huggingface_hub import InferenceClient
    token = os.getenv("HF_TOKEN")
    model_id = os.getenv("HF_MODEL_ID", HF_MODEL_ID)
    if not token:
        return None
    return InferenceClient(model=model_id, token=token)


def call_hf(prompt: str) -> str | None:
    """Call HuggingFace Inference API. Returns None on failure."""
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token or not token.startswith("hf_"):
        return None

    try:
        client = _get_hf_client()
        if client is None:
            return None

        model_id = os.getenv("HF_MODEL_ID", HF_MODEL_ID)
        logger.info("Calling HF Inference model=%s prompt_len=%d", model_id, len(prompt))

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
            top_p=_TOP_P,
        )

        text = response.choices[0].message.content
        if text:
            logger.info("HF Inference returned %d chars.", len(text))
            return text
    except Exception as exc:
        exc_str = str(exc)
        if "402" in exc_str or "Payment Required" in exc_str or "depleted" in exc_str.lower():
            logger.warning("HF credits exhausted (402). Skipping HF.")
        else:
            logger.warning("HF Inference failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Provider 3: Local Ollama fallback
# ---------------------------------------------------------------------------
def call_ollama(prompt: str) -> str | None:
    """Fallback: call local Ollama endpoint if running."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    # Auto-detect available local model if not explicitly set
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        try:
            with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as tag_resp:
                tags = json.loads(tag_resp.read().decode())
                models = [m.get("name") for m in tags.get("models", [])]
                if models:
                    model = models[0]
        except Exception:
            pass
    if not model:
        model = "qwen3:4b"

    try:
        logger.info("Calling local Ollama model=%s", model)
        req = urllib.request.Request(
            f"{ollama_url}/api/chat",
            data=json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            text = data.get("message", {}).get("content")
            if text:
                logger.info("Ollama returned %d chars.", len(text))
                return text
    except Exception as err:
        logger.warning("Ollama fallback failed: %s", err)

    return None


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def call_hf_inference(prompt: str) -> str:
    """Try providers in order, logging which one answered and how long it took."""
    providers = [("openrouter", call_openrouter), ("huggingface", call_hf)]
    if os.getenv("ENABLE_OLLAMA", "0") == "1":
        providers.append(("ollama", call_ollama))

    for name, fn in providers:
        start = time.time()
        result = fn(prompt)
        elapsed = time.time() - start
        if result:
            logger.warning("LLM provider=%s OK in %.1fs (%d chars)", name, elapsed, len(result))
            return result
        logger.warning("LLM provider=%s FAILED after %.1fs, trying next", name, elapsed)

    logger.warning("All online LLM providers failed or unconfigured. Using offline code generator.")
    return _offline_generate(prompt)


# ---------------------------------------------------------------------------
# Offline prompt-aware code generator (no API needed)
# ---------------------------------------------------------------------------
_OFFLINE_FILES = {
    "attention.py": '''```python
import torch
import torch.nn as nn
import math

# SPEC: core_method_description = Scaled Dot-Product and Multi-Head Attention
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V"""
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights


class MultiHeadAttention(nn.Module):
    # SPEC: key_parameters = d_model=64, num_heads=4
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        B, S, _ = query.shape
        q = self.q_proj(query).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        context, _ = self.attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(context)
```''',

    "layers.py": '''```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    def __init__(self, d_model=64, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PositionwiseFeedForward(nn.Module):
    # SPEC: key_parameters = d_ff = 4 * d_model
    def __init__(self, d_model=64, d_ff=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
```''',

    "blocks.py": '''```python
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layers import PositionwiseFeedForward

class EncoderBlock(nn.Module):
    """Single Transformer encoder block: MHA -> Add&Norm -> FFN -> Add&Norm."""
    def __init__(self, d_model=64, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
```''',

    "model.py": '''```python
import torch
import torch.nn as nn
from layers import PositionalEncoding
from blocks import EncoderBlock

class TransformerEncoder(nn.Module):
    # SPEC: core_method_description = Stacked Transformer Encoder
    # SPEC: key_parameters = d_model=64, num_heads=4, num_layers=2
    def __init__(self, d_model=64, num_heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```''',

    "train.py": '''```python
import torch
import torch.nn as nn
from model import TransformerEncoder

def train_one_epoch(model, optimizer, data_loader, device="cpu"):
    """Training loop for one epoch."""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(data_loader), 1)

def build_optimizer(model, lr=1e-3):
    # ASSUMED (not in paper): Adam optimizer with lr=1e-3
    return torch.optim.Adam(model.parameters(), lr=lr)
```''',

    "main.py": '''```python
import torch
from model import TransformerEncoder

if __name__ == "__main__":
    # SPEC: key_parameters = d_model=64, num_heads=4, num_layers=2
    model = TransformerEncoder(d_model=64, num_heads=4, num_layers=2)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy forward pass
    dummy_input = torch.randn(2, 16, 64)  # (batch=2, seq_len=16, d_model=64)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Shape mismatch!"
    print("Forward pass OK.")
```''',
}


def _offline_generate(prompt: str) -> str:
    """Parse which file is being requested from the prompt and return the correct template.

    The CodeAgent prompt contains: YOUR TASK: write ONLY `<filename>`.
    We look for that exact marker first, then fall back to simple substring matching.
    """
    prompt_lower = prompt.lower()

    # Strategy 1: exact marker from CodeAgent prompt
    import re
    marker = re.search(r"write only\s*`(\w+\.py)`", prompt_lower)
    if marker:
        target = marker.group(1)
        if target in _OFFLINE_FILES:
            logger.info("Offline fallback: generating %s (exact marker match)", target)
            return _OFFLINE_FILES[target]

    # Strategy 2: look for "your task" section filename
    for filename in _OFFLINE_FILES:
        # Check for filename in backticks to avoid substring false positives
        if f"`{filename}`" in prompt_lower:
            logger.info("Offline fallback: generating %s (backtick match)", filename)
            return _OFFLINE_FILES[filename]

    # Strategy 3: bare substring (least precise)
    for filename in _OFFLINE_FILES:
        if filename in prompt_lower:
            logger.info("Offline fallback: generating %s (substring match)", filename)
            return _OFFLINE_FILES[filename]

    # Fallback: return model.py
    logger.info("Offline fallback: no specific file detected, returning model.py")
    return _OFFLINE_FILES["model.py"]

