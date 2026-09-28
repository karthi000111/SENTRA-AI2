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

    raise RuntimeError(
        "All enabled code generation backends failed. Check the terminal for the "
        "provider error lines above. Set OPENROUTER_API_KEY, or set ENABLE_OLLAMA=1 "
        "to allow the local fallback."
    )