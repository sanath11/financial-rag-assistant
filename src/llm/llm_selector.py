"""
src/llm/llm_selector.py

Unified LLM interface allowing the app to switch between
Gemini, Llama (via Groq), or Qwen (via OpenRouter) via a single config flag.

Usage:
    from src.llm.llm_selector import generate_answer, generate_answer_stream, generate_suggested_questions

    # Switch model by setting the environment variable:
    # LLM_PROVIDER=gemini    (default)
    # LLM_PROVIDER=llama
    # LLM_PROVIDER=qwen

    answer = generate_answer(question, context_chunks, tickers=["NVDA"])
    for chunk in generate_answer_stream(question, context_chunks, tickers=["NVDA"]):
        print(chunk, end="")
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Provider Registry ─────────────────────────────────────────────────────────

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

# Map provider names to their service modules
_PROVIDERS = {
    "gemini": "src.llm.gemini_svc",
    "llama": "src.llm.groq_svc",
    "qwen": "src.llm.qwen_svc",
}


# ── Lazy Loader ────────────────────────────────────────────────────────────────

_mod = None  # caches the imported module on first access


def _get_module():
    """Import and cache the selected provider's service module."""
    global _mod
    if _mod is not None:
        return _mod

    provider_key = _PROVIDER if _PROVIDER in _PROVIDERS else "gemini"
    module_path = _PROVIDERS[provider_key]

    import importlib

    _mod = importlib.import_module(module_path)
    return _mod


def set_provider(provider: str):
    """
    Switch the active LLM provider at runtime.
    Valid values: 'gemini', 'llama', 'qwen'.
    """
    global _PROVIDER, _mod
    provider_key = provider.lower().strip()
    if provider_key not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(_PROVIDERS.keys())}"
        )
    _PROVIDER = provider_key
    _mod = None  # Force re-import on next call


def get_provider() -> str:
    """Return the currently active LLM provider."""
    return _PROVIDER


# ── Unified Interface (delegates to the active provider) ────────────────────


def generate_answer(
    question: str,
    context_chunks: list[dict],
    tickers: list[str] | None = None,
    temperature: float = 0.1,
) -> str:
    """Generate a financial answer using the active LLM given retrieved context."""
    mod = _get_module()
    return mod.generate_answer(question, context_chunks, tickers, temperature)


def generate_answer_stream(
    question: str,
    context_chunks: list[dict],
    tickers: list[str] | None = None,
):
    """Streaming answer using the active LLM. Yields text chunks as they arrive."""
    mod = _get_module()
    yield from mod.generate_answer_stream(question, context_chunks, tickers)


def generate_suggested_questions(tickers: list[str], years: list[str]) -> list[str]:
    """Generate suggested questions using the active LLM."""
    mod = _get_module()
    return mod.generate_suggested_questions(tickers, years)
