"""
src/llm/gemini_svc.py

Gemini 1.5 Flash API integration for financial RAG answers.
Free tier: 1,500 requests/day, 1M tokens/minute.
Get your key: https://aistudio.google.com/app/apikey
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ── Model Setup ───────────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-flash"   # Free tier
_gemini_ready = False


def _init_gemini():
    global _gemini_ready
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then add it to your .env file."
        )
    genai.configure(api_key=api_key)
    _gemini_ready = True


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior financial analyst assistant with expertise in 
reading SEC filings (10-K, 10-Q), earnings call transcripts, and financial reports.

Your role is to answer questions accurately using ONLY the provided document excerpts.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- Always cite your sources using the [Source: ...] format shown in the context.
- If the context doesn't contain enough information, say so clearly.
- Use clear, professional financial language.
- When comparing companies, structure your answer with clear headings.
- Highlight specific numbers, percentages, and dates when available.
- Flag any forward-looking statements or management guidance as such."""


# ── Prompt Templates ──────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the full RAG prompt combining retrieved context + user question.

    Args:
        question: The user's financial question
        context_chunks: List of dicts with keys: text, source, ticker, year, page
    """
    if not context_chunks:
        return f"Question: {question}\n\nNo relevant documents found."

    # Format context block with source labels
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = chunk.get("source", f"Document {i}")
        context_parts.append(
            f"[Source: {source_label}]\n{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    return f"""Based on the following excerpts from SEC filings and financial documents, answer the question below.

=== DOCUMENT EXCERPTS ===
{context_block}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
- Answer using only the information from the excerpts above.
- Cite sources using [Source: ...] inline wherever you reference specific data.
- If comparing multiple companies, use clear headings for each.
- Be specific: include exact numbers, dates, and percentages when available.

=== ANSWER ==="""


def build_comparison_prompt(question: str, context_chunks: list[dict], tickers: list[str]) -> str:
    """
    Specialized prompt for multi-company comparison questions.
    Groups context by ticker before sending to the model.
    """
    # Group chunks by ticker
    by_ticker: dict[str, list] = {}
    for chunk in context_chunks:
        t = chunk.get("ticker", "UNKNOWN")
        by_ticker.setdefault(t, []).append(chunk)

    context_parts = []
    for ticker in tickers:
        chunks = by_ticker.get(ticker, [])
        if chunks:
            ticker_context = "\n\n".join(
                f"[Source: {c.get('source', ticker)}]\n{c['text']}"
                for c in chunks
            )
            context_parts.append(f"### {ticker} EXCERPTS\n{ticker_context}")

    context_block = "\n\n".join(context_parts)

    return f"""You are comparing {', '.join(tickers)} based on their SEC filings.

=== DOCUMENT EXCERPTS BY COMPANY ===
{context_block}

=== COMPARISON QUESTION ===
{question}

=== INSTRUCTIONS ===
- Structure your answer with a section for each company.
- End with a brief comparative summary table if applicable.
- Cite [Source: ...] for every specific claim.

=== COMPARATIVE ANALYSIS ==="""


# ── Main Answer Function ──────────────────────────────────────────────────────

def generate_answer(
    question: str,
    context_chunks: list[dict],
    tickers: list[str] | None = None,
    temperature: float = 0.1,
) -> str:
    """
    Generate a financial answer using Gemini given retrieved context.

    Args:
        question:       User's question
        context_chunks: Retrieved chunks from ChromaDB
        tickers:        If multiple tickers, use comparison prompt
        temperature:    Lower = more factual (0.0–1.0)

    Returns:
        LLM-generated answer string
    """
    if not _gemini_ready:
        _init_gemini()

    # Choose prompt type
    is_comparison = tickers and len(tickers) > 1
    if is_comparison:
        prompt = build_comparison_prompt(question, context_chunks, tickers)
    else:
        prompt = build_rag_prompt(question, context_chunks)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    try:
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1500,
            ),
        )
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper():
            return "❌ Invalid Gemini API key. Check your .env file."
        elif "QUOTA" in error_msg.upper():
            return "❌ Gemini API quota exceeded. Free tier: 1,500 req/day."
        else:
            return f"❌ Gemini API error: {error_msg}"


# ── Streaming Version (for Streamlit) ────────────────────────────────────────

def generate_answer_stream(
    question: str,
    context_chunks: list[dict],
    tickers: list[str] | None = None,
):
    """
    Streaming version of generate_answer.
    Yields text chunks as they arrive — use with st.write_stream().
    """
    if not _gemini_ready:
        _init_gemini()

    is_comparison = tickers and len(tickers) > 1
    if is_comparison:
        prompt = build_comparison_prompt(question, context_chunks, tickers)
    else:
        prompt = build_rag_prompt(question, context_chunks)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    try:
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500,
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"❌ Error: {e}"
