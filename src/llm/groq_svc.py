"""
src/llm/groq_svc.py

Groq API integration for financial RAG answers.
Free tier: ~1,440 requests/day, 60 RPM.
Get your key: https://console.groq.com/keys
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model Setup ───────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
_groq_ready = False


def _init_groq():
    global _groq_ready
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Get a free key at: https://console.groq.com/keys\n"
            "Then add it to your .env file."
        )
    _groq_ready = True


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior financial analyst assistant with expertise in
reading SEC filings (10-K), earnings call transcripts, and financial reports.

Your role is to answer questions accurately using ONLY the provided document excerpts.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- Always cite your sources using the [Source: ...] format shown in the context.
- If the context does not contain enough information, say so clearly.
- Use clear, professional financial language.
- When comparing companies, structure your answer with clear headings.
- Highlight specific numbers, percentages, and dates when available.
- Flag any forward-looking statements or management guidance as such."""


# ── Prompt Builders ─────────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the full RAG prompt combining retrieved context + user question."""
    if not context_chunks:
        return f"Question: {question}\n\nNo relevant documents found."

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = chunk.get("source", f"Document {i}")
        context_parts.append(f"[Source: {source_label}]\n{chunk['text']}")

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
    """Specialized prompt for multi-company comparison questions."""
    by_ticker: dict[str, list] = {}
    for chunk in context_chunks:
        t = chunk.get("ticker", "UNKNOWN")
        by_ticker.setdefault(t, []).append(chunk)

    context_parts = []
    for ticker in tickers:
        chunks = by_ticker.get(ticker, [])
        if chunks:
            ticker_context = "\n\n".join(
                f"[Source: {c.get('source', ticker)}]\n{c['text']}" for c in chunks
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


# ── Client Helper ────────────────────────────────────────────────

from groq import Groq as _GroqClient


def _get_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Get a free key at: https://console.groq.com/keys\n"
            "Then add it to your .env file."
        )
    return _GroqClient(api_key=api_key)


# ── Main Answer Function ──────────────────────────────────────────────────────

def generate_answer(
    question: str,
    context_chunks: list[dict],
    tickers: list[str] | None = None,
    temperature: float = 0.1,
) -> str:
    """Generate a financial answer using Groq given retrieved context."""
    client = _get_client()

    is_comparison = tickers and len(tickers) > 1
    if is_comparison:
        prompt = build_comparison_prompt(question, context_chunks, tickers)
    else:
        prompt = build_rag_prompt(question, context_chunks)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature,
            max_tokens=1500,
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return "❌ Groq API rate limit exceeded. Free tier: ~1,440 req/day."
        elif "api_key" in error_msg.lower():
            return "❌ Invalid Groq API key. Check your .env file."
        else:
            return f"❌ Groq API error: {error_msg}"


# ── Streaming Version ────────────────────────────────────────────

def generate_answer_stream(
    question: str,
    context_chunks: list[dict],
    tickers: list[str] | None = None,
):
    """Streaming version of generate_answer. Yields text chunks as they arrive."""
    client = _get_client()

    is_comparison = tickers and len(tickers) > 1
    if is_comparison:
        prompt = build_comparison_prompt(question, context_chunks, tickers)
    else:
        prompt = build_rag_prompt(question, context_chunks)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
            max_tokens=1500,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"❌ Error: {e}"


# ── Suggested Questions Generator ──────────────────────────────

def generate_suggested_questions(tickers: list[str], years: list[str]) -> list[str]:
    """Ask Groq to generate 5 relevant questions based on available tickers and years."""
    client = _get_client()

    tickers_str = ", ".join(tickers) if tickers else "various companies"
    years_str = ", ".join(sorted(years)) if years else "recent years"

    prompt = f"""You are helping users explore SEC 10-K filings.
The database contains filings for these companies: {tickers_str}
Filing years available: {years_str}

Generate exactly 5 specific, interesting questions a financial analyst would ask.
Rules:
- Reference specific company names or tickers from the list above
- Mix single-company and cross-company comparison questions
- Focus on: risks, revenue, growth, strategy, competition, guidance
- Keep each question under 15 words
- Return ONLY a numbered list 1-5, no extra text

Example format:
1. What risks did NVDA highlight in their 2024 filing?
2. Compare TSLA and AAPL revenue growth strategies"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=300,
        )
        lines = response.choices[0].message.content.strip().splitlines()
        questions = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                q = line.split(".", 1)[-1].strip()
                if q:
                    questions.append(q)

        return questions[:5] if len(questions) >= 3 else _fallback_questions(tickers)

    except Exception:
        return _fallback_questions(tickers)


def _fallback_questions(tickers: list[str]) -> list[str]:
    """Generic questions used if Groq call fails."""
    t = tickers[0] if tickers else "the company"
    return [
        f"What risks did {t} mention in their latest filing?",
        f"Summarize {t}'s revenue and profitability",
        "Compare AI investment strategies across companies",
        "What forward guidance was provided?",
        "How does management describe competitive positioning?",
    ]
