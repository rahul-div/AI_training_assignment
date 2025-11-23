from __future__ import annotations

import html
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

import httpx
import logfire
from dotenv import load_dotenv
from fasthtml.common import *  # FastHTML components and helpers
from pydantic_ai import Agent
from starlette.responses import StreamingResponse

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Logfire configuration & instrumentation
# ---------------------------------------------------------------------------

# Configure Logfire. By default this reads configuration (including the write
# token) from the local `.logfire` directory or LOGFIRE_TOKEN env var. :contentReference[oaicite:4]{index=4}
logfire.configure()

# Instrument Pydantic-AI so every agent run, model call and tool call is traced
# and token usage is recorded. See: https://ai.pydantic.dev/logfire/
logfire.instrument_pydantic_ai()

# Also instrument HTTPX so calls to Wikipedia, arXiv and the Gemini provider
# are visible inside Logfire (request + response details). :contentReference[oaicite:5]{index=5}
logfire.instrument_httpx(capture_all=False)

# ---------------------------------------------------------------------------
# Pydantic-AI research agent using Gemini 2.5 Flash
# ---------------------------------------------------------------------------

# This uses the Google Generative Language API backend (“google-gla” provider)
# with the gemini-2.5-flash model. :contentReference[oaicite:6]{index=6}
research_agent = Agent(
    "google-gla:gemini-2.5-flash",
    system_prompt=(
        "You are a careful research assistant.\n"
        "- Decide which tools to call based on the user's question.\n"
        "- Prefer wikipedia_search for general background and definitions.\n"
        "- Prefer arxiv_search for technical / scientific / academic topics.\n"
        "- When you use tools, weave their results into a single coherent answer.\n"
        "- Always mention explicitly which tools you used.\n"
        "- Structure your final answer as:\n"
        "  1) Short high-level summary\n"
        "  2) Bullet points with key findings\n"
        "  3) Optional 'Next steps / further reading' section.\n"
    ),
)

# ----------------------------- Tools ---------------------------------------
# Tools are plain Python functions decorated with @agent.tool / @agent.tool_plain.
# Pydantic-AI uses type hints and docstrings to build the tool schema
# and decide when/how to call them. :contentReference[oaicite:7]{index=7}


@research_agent.tool_plain
def wikipedia_search(query: str, language: str = "en", max_results: int = 3) -> str:
    """Search Wikipedia and return a compact plain-text summary.

    Args:
        query: Natural-language text describing what to search for.
        language: Two-letter language code like "en" or "de".
        max_results: Maximum number of articles to summarize (1–5 recommended).

    Returns:
        A human-readable summary listing the top matching pages with titles,
        short snippets, and URLs. If there are problems, a short error
        message is returned instead so the calling agent can handle it.
    """
    # Clamp max_results into a safe range so the LLM can't abuse the tool.
    max_results = max(1, min(int(max_results), 5))

    api_url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
    }

    try:
        headers = {
            "User-Agent": "ResearchAgent/1.0 (https://github.com/user/research-agent; user@example.com)"
        }
        resp = httpx.get(api_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        # We keep the error as a string so the agent can gracefully explain
        # that Wikipedia failed rather than crashing the whole run.
        return f"Wikipedia search failed due to a network error: {exc}"

    data = resp.json()
    search_results = data.get("query", {}).get("search", [])

    if not search_results:
        return f"No Wikipedia results found for query: {query!r}"

    lines: list[str] = [f"Top {len(search_results)} Wikipedia results for {query!r}:"]

    for item in search_results:
        title = item.get("title", "").strip()
        snippet_html = item.get("snippet", "")
        # Strip simple HTML tags from the snippet and unescape entities
        snippet_text = re.sub(r"<.*?>", "", snippet_html)
        snippet_text = html.unescape(snippet_text)
        page_slug = title.replace(" ", "_")
        page_url = f"https://{language}.wikipedia.org/wiki/{page_slug}"
        lines.append(f"- {title}\n  URL: {page_url}\n  Snippet: {snippet_text}")

    # Returning plain text is enough; the agent will quote/cite this.
    return "\n".join(lines)


@research_agent.tool_plain
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search the arXiv preprint server for scientific papers.

    Args:
        query: A free-text or keyword query (e.g. "diffusion models for images").
        max_results: Maximum number of papers to return (1–10 recommended).

    Returns:
        A plain-text list of matching arXiv papers with titles, short summaries,
        and URLs, or an informative error string.
    """
    max_results = max(1, min(int(max_results), 10))

    api_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        headers = {
            "User-Agent": "ResearchAgent/1.0 (https://github.com/user/research-agent; user@example.com)"
        }
        resp = httpx.get(api_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"arXiv search failed due to a network error: {exc}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        return f"arXiv search returned malformed XML: {exc}"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: list[str] = []

    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()

        link = None
        for link_el in entry.findall("atom:link", ns):
            if link_el.attrib.get("rel") == "alternate":
                link = link_el.attrib.get("href")
                break
        if not link:
            link = entry.findtext("atom:id", default="", namespaces=ns).strip()

        entries.append(f"- {title}\n  URL: {link}\n  Summary: {summary}")

    if not entries:
        return f"No arXiv results found for query: {query!r}"

    header = f"Top {len(entries)} arXiv results for {query!r}:"
    return "\n".join([header, *entries])


# ---------------------------------------------------------------------------
# FastHTML app & /research route
# ---------------------------------------------------------------------------

# FastHTML "minimal app" pattern, using Pico.css by default for nice base UI.
app, rt = fast_app()


def research_page(
    question: str = "",
    answer: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[str] = None,
):
    """Render the research UI using FastHTML components.

    This returns a tuple of FastTags which FastHTML renders as a full HTML page. :contentReference[oaicite:9]{index=9}
    """
    # Build a small usage pill if we have token info
    usage_ft = None
    if input_tokens is not None and output_tokens is not None:
        usage_ft = Div(
            Span("Tokens", cls="muted"),
            Span(f"in: {input_tokens}", cls="mono"),
            Span("·", cls="muted"),
            Span(f"out: {output_tokens}", cls="mono"),
            cls="token-pill",
        )

    return (
        Title("Research Agent · Pydantic-AI + FastHTML"),
        Div(
            Div(
                Div(
                    # Header row: title + “observed by Logfire” pill
                    Div(
                        Div(
                            H1("Research Agent"),
                            P(
                                "Backed by Gemini 2.5 Flash, with tools for Wikipedia and arXiv.",
                                cls="muted",
                            ),
                            cls="research-title-box",
                        ),
                        Div(
                            Span(cls="research-pill-dot"),
                            Span("Tool-aware agent · observed by Logfire"),
                            cls="research-pill",
                        ),
                        cls="research-header",
                    ),
                    # Two-column layout: form (left), answer (right)
                    Div(
                        # Form column
                        Div(
                            H2("Ask a research question"),
                            Form(id="research-form")(
                                Label("Question", _for="question"),
                                Textarea(
                                    question or "",
                                    id="question",
                                    name="question",
                                    placeholder=(
                                        "e.g. Compare diffusion models and transformers "
                                        "for image generation, with recent papers."
                                    ),
                                    required=True,
                                ),
                                Div(
                                    Span(
                                        "The agent will decide when to call Wikipedia and arXiv tools.",
                                        cls="muted",
                                    ),
                                    Button(
                                        "Run research →",
                                        type="submit",
                                        id="submit-btn",
                                        cls="btn-primary",
                                    ),
                                    cls="research-footer",
                                ),
                            ),
                            cls="research-form-card",
                        ),
                        # Answer column
                        Div(
                            H2("Answer"),
                            Div(
                                # Answer area that will be populated by streaming
                                Div(
                                    P(
                                        "Your answer will appear here once you run a query.",
                                        cls="muted",
                                    ),
                                    id="answer-content",
                                ),
                            ),
                            Div(
                                Div(
                                    Span(
                                        "Tools available: wikipedia_search, arxiv_search",
                                        cls="muted",
                                    ),
                                ),
                                Div(
                                    Span(
                                        "Tokens will be shown after the first run.",
                                        cls="muted",
                                    ),
                                    id="token-info",
                                    cls="research-footer-right",
                                ),
                                cls="research-footer",
                            ),
                            cls="research-answer-card",
                        ),
                        cls="research-layout",
                    ),
                    cls="research-card",
                ),
                cls="research-shell",
            ),
        ),
        # JavaScript for handling streaming
        Script("""
            document.getElementById('research-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const question = document.getElementById('question').value;
                const submitBtn = document.getElementById('submit-btn');
                const answerContent = document.getElementById('answer-content');
                const tokenInfo = document.getElementById('token-info');
                
                if (!question.trim()) return;
                
                // Update UI
                submitBtn.textContent = 'Researching...';
                submitBtn.disabled = true;
                answerContent.innerHTML = '<div style="font-style: italic;">🔍 Starting research...</div>';
                
                try {
                    const response = await fetch('/research/stream', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const reader = response.body.getReader();
                    let accumulatedText = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        
                        if (done) break;
                        
                        const chunk = new TextDecoder().decode(value);
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') {
                                    break;
                                }
                                
                                try {
                                    const parsed = JSON.parse(data);
                                    
                                    if (parsed.type === 'content') {
                                        accumulatedText += parsed.content;
                                        answerContent.innerHTML = '<pre style="white-space: pre-wrap; font-family: monospace;">' + 
                                            accumulatedText.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre>';
                                    } else if (parsed.type === 'tokens') {
                                        tokenInfo.innerHTML = '<span style="font-family: monospace;">📊 Tokens - Input: ' + 
                                            parsed.input_tokens + ', Output: ' + parsed.output_tokens + '</span>';
                                    } else if (parsed.type === 'error') {
                                        answerContent.innerHTML = '<div style="color: red; padding: 10px;">❌ Error: ' + 
                                            parsed.message.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>';
                                    }
                                } catch (e) {
                                    console.log('Failed to parse SSE data:', data);
                                }
                            }
                        }
                    }
                } catch (error) {
                    answerContent.innerHTML = '<div style="color: red; padding: 10px;">❌ Error: ' + error.message + '</div>';
                } finally {
                    submitBtn.textContent = 'Run research →';
                    submitBtn.disabled = false;
                }
            });
        """),
    )


@rt("/research")
def research(question: str = ""):
    """Research route: render the form and, if a question is provided, run the agent.

    FastHTML maps the form field named "question" to this parameter on POST,
    and also supports GET with a query string (?question=...). :contentReference[oaicite:10]{index=10}
    """
    # Initial load: show empty form
    if not question.strip():
        return research_page()

    try:
        # Run the agent synchronously for this request. :contentReference[oaicite:11]{index=11}
        result = research_agent.run_sync(question)
        output_text = result.output

        # Ask Pydantic-AI for usage stats (input/output tokens).
        usage = result.usage()
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        return research_page(
            question=question,
            answer=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 – in demo, we surface any errors to the UI
        # In a production app you’d log & maybe show a nicer error page.
        return research_page(
            question=question,
            error=f"Something went wrong while running the research agent: {exc}",
        )


async def stream_research_generator(question: str):
    """Generate streaming responses for research queries."""
    import asyncio

    try:
        # Send initial status
        yield f"data: {json.dumps({'type': 'content', 'content': '🔍 Starting research...\n\n'})}\n\n"
        await asyncio.sleep(0.1)

        # Run the research agent in a background thread since it's synchronous
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(research_agent.run_sync, question)

            # Send periodic status updates while waiting
            while not future.done():
                yield f"data: {json.dumps({'type': 'content', 'content': '🤖 Thinking...\n'})}\n\n"
                await asyncio.sleep(1)

            result = future.result()

        # Clear the thinking message and send the actual response
        yield f"data: {json.dumps({'type': 'content', 'content': '\n🎯 Research Results:\n\n'})}\n\n"
        await asyncio.sleep(0.1)

        # Stream the response character by character or word by word
        response_text = result.output
        words = response_text.split(" ")

        for i, word in enumerate(words):
            # Add word with space
            word_to_send = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'content', 'content': word_to_send})}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect

        # Send token usage information
        usage = result.usage()
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        if input_tokens is not None and output_tokens is not None:
            yield f"data: {json.dumps({'type': 'tokens', 'input_tokens': input_tokens, 'output_tokens': output_tokens})}\n\n"

        # Send completion signal
        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"


@rt("/research/stream", methods=["POST"])
async def research_stream(request):
    """Handle streaming research requests."""
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        return {"error": "Question is required"}, 400

    return StreamingResponse(
        stream_research_generator(question),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


# For local development: `python main.py` then visit http://localhost:5001/research
if __name__ == "__main__":
    # FastHTML's serve() wraps uvicorn with sensible defaults.
    print("Starting FastHTML server...")
    print("Visit: http://localhost:5001/research")
    serve()
