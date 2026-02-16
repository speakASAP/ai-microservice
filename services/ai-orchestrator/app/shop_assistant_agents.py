"""
Shop-assistant agents: COMMUNICATION, SEARCH, PRESENTATION, COMPARISON, LOCATION.
All agent logic runs in ai-microservice; shop-assistant calls these endpoints only.
"""

import os
from typing import Any, Dict, List, Optional
import httpx

# Logger set by caller (main.py) or use module logger
logger = None


def set_logger(l):
    global logger
    logger = l


# Default prompts when caller does not send overrides
DEFAULT_REFINE_QUERY_PROMPT = (
    'Extract a single web product search query from: "{{user_input}}". '
    'Reply with ONE short search query only (max 200 chars), no other text.'
)
DEFAULT_REFINE_QUERY_PROMPT_WITH_PARAMS = (
    'Refine this product search. User said: "{{user_input}}. Previous constraints: {{previous_params}}. '
    'Reply with ONE short search query only (max 200 chars), no other text.'
)
DEFAULT_PRESENTATION_PROMPT = (
    'Format these product search results for the user in a clear, readable way (dialog or list). '
    'Query: {{queryText}}. Results: {{searchResults}}. '
    'Return only the formatted text, no extra commentary.'
)
DEFAULT_COMPARISON_PROMPT = (
    'You are a shopping assistant comparing products for a user.\n\n'
    'User query: {{queryText}}\n'
    'Priorities (price, quality, location, etc.): {{priorityOrder}}\n\n'
    'Search results:\n{{searchResults}}\n\n'
    'Write a short comparison focusing on the user priorities. '
    'Recommend several best options and explain briefly. Keep it concise.'
)
DEFAULT_LOCATION_PROMPT = (
    'User is shopping online.\n\n'
    'User input: {{userInput}}\n'
    'Current query: {{queryText}}\n'
    'Priorities (price, quality, location, etc.): {{priorityOrder}}\n\n'
    'Extract a short phrase describing the delivery region or shipping location that matters for this request, '
    'for example "delivery Czech Republic" or "ships to EU". If region is not specified, respond with an empty string.\n'
    'Answer with this short phrase only, no other text.'
)


def _get_nlp_url() -> str:
    return os.getenv("NLP_SERVICE_URL", "http://free-ai-service:3386").rstrip("/")


def _get_asr_url() -> str:
    return os.getenv("ASR_SERVICE_URL", "http://asr-service:3382").rstrip("/")


def _get_search_config() -> tuple:
    url = os.getenv("SEARCH_API_URL", "").strip()
    key = os.getenv("SEARCH_API_KEY", "").strip()
    return url, key


async def agent_transcribe(voice_file_url: str) -> Dict[str, Any]:
    """ASR agent: transcribe audio URL to text. Proxies to ASR service."""
    asr_url = _get_asr_url()
    if not voice_file_url:
        return {"transcript": ""}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{asr_url}/api/transcribe",
                json={"voice_file_url": voice_file_url},
            )
            r.raise_for_status()
            data = r.json()
            transcript = data.get("transcript") or data.get("text") or ""
            return {"transcript": transcript}
    except Exception as e:
        if logger:
            logger.error("Shop-assistant ASR transcribe failed: %s", e)
        raise


async def agent_refine_query(
    user_text: str,
    previous_params: Optional[Dict[str, Any]] = None,
    role: str = "default",
    prompt_content: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    COMMUNICATION agent: refine raw user text into a search-friendly query.
    Uses NLP (free-ai) content_generation. Optional prompt_content/model override from admin.
    """
    nlp_url = _get_nlp_url()
    if not user_text or not user_text.strip():
        return {"query_text": "", "refined_params": previous_params or {}}

    if prompt_content:
        text_content = (
            prompt_content.replace("{{user_input}}", user_text)
            .replace("{{userInput}}", user_text)
            .replace("{{previous_params}}", (previous_params and str(previous_params)) or "")
            .replace("{{previousParams}}", (previous_params and str(previous_params)) or "")
        )
    else:
        if previous_params:
            text_content = (
                DEFAULT_REFINE_QUERY_PROMPT_WITH_PARAMS
                .replace("{{user_input}}", user_text)
                .replace("{{previous_params}}", str(previous_params))
            )
        else:
            text_content = DEFAULT_REFINE_QUERY_PROMPT.replace("{{user_input}}", user_text)

    body: Dict[str, Any] = {
        "text_content": text_content,
        "analysis_type": "content_generation",
        "user_name": "shop-assistant",
    }
    if model:
        body["model"] = model

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{nlp_url}/api/analyze-text", json=body)
            r.raise_for_status()
            data = r.json()
            analysis = data.get("analysis") or data
            content = (
                analysis.get("summary")
                or (analysis.get("key_insights") and analysis["key_insights"][0])
                or analysis.get("text")
                or ""
            )
            if isinstance(content, list):
                content = content[0] if content else ""
            query_text = (content if isinstance(content, str) else str(content))[:200].strip() or user_text.strip()
            return {
                "query_text": query_text,
                "refined_params": previous_params or {},
            }
    except Exception as e:
        if logger:
            logger.warning("Shop-assistant refine-query failed, using raw text: %s", e)
        return {"query_text": user_text.strip(), "refined_params": previous_params or {}}


async def agent_search(query_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    SEARCH agent: run external search (Serper) and return product-like items.
    Requires SEARCH_API_URL and SEARCH_API_KEY in .env.
    """
    search_url, api_key = _get_search_config()
    if not search_url or not api_key:
        if logger:
            logger.warning("SEARCH_API_URL or SEARCH_API_KEY not set, returning empty results")
        return {"items": []}

    serper_url = search_url if "serper" in search_url else "https://google.serper.dev/search"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                serper_url,
                json={"q": query_text, "num": min(limit, 30)},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            organic = data.get("organic") or data.get("results") or []
            items = []
            for i, row in enumerate((organic or [])[:limit]):
                item = {
                    "title": row.get("title") or "",
                    "url": row.get("link") or row.get("url") or "",
                    "price": row.get("price"),
                    "source": row.get("source"),
                    "position": i + 1,
                    "snippet": row.get("snippet"),
                }
                items.append(item)
            if logger:
                logger.info("Shop-assistant SEARCH completed: query=%s, count=%s", query_text[:80], len(items))
            return {"items": items}
    except Exception as e:
        if logger:
            logger.error("Shop-assistant SEARCH failed: %s", e)
        return {"items": []}


def _fallback_presentation(results: List[Dict], query_text: str) -> str:
    """Fallback formatted list when LLM is unavailable."""
    lines = []
    for i, r in enumerate((results or [])[:20]):
        title = r.get("title") or ""
        price = f" — {r['price']}" if r.get("price") else ""
        source = f" ({r['source']})" if r.get("source") else ""
        url = f"\n   {r['url']}" if r.get("url") else ""
        lines.append(f"{i + 1}. {title}{price}{source}{url}")
    return f"Found {len(results)} results for: {query_text}\n\n" + "\n".join(lines) + "\n\nClick a link to open the product."


async def agent_format_presentation(
    results: List[Dict[str, Any]],
    query_text: str,
    role: str = "default",
    prompt_content: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    PRESENTATION agent: format search results for user chat (dialog/tables/photos).
    Uses NLP content_generation. Optional prompt_content/model override from admin.
    """
    nlp_url = _get_nlp_url()
    search_results_text = "\n".join(
        f"{i+1}. {r.get('title','')}{' — ' + str(r.get('price','')) if r.get('price') else ''}"
        f"{' (' + str(r.get('source','')) + ')' if r.get('source') else ''}"
        f"{': ' + (str(r.get('snippet',''))[:120] + ('…' if len(str(r.get('snippet',''))) > 120 else '')) if r.get('snippet') else ''}"
        for i, r in enumerate((results or [])[:30])
    )

    if prompt_content:
        text_content = (
            prompt_content.replace("{{searchResults}}", search_results_text)
            .replace("{{search_results}}", search_results_text)
            .replace("{{queryText}}", query_text)
            .replace("{{query_text}}", query_text)
        )
    else:
        text_content = (
            DEFAULT_PRESENTATION_PROMPT
            .replace("{{searchResults}}", search_results_text)
            .replace("{{queryText}}", query_text)
        )

    body: Dict[str, Any] = {
        "text_content": text_content,
        "analysis_type": "content_generation",
        "user_name": "shop-assistant",
    }
    if model:
        body["model"] = model

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{nlp_url}/api/analyze-text", json=body)
            r.raise_for_status()
            data = r.json()
            analysis = data.get("analysis") or data
            content = (
                analysis.get("summary")
                or (analysis.get("key_insights") and analysis["key_insights"][0])
                or analysis.get("text")
                or ""
            )
            if isinstance(content, list):
                content = content[0] if content else ""
            formatted = (content if isinstance(content, str) else str(content)).strip()
            if not formatted:
                formatted = _fallback_presentation(results, query_text)
            if logger:
                logger.info("Shop-assistant PRESENTATION format success, length=%s", len(formatted))
            return {"formatted_content": formatted}
    except Exception as e:
        if logger:
            logger.warning("Shop-assistant format-presentation failed, using fallback: %s", e)
        return {"formatted_content": _fallback_presentation(results, query_text)}


async def agent_compare_prices(
    results: List[Dict[str, Any]],
    query_text: str,
    role: str = "default",
    prompt_content: Optional[str] = None,
    model: Optional[str] = None,
    priority_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    COMPARISON agent: compare prices (and other priorities) for search results.
    Uses NLP content_generation with optional prompt_content/model override from admin.
    """
    nlp_url = _get_nlp_url()
    search_results_text = "\n".join(
        f"{i+1}. {r.get('title','')}"
        f"{' — ' + str(r.get('price','')) if r.get('price') else ''}"
        f"{' (' + str(r.get('source','')) + ')' if r.get('source') else ''}"
        for i, r in enumerate((results or [])[:30])
    )
    priority_order_str = ", ".join(priority_order) if priority_order else "not specified"

    if prompt_content:
        text_content = (
            prompt_content.replace("{{searchResults}}", search_results_text)
            .replace("{{search_results}}", search_results_text)
            .replace("{{queryText}}", query_text)
            .replace("{{query_text}}", query_text)
            .replace("{{priorityOrder}}", priority_order_str)
        )
    else:
        text_content = (
            DEFAULT_COMPARISON_PROMPT
            .replace("{{searchResults}}", search_results_text)
            .replace("{{queryText}}", query_text)
            .replace("{{priorityOrder}}", priority_order_str)
        )

    body: Dict[str, Any] = {
        "text_content": text_content,
        "analysis_type": "content_generation",
        "user_name": "shop-assistant",
    }
    if model:
        body["model"] = model

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{nlp_url}/api/analyze-text", json=body)
            r.raise_for_status()
            data = r.json()
            analysis = data.get("analysis") or data
            content = (
                analysis.get("summary")
                or (analysis.get("key_insights") and analysis["key_insights"][0])
                or analysis.get("text")
                or ""
            )
            if isinstance(content, list):
                content = content[0] if content else ""
            summary = (content if isinstance(content, str) else str(content)).strip()
            if logger:
                logger.info("Shop-assistant COMPARISON success, length=%s", len(summary))
            return {"summary": summary}
    except Exception as e:
        if logger:
            logger.warning("Shop-assistant COMPARISON failed: %s", e)
        return {"summary": ""}


async def agent_extract_location(
    user_text: str,
    query_text: str,
    role: str = "default",
    prompt_content: Optional[str] = None,
    model: Optional[str] = None,
    priority_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    LOCATION agent: extract or validate delivery region / shipping location.
    Returns short region phrase and augmented_query (suffix for search) when applicable.
    """
    nlp_url = _get_nlp_url()
    priority_order_str = ", ".join(priority_order) if priority_order else "not specified"

    if prompt_content:
        text_content = (
            prompt_content.replace("{{userInput}}", user_text)
            .replace("{{user_input}}", user_text)
            .replace("{{queryText}}", query_text)
            .replace("{{query_text}}", query_text)
            .replace("{{previousParams}}", "")
            .replace("{{priorityOrder}}", priority_order_str)
        )
    else:
        text_content = (
            DEFAULT_LOCATION_PROMPT
            .replace("{{userInput}}", user_text)
            .replace("{{queryText}}", query_text)
            .replace("{{priorityOrder}}", priority_order_str)
        )

    body: Dict[str, Any] = {
        "text_content": text_content,
        "analysis_type": "content_generation",
        "user_name": "shop-assistant",
    }
    if model:
        body["model"] = model

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{nlp_url}/api/analyze-text", json=body)
            r.raise_for_status()
            data = r.json()
            analysis = data.get("analysis") or data
            content = (
                analysis.get("summary")
                or (analysis.get("key_insights") and analysis["key_insights"][0])
                or analysis.get("text")
                or ""
            )
            if isinstance(content, list):
                content = content[0] if content else ""
            reply = (content if isinstance(content, str) else str(content)).strip()
            region = None
            augmented_query = None
            if reply and len(reply) < 150:
                region = reply
                augmented_query = reply
            if logger:
                logger.info("Shop-assistant LOCATION success, reply_length=%s", len(reply))
            return {"region": region, "augmented_query": augmented_query}
    except Exception as e:
        if logger:
            logger.warning("Shop-assistant LOCATION failed: %s", e)
        return {"region": None, "augmented_query": None}

