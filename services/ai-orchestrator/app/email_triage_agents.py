"""
Email-triage agents: Ingest, Classifier, Extractor, Action/Decider.
Contracts: docs/contracts (agentic-email-processing-system): email-schema, intent-taxonomy,
extractor-contract, action-set, routing-rules, escalation-contract.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Logger set by main.py
logger = None

MAX_ITEMS = 30
INTENTS_PRIMARY = ["support", "sales", "contract", "technical", "billing", "spam"]
INTENTS_ALL = INTENTS_PRIMARY + ["unknown", "multi_intent"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Keywords (DE/EN) per primary intent — prototype; align with intent-taxonomy
KEYWORDS = {
    "billing": re.compile(
        r"\b(rechnung|invoice|zahlung|payment|kosten|preis|refund|rückerstattung|abbuchung|debit)\b",
        re.I,
    ),
    "contract": re.compile(
        r"\b(vertrag|contract|kündigung|cancel|änderung|change|agb|terms)\b",
        re.I,
    ),
    "technical": re.compile(
        r"\b(verbindung|connection|internet|störung|outage|fehler|error|router|modem|technisch)\b",
        re.I,
    ),
    "sales": re.compile(
        r"\b(angebot|offer|kaufen|buy|tarif|plan|bestellen|order)\b",
        re.I,
    ),
    "spam": re.compile(
        r"\b(casino|lottery|winner|click here|unsubscribe|opt.?out)\b",
        re.I,
    ),
    "support": re.compile(
        r"\b(hilfe|help|frage|question|problem|beschwerde|complaint|support)\b",
        re.I,
    ),
}


def set_logger(l):
    """Set module logger (called from main.py)."""
    global logger
    logger = l


def _log(msg: str, *args: Any, **kwargs: Any) -> None:
    if logger:
        logger.info(msg, *args, **kwargs)
    elif kwargs:
        import logging
        logging.getLogger(__name__).info(msg, *args, **kwargs)


def validate_and_normalize(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Ingest: validate per email-schema; return (normalized_payload, error, escalation_reason).
    On success: (payload, None, None). On failure: (None, error, escalation_reason).
    """
    if not raw or not isinstance(raw, dict):
        return None, "Payload must be an object", "incomplete_data"

    message_id = raw.get("message_id")
    if message_id is None:
        return None, "message_id is required", "incomplete_data"
    message_id = str(message_id).strip()
    if not message_id:
        return None, "message_id is required", "incomplete_data"

    tenant_id = raw.get("tenant_id")
    if tenant_id is None:
        return None, "tenant_id is required", "incomplete_data"
    tenant_id = str(tenant_id).strip()
    if not tenant_id:
        return None, "tenant_id is required", "incomplete_data"

    timestamp = raw.get("timestamp")
    if timestamp is None:
        return None, "timestamp is required", "incomplete_data"

    body_plain = (raw.get("body_plain") or "").strip() if raw.get("body_plain") is not None else ""
    body_html = (raw.get("body_html") or "").strip() if raw.get("body_html") is not None else ""
    if not body_plain and not body_html:
        return None, "At least one of body_plain or body_html is required", "incomplete_data"

    recipients = raw.get("recipients")
    if isinstance(recipients, list):
        if len(recipients) > MAX_ITEMS:
            return None, f"recipients length must be ≤ {MAX_ITEMS}", "incomplete_data"
    else:
        recipients = []

    attachments = raw.get("attachments")
    if isinstance(attachments, list):
        if len(attachments) > MAX_ITEMS:
            return None, f"attachments length must be ≤ {MAX_ITEMS}", "incomplete_data"
    else:
        attachments = []

    ts_out = timestamp if isinstance(timestamp, (int, float)) else str(timestamp)
    normalized = {
        "message_id": message_id,
        "tenant_id": tenant_id,
        "timestamp": ts_out,
        "sender": str(raw.get("sender", "")).strip() if raw.get("sender") is not None else "",
        "recipients": list(recipients),
        "subject": str(raw.get("subject", "")).strip() if raw.get("subject") is not None else "",
        "body_plain": body_plain or "",
        "body_html": body_html or "",
        "attachments": list(attachments),
    }
    if raw.get("locale") is not None:
        normalized["locale"] = str(raw["locale"]).strip()
    if isinstance(raw.get("metadata"), dict):
        normalized["metadata"] = raw["metadata"]
    return normalized, None, None


def _text_from_payload(payload: Dict[str, Any]) -> str:
    """Build combined text from subject + body_plain or stripped body_html."""
    parts = []
    if payload.get("subject"):
        parts.append(str(payload["subject"]).strip())
    if payload.get("body_plain"):
        parts.append(str(payload["body_plain"]).strip())
    elif payload.get("body_html"):
        html = str(payload["body_html"])
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        parts.append(text)
    return " ".join(parts).strip()


def _get_confidence_threshold() -> float:
    """From env CLASSIFIER_CONFIDENCE_THRESHOLD; default 0.75."""
    v = os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD")
    if v is None or v == "":
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        n = float(v)
        if 0 <= n <= 1:
            return n
    except (ValueError, TypeError):
        pass
    return DEFAULT_CONFIDENCE_THRESHOLD


def classify_payload(payload: Dict[str, Any], threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Classify: intent + confidence + raw_scores per intent-taxonomy.
    Below threshold -> unknown; two or more above threshold -> multi_intent.
    """
    if threshold is None:
        threshold = _get_confidence_threshold()
    text = _text_from_payload(payload)
    if not text:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "raw_scores": {k: 0.2 for k in INTENTS_PRIMARY},
        }

    raw_scores: Dict[str, float] = {}
    for intent, pattern in KEYWORDS.items():
        matches = pattern.findall(text)
        raw_scores[intent] = min(0.5 + len(matches) * 0.15, 0.95) if matches else 0.2

    entries = [(k, v) for k, v in raw_scores.items() if v > 0.2]
    by_score = sorted(entries, key=lambda x: -x[1])

    if not by_score:
        return {
            "intent": "unknown",
            "confidence": 0.2,
            "raw_scores": raw_scores,
        }

    top_intent, top_score = by_score[0]
    second_score = by_score[1][1] if len(by_score) > 1 else 0.0

    if top_score >= threshold and second_score >= threshold:
        return {
            "intent": "multi_intent",
            "confidence": (top_score + second_score) / 2,
            "raw_scores": raw_scores,
        }
    if top_score < threshold:
        return {
            "intent": "unknown",
            "confidence": top_score,
            "raw_scores": raw_scores,
        }
    return {
        "intent": top_intent,
        "confidence": top_score,
        "raw_scores": raw_scores,
    }


# --- Extractor (Phase 2): entities from normalized payload; no PII in output ---
# Simple regex-based extraction; shape per extractor-contract
_AMOUNT_RE = re.compile(r"\b(\d+(?:[.,]\d{2})?)\s*(\€|eur|euro|chf|usd|\$|kč|czk)?\b", re.I)
_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b|\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b")
_CONTRACT_REF = re.compile(r"\b(vertrag|contract|auftrag|order)\s*#?\s*([A-Z0-9-]+)\b", re.I)
_PRODUCT_REF = re.compile(r"\b(artikel|article|produkt|product)\s*#?\s*([A-Z0-9-]+)\b", re.I)


def extract_payload(payload: Dict[str, Any], intent: Optional[str] = None) -> Dict[str, Any]:
    """
    Extractor: extract entities from normalized email per extractor-contract.
    Returns { message_id, entities, summary? }. No PII in entities or summary.
    """
    message_id = (payload.get("message_id") or "").strip() or "unknown"
    text = _text_from_payload(payload)

    entities: Dict[str, Any] = {
        "product_refs": [],
        "amounts": [],
        "dates": [],
        "contract_refs": [],
    }

    for m in _AMOUNT_RE.finditer(text):
        value_str = m.group(1).replace(",", ".")
        unit = (m.group(2) or "").strip().upper() or None
        try:
            value = float(value_str)
            entities["amounts"].append({"value": value, "unit": unit})
        except ValueError:
            pass

    for m in _DATE_RE.finditer(text):
        entities["dates"].append(m.group(0))

    for m in _CONTRACT_REF.finditer(text):
        entities["contract_refs"].append(m.group(2))

    for m in _PRODUCT_REF.finditer(text):
        entities["product_refs"].append(m.group(2))

    summary_parts = []
    if entities["amounts"]:
        summary_parts.append("amount reference")
    if entities["contract_refs"]:
        summary_parts.append("contract reference")
    if intent:
        summary_parts.append(f"intent:{intent}")
    summary = "; ".join(summary_parts) if summary_parts else None

    return {
        "message_id": message_id,
        "entities": entities,
        "summary": summary,
    }


# --- Action/Decider (Phase 2): intent + confidence + optional entities -> action per routing-rules ---
ACTIONS = ["auto_respond", "route_to_queue", "escalate"]


def _get_auto_respond_enabled() -> bool:
    v = os.getenv("AUTO_RESPOND_ENABLED", "").strip().lower()
    return v in ("1", "true", "yes")


def decide_action(
    intent: str,
    confidence: float,
    threshold: Optional[float] = None,
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Decider: choose action per action-set and routing-rules.
    Returns { action, escalation_reason?, queue? }.
    unknown/multi_intent/below threshold/contract -> escalate; else route_to_queue with queue by intent.
    """
    if threshold is None:
        threshold = _get_confidence_threshold()
    entities = entities or {}

    if intent in ("unknown", "multi_intent"):
        reason = "ambiguous_intent" if intent == "unknown" else "multi_intent"
        return {"action": "escalate", "escalation_reason": reason, "queue": None}

    if confidence < threshold:
        return {"action": "escalate", "escalation_reason": "low_confidence", "queue": None}

    if intent == "contract":
        return {"action": "escalate", "escalation_reason": "contract_change", "queue": None}

    queue_by_intent = {
        "support": "support",
        "sales": "sales",
        "technical": "technical",
        "billing": "billing",
        "spam": "spam_review",
    }
    queue = queue_by_intent.get(intent, "support")

    if _get_auto_respond_enabled() and intent == "support" and confidence >= 0.85:
        return {"action": "auto_respond", "escalation_reason": None, "queue": None}

    return {"action": "route_to_queue", "escalation_reason": None, "queue": queue}
