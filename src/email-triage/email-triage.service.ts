import { Injectable, Logger } from '@nestjs/common';

const MAX_ITEMS = 30;
const INTENTS_PRIMARY = ['support', 'sales', 'contract', 'technical', 'billing', 'spam'] as const;
type PrimaryIntent = (typeof INTENTS_PRIMARY)[number];
type Intent = PrimaryIntent | 'unknown' | 'multi_intent';

const KEYWORDS: Record<PrimaryIntent, RegExp> = {
  billing: /\b(rechnung|invoice|zahlung|payment|kosten|preis|refund|rückerstattung|abbuchung|debit|charges|active|verify|credit|card|retry|accounting|copy|charged|subscription|duplicate|twice|billing|address|invoices|VAT|finance|downgrade|enterprise|plan|steps|bank|transfer|confirmation|reflected)\b/gi,
  contract: /\b(vertrag|contract|kündigung|cancel|änderung|change|agb|terms|renewal|notice|automatic|months|amendment|subsidiaries|agreement|additional|existing|data|processing|DPA|GDPR|compliance|legal|standard|termination|clause|section|reviewing|clarify|conditions|SLA|uptime|guarantees|procurement|approval|start|date|signed|period|begins|entity|merged|organization|update|liability|limitations|situations|covered)\b/gi,
  technical: /\b(verbindung|connection|internet|störung|outage|fehler|error|router|modem|technisch|API|endpoint|integration|responses|webhook|triggering|events|configured|trigger|authentication|key|token|rotated|requests|rejected|troubleshoot|workflow|stuck|processing|automation|completes|latency|response|dashboard|load|upload|file|failing|size|limit|Slack|notifications|broke|AI|classification|misclassified|spam|retrain|model|SSO|login|redirect|export|data|job|failed|dataset|investigate|account)\b/gi,
  sales: /\b(angebot|offer|kaufen|buy|tarif|plan|bestellen|order|pricing|enterprise|evaluating|employees|demo|product|webinar|volume|discounts|deployments|onboard|comparing|platform|alternatives|capabilities|ServiceNow|evaluation|confirm|incident|startup|trial|access|test|security|certifications|SOC2|ISO27001|certified|automation|email|processing|ticket|routing|tiers|information)\b/gi,
  spam: /\b(casino|lottery|winner|click here|unsubscribe|opt.?out|money|rich|opportunity|reply|SEO|website|Google|guaranteed|crypto|investment|trading|returns|newsletter|subscription|confirmation|test|message|monitoring|office)\b/gi,
  support: /\b(hilfe|help|frage|question|problem|beschwerde|complaint|support|check|dashboard|portal|account|access|empty|unable|not working|doesn't work|doesnt work|logged|page|permissions|admin|reporting|restore|guide|removed|unlock|reset|export|reports|analytics|excel|onboarding|documentation|configure|workflows|invitations|invitation|expired|link|invite|integration|setup|assist|connect|instructions|mobile|app|login|logs|training|session|automation|features|adopted|struggling|schedule)\b/gi,
};

const AMOUNT_RE = /\b(\d+(?:[.,]\d{2})?)\s*(\€|eur|euro|chf|usd|\$|kč|czk)?\b/gi;
const DATE_RE = /\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b|\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b/g;
const CONTRACT_REF = /\b(vertrag|contract|auftrag|order)\s*#?\s*([A-Z0-9-]+)\b/gi;
const PRODUCT_REF = /\b(artikel|article|produkt|product)\s*#?\s*([A-Z0-9-]+)\b/gi;

export interface NormalizedEmail {
  message_id: string;
  tenant_id: string;
  timestamp: string | number;
  sender: string;
  recipients: string[];
  subject: string;
  body_plain: string;
  body_html: string;
  attachments: unknown[];
  locale?: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResult {
  success: boolean;
  payload?: NormalizedEmail;
  error?: string;
  escalation_reason?: string;
  duration_ms?: number;
  model_used?: string;
}

export interface ClassifyResult {
  intent: Intent;
  confidence: number;
  raw_scores?: Record<string, number>;
  model_used?: string;
  llm_output?: unknown;
}

export interface ExtractResult {
  message_id: string;
  entities: {
    product_refs: string[];
    amounts: Array<{ value: number; unit: string | null }>;
    dates: string[];
    contract_refs: string[];
  };
  summary: string | null;
}

export interface DecideResult {
  action: string;
  escalation_reason: string | null;
  queue: string | null;
  model_used?: string;
}

@Injectable()
export class EmailTriageService {
  private readonly logger = new Logger(EmailTriageService.name);

  private getConfidenceThreshold(): number {
    const v = process.env.CLASSIFIER_CONFIDENCE_THRESHOLD;
    if (v) {
      const n = parseFloat(v);
      if (!isNaN(n) && n >= 0 && n <= 1) return n;
    }
    return 0.75;
  }

  private textFromPayload(payload: Record<string, unknown>): string {
    const parts: string[] = [];
    if (payload.subject) parts.push(String(payload.subject).trim());
    if (payload.body_plain) {
      parts.push(String(payload.body_plain).trim());
    } else if (payload.body_html) {
      const text = String(payload.body_html)
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      parts.push(text);
    }
    return parts.join(' ').trim();
  }

  validateAndNormalize(raw: unknown): { payload: NormalizedEmail | null; error: string | null; escalation_reason: string | null } {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      return { payload: null, error: 'Payload must be an object', escalation_reason: 'incomplete_data' };
    }
    const r = raw as Record<string, unknown>;

    const messageId = r.message_id != null ? String(r.message_id).trim() : null;
    if (!messageId) {
      return { payload: null, error: 'message_id is required', escalation_reason: 'incomplete_data' };
    }

    const tenantId = r.tenant_id != null ? String(r.tenant_id).trim() : null;
    if (!tenantId) {
      return { payload: null, error: 'tenant_id is required', escalation_reason: 'incomplete_data' };
    }

    if (r.timestamp == null) {
      return { payload: null, error: 'timestamp is required', escalation_reason: 'incomplete_data' };
    }

    const bodyPlain = r.body_plain != null ? String(r.body_plain).trim() : '';
    const bodyHtml = r.body_html != null ? String(r.body_html).trim() : '';
    if (!bodyPlain && !bodyHtml) {
      return { payload: null, error: 'At least one of body_plain or body_html is required', escalation_reason: 'incomplete_data' };
    }

    let recipients: unknown[] = [];
    if (Array.isArray(r.recipients)) {
      if (r.recipients.length > MAX_ITEMS) {
        return { payload: null, error: `recipients length must be ≤ ${MAX_ITEMS}`, escalation_reason: 'incomplete_data' };
      }
      recipients = r.recipients;
    }

    let attachments: unknown[] = [];
    if (Array.isArray(r.attachments)) {
      if (r.attachments.length > MAX_ITEMS) {
        return { payload: null, error: `attachments length must be ≤ ${MAX_ITEMS}`, escalation_reason: 'incomplete_data' };
      }
      attachments = r.attachments;
    }

    const tsOut = typeof r.timestamp === 'number' ? r.timestamp : String(r.timestamp);
    const normalized: NormalizedEmail = {
      message_id: messageId,
      tenant_id: tenantId,
      timestamp: tsOut,
      sender: r.sender != null ? String(r.sender).trim() : '',
      recipients: recipients as string[],
      subject: r.subject != null ? String(r.subject).trim() : '',
      body_plain: bodyPlain,
      body_html: bodyHtml,
      attachments,
    };
    if (r.locale != null) normalized.locale = String(r.locale).trim();
    if (r.metadata && typeof r.metadata === 'object' && !Array.isArray(r.metadata)) {
      normalized.metadata = r.metadata as Record<string, unknown>;
    }

    return { payload: normalized, error: null, escalation_reason: null };
  }

  getEmailTextForLlm(payload: Record<string, unknown>): string {
    return this.textFromPayload(payload);
  }

  classifyPayload(payload: Record<string, unknown>, threshold?: number): ClassifyResult {
    const t = threshold ?? this.getConfidenceThreshold();
    const text = this.textFromPayload(payload);
    if (!text) {
      return { intent: 'unknown', confidence: 0.0, raw_scores: Object.fromEntries(INTENTS_PRIMARY.map((k) => [k, 0.2])) };
    }

    const rawScores: Record<string, number> = {};
    const matchCounts: Record<string, number> = {};
    for (const intent of INTENTS_PRIMARY) {
      const matches = text.match(new RegExp(KEYWORDS[intent].source, 'gi')) ?? [];
      matchCounts[intent] = matches.length;
      rawScores[intent] = matches.length > 0 ? Math.min(0.5 + matches.length * 0.15, 0.95) : 0.2;
    }

    const entries = Object.entries(rawScores).filter(([, v]) => v > 0.2);
    entries.sort(([ka, va], [kb, vb]) => vb - va || (matchCounts[kb] ?? 0) - (matchCounts[ka] ?? 0));

    if (!entries.length) {
      return { intent: 'unknown', confidence: 0.2, raw_scores: rawScores };
    }

    const [topIntent, topScore] = entries[0];
    const [secondIntent, secondScore] = entries[1] ?? [null, 0];

    if (topScore >= t && secondScore >= t) {
      const topMatches = matchCounts[topIntent] ?? 0;
      const secondMatches = secondIntent ? (matchCounts[secondIntent] ?? 0) : 0;
      if (topMatches > secondMatches) {
        return { intent: topIntent as Intent, confidence: topScore, raw_scores: rawScores };
      }
      return { intent: 'multi_intent', confidence: (topScore + secondScore) / 2, raw_scores: rawScores };
    }
    if (topScore < t) {
      return { intent: 'unknown', confidence: topScore, raw_scores: rawScores };
    }
    return { intent: topIntent as Intent, confidence: topScore, raw_scores: rawScores };
  }

  extractPayload(payload: Record<string, unknown>, intent?: string): ExtractResult {
    const messageId = payload.message_id ? String(payload.message_id).trim() : 'unknown';
    const text = this.textFromPayload(payload);

    const amounts: Array<{ value: number; unit: string | null }> = [];
    const dates: string[] = [];
    const contractRefs: string[] = [];
    const productRefs: string[] = [];

    let m: RegExpExecArray | null;
    const amountRe = new RegExp(AMOUNT_RE.source, 'gi');
    while ((m = amountRe.exec(text)) !== null) {
      const valueStr = m[1].replace(',', '.');
      const unit = m[2] ? m[2].trim().toUpperCase() : null;
      const value = parseFloat(valueStr);
      if (!isNaN(value)) amounts.push({ value, unit });
    }

    const dateRe = new RegExp(DATE_RE.source, 'g');
    while ((m = dateRe.exec(text)) !== null) dates.push(m[0]);

    const contractRe = new RegExp(CONTRACT_REF.source, 'gi');
    while ((m = contractRe.exec(text)) !== null) contractRefs.push(m[2]);

    const productRe = new RegExp(PRODUCT_REF.source, 'gi');
    while ((m = productRe.exec(text)) !== null) productRefs.push(m[2]);

    const summaryParts: string[] = [];
    if (amounts.length) summaryParts.push('amount reference');
    if (contractRefs.length) summaryParts.push('contract reference');
    if (intent) summaryParts.push(`intent:${intent}`);
    const summary = summaryParts.length ? summaryParts.join('; ') : null;

    return { message_id: messageId, entities: { product_refs: productRefs, amounts, dates, contract_refs: contractRefs }, summary };
  }

  decideAction(intent: string, confidence: number, threshold?: number, entities?: Record<string, unknown>): DecideResult {
    const t = threshold ?? this.getConfidenceThreshold();

    if (intent === 'unknown' || intent === 'multi_intent') {
      const reason = intent === 'unknown' ? 'ambiguous_intent' : 'multi_intent';
      return { action: 'escalate', escalation_reason: reason, queue: null };
    }

    if (confidence < t) {
      return { action: 'escalate', escalation_reason: 'low_confidence', queue: null };
    }

    if (intent === 'contract') {
      return { action: 'escalate', escalation_reason: 'contract_change', queue: null };
    }

    const queueByIntent: Record<string, string> = {
      support: 'support',
      sales: 'sales',
      technical: 'technical',
      billing: 'billing',
      spam: 'spam_review',
    };
    const queue = queueByIntent[intent] ?? 'support';

    const autoRespondEnabled = ['1', 'true', 'yes'].includes(
      (process.env.AUTO_RESPOND_ENABLED ?? '').trim().toLowerCase(),
    );
    if (autoRespondEnabled && intent === 'support' && confidence >= 0.85) {
      return { action: 'auto_respond', escalation_reason: null, queue: null };
    }

    return { action: 'route_to_queue', escalation_reason: null, queue };
  }
}
