import { Injectable, Logger } from '@nestjs/common';
import { AiService } from '../ai/ai.service';
import { ModelTierSchema, type ModelTier } from '../contracts/ai-complete.contract';

function resolveModelTier(model?: string): ModelTier {
  const parsed = ModelTierSchema.safeParse(model ?? 'free');
  return parsed.success ? parsed.data : 'free';
}

const DEFAULT_REFINE_QUERY_PROMPT =
  'Extract a single web product search query from: "{{user_input}}". Reply with ONE short search query only (max 200 chars), no other text.';
const DEFAULT_REFINE_QUERY_PROMPT_WITH_PARAMS =
  'Refine this product search. User said: "{{user_input}}". Previous constraints: {{previous_params}}. Reply with ONE short search query only (max 200 chars), no other text.';
const DEFAULT_PRESENTATION_PROMPT =
  'Format these product search results for the user in a clear, readable way (dialog or list). Query: {{queryText}}. Results: {{searchResults}}. Return only the formatted text, no extra commentary.';
const DEFAULT_COMPARISON_PROMPT =
  'You are a shopping assistant comparing products for a user.\n\nUser query: {{queryText}}\nPriorities (price, quality, location, etc.): {{priorityOrder}}\n\nSearch results:\n{{searchResults}}\n\nWrite a short comparison focusing on the user priorities. Recommend several best options and explain briefly. Keep it concise.';
const DEFAULT_LOCATION_PROMPT =
  'User is shopping online.\n\nUser input: {{userInput}}\nCurrent query: {{queryText}}\nPriorities (price, quality, location, etc.): {{priorityOrder}}\n\nExtract a short phrase describing the delivery region or shipping location that matters for this request, for example "delivery Czech Republic" or "ships to EU". If region is not specified, respond with an empty string.\nAnswer with this short phrase only, no other text.';

export interface SearchItem {
  title: string;
  url: string;
  price?: unknown;
  source?: string;
  position: number;
  snippet?: string;
}

@Injectable()
export class ShopAssistantService {
  private readonly logger = new Logger(ShopAssistantService.name);

  constructor(private readonly aiService: AiService) {}

  private getAsrUrl(): string {
    return (process.env.ASR_SERVICE_URL ?? 'http://asr-service:3382').replace(/\/$/, '');
  }

  private getSearchConfig(): { url: string; key: string } {
    return {
      url: (process.env.SEARCH_API_URL ?? '').trim(),
      key: (process.env.SEARCH_API_KEY ?? '').trim(),
    };
  }

  private extractTextFromAiResult(result: Record<string, unknown>): string {
    const analysis = (result.analysis as Record<string, unknown> | undefined) ?? result;
    let content: unknown =
      analysis.summary ??
      (Array.isArray(analysis.key_insights) && analysis.key_insights[0]) ??
      analysis.text ??
      result.text ??
      '';
    if (Array.isArray(content)) content = content[0] ?? '';
    return (typeof content === 'string' ? content : String(content)).trim();
  }

  async transcribe(voiceFileUrl: string): Promise<{ transcript: string }> {
    if (!voiceFileUrl) return { transcript: '' };
    const asrUrl = this.getAsrUrl();
    const res = await fetch(`${asrUrl}/api/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voice_file_url: voiceFileUrl }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) throw new Error(`ASR service error ${res.status}`);
    const data = (await res.json()) as Record<string, unknown>;
    const transcript = String((data.transcript ?? data.text ?? '') as string);
    return { transcript };
  }

  async refineQuery(
    userText: string,
    previousParams?: Record<string, unknown>,
    _role = 'default',
    promptContent?: string,
    model?: string,
  ): Promise<{ query_text: string; refined_params: Record<string, unknown> }> {
    if (!userText?.trim()) return { query_text: '', refined_params: previousParams ?? {} };

    let textContent: string;
    if (promptContent) {
      textContent = promptContent
        .replace(/{{user_input}}/g, userText)
        .replace(/{{userInput}}/g, userText)
        .replace(/{{previous_params}}/g, previousParams ? String(previousParams) : '')
        .replace(/{{previousParams}}/g, previousParams ? String(previousParams) : '');
    } else if (previousParams && Object.keys(previousParams).length > 0) {
      textContent = DEFAULT_REFINE_QUERY_PROMPT_WITH_PARAMS
        .replace('{{user_input}}', userText)
        .replace('{{previous_params}}', String(previousParams));
    } else {
      textContent = DEFAULT_REFINE_QUERY_PROMPT.replace('{{user_input}}', userText);
    }

    const result = await this.aiService.complete({
      model_tier: resolveModelTier(model),
      system_prompt: 'You are a search query refiner. Output only the refined query, no explanation.',
      user_prompt: textContent,
      max_tokens: 100,
    });

    const queryText = ((result.text as string | undefined) ?? '').slice(0, 200).trim() || userText.trim();
    return { query_text: queryText, refined_params: previousParams ?? {} };
  }

  private decodeHtml(value: string): string {
    return value
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&#(\d+);/g, (_match, code: string) => String.fromCharCode(Number(code)));
  }

  private stripHtml(value: string): string {
    return this.decodeHtml(value.replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim();
  }

  private normalizeDuckDuckGoUrl(rawUrl: string): string | null {
    const decoded = this.decodeHtml(rawUrl);
    try {
      const parsed = new URL(decoded, 'https://duckduckgo.com');
      const redirected = parsed.searchParams.get('uddg');
      const candidate = redirected ? new URL(redirected) : parsed;
      if (candidate.protocol !== 'http:' && candidate.protocol !== 'https:') return null;
      return candidate.toString();
    } catch {
      return null;
    }
  }

  private async searchDuckDuckGo(queryText: string, limit: number, searchUrl?: string): Promise<{ items: SearchItem[] }> {
    const baseUrl = searchUrl?.trim() || 'https://html.duckduckgo.com/html/';
    const normalizedQuery = queryText
      .replace(/[“”]/g, '"')
      .replace(/[‘’]/g, "'")
      .replace(/^["']+|["']+$/g, '')
      .trim();
    const url = new URL(baseUrl);
    url.searchParams.set('q', normalizedQuery || queryText);

    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; AlfaresShopAssistant/1.0)',
        Accept: 'text/html,application/xhtml+xml',
      },
      signal: AbortSignal.timeout(15_000),
    });
    if (!res.ok) throw new Error('DuckDuckGo search error ' + res.status);

    const html = await res.text();
    const matches = Array.from(html.matchAll(/<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g));
    const items: SearchItem[] = [];
    const seen = new Set<string>();

    for (const match of matches) {
      const url = this.normalizeDuckDuckGoUrl(match[1] ?? '');
      const title = this.stripHtml(match[2] ?? '');
      if (!url || !title || seen.has(url)) continue;
      seen.add(url);
      items.push({
        title,
        url,
        source: new URL(url).hostname.replace(/^www\./, ''),
        position: items.length + 1,
      });
      if (items.length >= limit) break;
    }

    return { items };
  }

  async search(queryText: string, limit = 20): Promise<{ items: SearchItem[] }> {
    const { url: searchUrl, key: apiKey } = this.getSearchConfig();
    const normalizedLimit = Math.min(limit, 30);

    if (!searchUrl || searchUrl.includes('duckduckgo.com')) {
      return this.searchDuckDuckGo(queryText, normalizedLimit, searchUrl);
    }

    if (!apiKey) {
      throw new Error('SEARCH_API_KEY not set; cannot run configured search provider');
    }

    const serperUrl = searchUrl.includes('serper') ? searchUrl : 'https://google.serper.dev/search';
    const res = await fetch(serperUrl, {
      method: 'POST',
      headers: { 'X-API-KEY': apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: queryText, num: normalizedLimit }),
      signal: AbortSignal.timeout(15_000),
    });
    if (!res.ok) throw new Error('Search API error ' + res.status);
    const data = (await res.json()) as Record<string, unknown>;
    const organic = ((data.organic ?? data.results ?? []) as Record<string, unknown>[]).slice(0, normalizedLimit);
    const items: SearchItem[] = organic.map((row, i) => ({
      title: String(row.title ?? ''),
      url: String(row.link ?? row.url ?? ''),
      price: row.price,
      source: row.source != null ? String(row.source) : undefined,
      position: i + 1,
      snippet: row.snippet != null ? String(row.snippet) : undefined,
    }));
    return { items };
  }

  async formatPresentation(
    results: Record<string, unknown>[],
    queryText: string,
    _role = 'default',
    promptContent?: string,
    model?: string,
  ): Promise<{ formatted_content: string }> {
    const searchResultsText = (results ?? [])
      .slice(0, 30)
      .map((r, i) => {
        const price = r.price ? ` — ${String(r.price)}` : '';
        const source = r.source ? ` (${String(r.source)})` : '';
        const snippet = r.snippet ? `: ${String(r.snippet).slice(0, 120)}${String(r.snippet).length > 120 ? '…' : ''}` : '';
        return `${i + 1}. ${String(r.title ?? '')}${price}${source}${snippet}`;
      })
      .join('\n');

    let textContent: string;
    if (promptContent) {
      textContent = promptContent
        .replace(/{{searchResults}}/g, searchResultsText)
        .replace(/{{search_results}}/g, searchResultsText)
        .replace(/{{queryText}}/g, queryText)
        .replace(/{{query_text}}/g, queryText);
    } else {
      textContent = DEFAULT_PRESENTATION_PROMPT
        .replace('{{searchResults}}', searchResultsText)
        .replace('{{queryText}}', queryText);
    }

    const result = await this.aiService.complete({
      model_tier: resolveModelTier(model),
      system_prompt: 'Format search results clearly for the user.',
      user_prompt: textContent,
      max_tokens: 500,
    });
    const formatted = ((result.text as string | undefined) ?? '').trim();
    if (!formatted) throw new Error('AI returned empty formatted content');
    return { formatted_content: formatted };
  }

  async comparePrices(
    results: Record<string, unknown>[],
    queryText: string,
    _role = 'default',
    promptContent?: string,
    model?: string,
    priorityOrder?: string[],
  ): Promise<{ summary: string }> {
    const searchResultsText = (results ?? [])
      .slice(0, 30)
      .map((r, i) => {
        const price = r.price ? ` — ${String(r.price)}` : '';
        const source = r.source ? ` (${String(r.source)})` : '';
        return `${i + 1}. ${String(r.title ?? '')}${price}${source}`;
      })
      .join('\n');
    const priorityOrderStr = priorityOrder?.length ? priorityOrder.join(', ') : 'not specified';

    let textContent: string;
    if (promptContent) {
      textContent = promptContent
        .replace(/{{searchResults}}/g, searchResultsText)
        .replace(/{{search_results}}/g, searchResultsText)
        .replace(/{{queryText}}/g, queryText)
        .replace(/{{query_text}}/g, queryText)
        .replace(/{{priorityOrder}}/g, priorityOrderStr);
    } else {
      textContent = DEFAULT_COMPARISON_PROMPT
        .replace('{{searchResults}}', searchResultsText)
        .replace('{{queryText}}', queryText)
        .replace('{{priorityOrder}}', priorityOrderStr);
    }

    const result = await this.aiService.complete({
      model_tier: resolveModelTier(model),
      system_prompt: 'You are a shopping comparison assistant.',
      user_prompt: textContent,
      max_tokens: 500,
    });
    return { summary: ((result.text as string | undefined) ?? '').trim() };
  }

  async extractLocation(
    userText: string,
    queryText: string,
    _role = 'default',
    promptContent?: string,
    model?: string,
    priorityOrder?: string[],
  ): Promise<{ region: string | null; augmented_query: string | null }> {
    const priorityOrderStr = priorityOrder?.length ? priorityOrder.join(', ') : 'not specified';

    let textContent: string;
    if (promptContent) {
      textContent = promptContent
        .replace(/{{userInput}}/g, userText)
        .replace(/{{user_input}}/g, userText)
        .replace(/{{queryText}}/g, queryText)
        .replace(/{{query_text}}/g, queryText)
        .replace(/{{previousParams}}/g, '')
        .replace(/{{priorityOrder}}/g, priorityOrderStr);
    } else {
      textContent = DEFAULT_LOCATION_PROMPT
        .replace('{{userInput}}', userText)
        .replace('{{queryText}}', queryText)
        .replace('{{priorityOrder}}', priorityOrderStr);
    }

    const result = await this.aiService.complete({
      model_tier: resolveModelTier(model),
      system_prompt: 'Extract delivery region from user shopping request. Return short phrase or empty string.',
      user_prompt: textContent,
      max_tokens: 60,
    });
    const reply = ((result.text as string | undefined) ?? '').trim();
    const region = reply && reply.length < 150 ? reply : null;
    return { region, augmented_query: region };
  }
}
