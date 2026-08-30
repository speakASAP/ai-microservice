export interface AdminModelInfo {
  id: string;
  label: string;
  tier: 'free' | 'cheap' | 'smart' | 'premium';
  provider: string;
  price: string;
  bestFor: string;
  caution: string;
}

export const ADMIN_MODEL_CATALOG: AdminModelInfo[] = [
  {
    id: '',
    label: 'Default route for selected tier',
    tier: 'free',
    provider: 'LiteLLM router',
    price: 'Uses the tier default',
    bestFor: 'Keep this when you want the router to choose the configured model.',
    caution: 'Actual provider depends on litellm_config.yaml.',
  },
  {
    id: 'ollama/qwen2.5-coder:0.5b',
    label: 'Qwen2.5 Coder 0.5B local',
    tier: 'free',
    provider: 'Ollama',
    price: 'No API token cost; local compute only',
    bestFor: 'Tiny classification, extraction, routing, and low-risk formatting.',
    caution: 'Weak for complex reasoning and nuanced copy.',
  },
  {
    id: 'openrouter/google/gemma-3-27b-it:free',
    label: 'Gemma 3 27B IT free route',
    tier: 'cheap',
    provider: 'OpenRouter',
    price: 'Free route when available',
    bestFor: 'General text work when local output quality is too low.',
    caution: 'Free routes can be rate-limited or unavailable.',
  },
  {
    id: 'gemini/gemini-2.0-flash',
    label: 'Gemini 2.0 Flash',
    tier: 'smart',
    provider: 'Gemini',
    price: 'Low paid API tier',
    bestFor: 'Fast structured reasoning, summaries, classification, and prompt-heavy workflows.',
    caution: 'Use for tasks where quality matters but premium coding autonomy is unnecessary.',
  },
  {
    id: 'openrouter/anthropic/claude-sonnet-4.6',
    label: 'Claude Sonnet 4.6',
    tier: 'premium',
    provider: 'Anthropic',
    price: 'Premium paid API tier; human approval required here',
    bestFor: 'Repository work, coding agents, long-context planning, and high-stakes implementation.',
    caution: 'Do not route casual formatting or extraction tasks here.',
  },
];
