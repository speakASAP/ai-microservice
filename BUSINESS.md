# Business: ai-microservice
>
> ⚠️ IMMUTABLE BY AI.

## Goal

Centralized AI inference gateway for all Statex services. Routes LLM calls by model tier, provides NLP, ASR, Document AI, and prototype generation.

## Constraints

- AI agents must never call external LLM APIs directly — route through this service
- Model selection by tier: free (Ollama) → cheap (OpenRouter) → smart (Gemini/Claude)
- Premium tier requires explicit human approval per invocation
- API costs tracked per service/business_id

## Consumers

business-orchestrator, statex, shop-assistant, crypto-ai-agent, agentic-email.

## SLA

- Port: 3380 (<http://ai-microservice:3380>)
- Production: <https://ai.statex.cz>
