# Intent: AI Microservice Admin

The user wants a usable admin frontend for the AI microservice where existing and future AI agents can be managed from a browser.

Required outcome:
- Public browser entry at `/admin` on the AI microservice host.
- CRUD for AI agent definitions.
- Editable prompts, model tier, provider model override, token limits, temperature, output schema, metadata, tags, route path, service scope, and status.
- Preserve the existing AI runtime endpoints and deployment flow.
- Avoid dark-heavy UI; keep the admin surface light, balanced, dense, and operational.

Current system facts:
- The Alfares workspace contains 37 Git repositories, with AI-related systems including `ai-microservice`, `prompts-microservice`, `docs-rag-microservice`, `runlayer`, `agentic-email-processing-system`, `crypto-ai-agent`, and `shop-assistant`.
- `ai-microservice` is exposed at `https://ai.alfares.cz/` through Kubernetes ingress.
- The current backend has inference/workflow endpoints but no persisted generic AI-agent registry.
- Existing service auth is JWT bearer auth. The admin API should reuse that guard until a dedicated human-admin auth model exists.
