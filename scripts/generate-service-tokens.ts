#!/usr/bin/env ts-node
/**
 * RETIRED — local HS256 minting is prohibited.
 *
 * Service tokens for ai-microservice callers must be Auth-issued RS256
 * principals minted only via:
 *   auth-microservice/scripts/provision-service-token.js
 *
 * Identity: svc-<caller>--ai-microservice@internal.alfares.cz
 * Role:     internal:ai-microservice:invoke  (or :operator for Claude Code)
 * Delivery: Vault → ExternalSecret → Secret → secretKeyRef as AI_SERVICE_TOKEN
 *
 * See auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md
 */

console.error(
  JSON.stringify(
    {
      status: 'failed',
      error:
        'generate-service-tokens.ts is retired. Use auth-microservice/scripts/provision-service-token.js',
    },
    null,
    2,
  ),
);
process.exit(1);
