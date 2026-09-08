#!/usr/bin/env ts-node
/**
 * RETIRED. Follow auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md;
 * do not invent alternate S2S protocols. Mint only via
 * auth-microservice/scripts/provision-service-token.js.
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
