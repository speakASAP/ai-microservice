#!/usr/bin/env ts-node
/**
 * Generates signed JWT service tokens for all known consumers of ai-microservice.
 * Usage:
 *   JWT_SECRET=<your-secret> npx ts-node scripts/generate-service-tokens.ts
 *
 * Each token:
 *   - Identifies the calling service via { serviceId }
 *   - Is issued by "ai-microservice"
 *   - Expires in 1 year (rotate annually or on secret rotation)
 *
 * Add the printed token to each consumer's .env as AI_SERVICE_TOKEN=<token>
 * Consumers send: Authorization: Bearer <token>
 */

import { createHmac, timingSafeEqual } from 'crypto';

const CONSUMERS = [
  'business-orchestrator',
  'statex',
  'shop-assistant',
  'crypto-ai-agent',
  'agentic-email',
];

function base64url(input: Buffer): string {
  return input.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function sign(serviceId: string, secret: string): string {
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64url');
  const now = Math.floor(Date.now() / 1000);
  const payload = Buffer.from(
    JSON.stringify({ serviceId, iss: 'ai-microservice', iat: now, exp: now + 365 * 24 * 3600 }),
  ).toString('base64url');
  const sig = base64url(createHmac('sha256', secret).update(`${header}.${payload}`).digest());
  return `${header}.${payload}.${sig}`;
}

const secret = process.env.JWT_SECRET;
if (!secret) {
  console.error('ERROR: JWT_SECRET environment variable is required');
  process.exit(1);
}

console.log('\nService tokens for ai-microservice consumers');
console.log('Add AI_SERVICE_TOKEN=<token> to each consumer .env\n');
console.log('='.repeat(80));

for (const consumer of CONSUMERS) {
  const token = sign(consumer, secret);
  console.log(`\n# ${consumer}`);
  console.log(`AI_SERVICE_TOKEN=${token}`);
}

console.log('\n' + '='.repeat(80));
console.log('\nTokens expire in 1 year. Rotate by re-running with the same JWT_SECRET.');
