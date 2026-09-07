/**
 * Auth-issued RS256 verification via JWKS.
 *
 * This service holds only auth's public key and cannot sign. Modelled on
 * docs-rag-microservice/src/auth/jwt-verifier.ts. AI-issued / HS256 tokens are
 * refused here — ServiceAuthGuard has no legacy fallback.
 */

import { UnauthorizedException } from '@nestjs/common';
import { createPublicKey, createVerify, JsonWebKey, KeyObject } from 'crypto';

const JWKS_TTL_MS = 5 * 60 * 1000;

type Jwk = JsonWebKey & { kid?: string; kty?: string };

let cachedKeys = new Map<string, KeyObject>();
let cachedAt = 0;
let inFlight: Promise<void> | null = null;

function jwksUrl(): string {
  const base = process.env.AUTH_SERVICE_URL || 'http://auth-microservice:3370';
  return `${base.replace(/\/$/, '')}/.well-known/jwks.json`;
}

function base64urlDecode(input: string): Buffer {
  return Buffer.from(input.replace(/-/g, '+').replace(/_/g, '/'), 'base64');
}

async function refreshJwks(): Promise<void> {
  if (inFlight) return inFlight;

  inFlight = (async () => {
    const url = jwksUrl();
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) throw new Error(`JWKS fetch failed: HTTP ${res.status} from ${url}`);

      const body = (await res.json()) as { keys?: Jwk[] };
      const next = new Map<string, KeyObject>();
      for (const k of body.keys ?? []) {
        if (k.kty !== 'RSA' || !k.kid) continue;
        next.set(k.kid, createPublicKey({ key: k, format: 'jwk' }));
      }
      cachedKeys = next;
      cachedAt = Date.now();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[jwt-verifier] JWKS refresh failed from ${url}: ${message}`);
      throw err;
    } finally {
      inFlight = null;
    }
  })();

  return inFlight;
}

async function publicKeyFor(kid: string): Promise<KeyObject | null> {
  const stale = Date.now() - cachedAt > JWKS_TTL_MS;
  if (cachedKeys.size === 0 || stale) {
    await refreshJwks().catch(() => undefined);
  }
  if (!cachedKeys.has(kid) && Date.now() - cachedAt > 5000) {
    await refreshJwks().catch(() => undefined);
  }
  return cachedKeys.get(kid) ?? null;
}

export interface VerifiedPayload {
  sub: string;
  email?: string;
  roles?: string[];
  serviceName?: string;
  [key: string]: unknown;
}

/**
 * Verify an auth-issued RS256 token. Throws UnauthorizedException on any
 * failure. HS256 and other algorithms are refused immediately.
 */
export async function verifyAuthToken(token: string): Promise<VerifiedPayload> {
  const parts = token.split('.');
  if (parts.length !== 3) throw new UnauthorizedException('Malformed token');
  const [header, payload, signature] = parts;

  let head: { alg?: string; kid?: string };
  try {
    head = JSON.parse(base64urlDecode(header).toString()) as { alg?: string; kid?: string };
  } catch {
    throw new UnauthorizedException('Malformed token header');
  }

  if (head.alg !== 'RS256') {
    throw new UnauthorizedException(`Unsupported token algorithm ${head.alg ?? 'none'}; RS256 required`);
  }
  if (!head.kid) throw new UnauthorizedException('RS256 token has no kid');

  const key = await publicKeyFor(head.kid);
  if (!key) throw new UnauthorizedException(`No JWKS key for kid ${head.kid}`);

  const ok = createVerify('RSA-SHA256')
    .update(`${header}.${payload}`)
    .verify(key, base64urlDecode(signature));
  if (!ok) throw new UnauthorizedException('Invalid signature');

  let claims: VerifiedPayload;
  try {
    claims = JSON.parse(base64urlDecode(payload).toString()) as VerifiedPayload;
  } catch {
    throw new UnauthorizedException('Malformed token payload');
  }

  const now = Math.floor(Date.now() / 1000);
  if (typeof claims.exp === 'number' && claims.exp < now) {
    throw new UnauthorizedException('Token expired');
  }
  if (typeof claims.nbf === 'number' && claims.nbf > now) {
    throw new UnauthorizedException('Token not yet valid');
  }

  return claims;
}
