import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { generateKeyPairSync } from 'crypto';
import { JwtUtil } from './jwt.util';
import { ServiceAuthGuard } from './service-auth.guard';

const HS_SECRET = 'shared-secret-for-tests';

const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

function contextWith(token?: string): ExecutionContext {
  const request: { headers: Record<string, string | undefined>; serviceId?: string } = {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  };
  return {
    switchToHttp: () => ({ getRequest: () => request }),
    getHandler: () => undefined,
    getClass: () => undefined,
  } as unknown as ExecutionContext;
}

function guard(): ServiceAuthGuard {
  const reflector = { getAllAndOverride: () => false } as unknown as Reflector;
  return new ServiceAuthGuard(reflector);
}

describe('ServiceAuthGuard', () => {
  const original = { ...process.env };

  afterEach(() => {
    process.env = { ...original };
  });

  it('accepts an RS256 token when a public key is configured', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    const token = JwtUtil.signRS256('runlayer', privateKey);
    expect(guard().canActivate(contextWith(token))).toBe(true);
  });

  it('still accepts a legacy HS256 token while the fallback is open', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.JWT_SECRET = HS_SECRET;
    process.env.ALLOW_HS256_FALLBACK = 'true';
    const token = JwtUtil.sign('runlayer', HS_SECRET);
    expect(guard().canActivate(contextWith(token))).toBe(true);
  });

  it('rejects a legacy HS256 token once the fallback is closed', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.JWT_SECRET = HS_SECRET;
    process.env.ALLOW_HS256_FALLBACK = 'false';
    const token = JwtUtil.sign('runlayer', HS_SECRET);
    expect(() => guard().canActivate(contextWith(token))).toThrow(UnauthorizedException);
  });

  it('still accepts RS256 once the fallback is closed', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.JWT_SECRET = HS_SECRET;
    process.env.ALLOW_HS256_FALLBACK = 'false';
    const token = JwtUtil.signRS256('runlayer', privateKey);
    expect(guard().canActivate(contextWith(token))).toBe(true);
  });

  it('rejects a token signed by a foreign private key', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.ALLOW_HS256_FALLBACK = 'false';
    const foreign = generateKeyPairSync('rsa', {
      modulusLength: 2048,
      publicKeyEncoding: { type: 'spki', format: 'pem' },
      privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
    });
    const token = JwtUtil.signRS256('runlayer', foreign.privateKey);
    expect(() => guard().canActivate(contextWith(token))).toThrow(UnauthorizedException);
  });

  it('works HS256-only when no public key is configured yet (pre-migration)', () => {
    delete process.env.JWT_PUBLIC_KEY;
    process.env.JWT_SECRET = HS_SECRET;
    const token = JwtUtil.sign('runlayer', HS_SECRET);
    expect(guard().canActivate(contextWith(token))).toBe(true);
  });

  it('rejects a request with no bearer token', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    expect(() => guard().canActivate(contextWith())).toThrow('Missing service token');
  });

  it('propagates the serviceId onto the request', () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    const token = JwtUtil.signRS256('domain-research', privateKey);
    const ctx = contextWith(token);
    guard().canActivate(ctx);
    expect((ctx.switchToHttp().getRequest() as { serviceId?: string }).serviceId).toBe(
      'domain-research',
    );
  });
});
