/**
 * Integration-style ServiceAuthGuard tests (Auth RS256 only).
 * Prefer src/service-identity/service-auth.guard.spec.ts for role matrix coverage.
 */

import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { generateKeyPairSync } from 'crypto';
import { ServiceAuthGuard } from '../../src/service-identity/service-auth.guard';
import { JwtUtil } from '../../src/service-identity/jwt.util';
import { IS_PUBLIC_KEY } from '../../src/service-identity/public.decorator';
import { ROLES_KEY } from '../../src/auth/roles.decorator';
import { AI_INVOKE_ROLES } from '../../src/auth/roles.constants';

const SECRET = 'test-secret-at-least-32-chars-long!!';
const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

function makeContext(authHeader: string | undefined): ExecutionContext {
  const request = { headers: { authorization: authHeader }, path: '/task/draft', method: 'POST' } as never;
  return {
    switchToHttp: () => ({ getRequest: () => request }),
    getHandler: () => function handler() {},
    getClass: () => class TestController {},
  } as unknown as ExecutionContext;
}

function reflectorFor(roles?: readonly string[], isPublic = false): Reflector {
  return {
    getAllAndOverride: (key: string) => {
      if (key === IS_PUBLIC_KEY) return isPublic;
      if (key === ROLES_KEY) return roles ? { roles } : undefined;
      return undefined;
    },
  } as unknown as Reflector;
}

describe('ServiceAuthGuard (test/)', () => {
  const original = { ...process.env };

  afterEach(() => {
    process.env = { ...original };
  });

  it('allows @Public() routes without token', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(undefined, true));
    await expect(guard.canActivate(makeContext(undefined))).resolves.toBe(true);
  });

  it('rejects missing Authorization header', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext(undefined))).rejects.toThrow(UnauthorizedException);
  });

  it('rejects non-Bearer scheme', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext('Basic abc123'))).rejects.toThrow(UnauthorizedException);
  });

  it('rejects legacy ai-issued RS256', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    const token = JwtUtil.signRS256('shop-assistant', privateKey);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext(`Bearer ${token}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });

  it('rejects expired legacy token', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    const token = JwtUtil.signRS256('shop-assistant', privateKey, -1);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext(`Bearer ${token}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });

  it('rejects HS256 tokens', async () => {
    process.env.JWT_SECRET = SECRET;
    const token = JwtUtil.sign('shop-assistant', SECRET);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext(`Bearer ${token}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });
});
