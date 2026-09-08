/**
 * Integration-style ServiceAuthGuard tests (Auth RS256 only).
 * Prefer src/service-identity/service-auth.guard.spec.ts for role matrix coverage.
 *
 * Local JwtUtil mint is deleted — rejection fixtures are unsigned JWT shells.
 */

import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ServiceAuthGuard } from '../../src/service-identity/service-auth.guard';
import { IS_PUBLIC_KEY } from '../../src/service-identity/public.decorator';
import { ROLES_KEY } from '../../src/auth/roles.decorator';
import { AI_INVOKE_ROLES } from '../../src/auth/roles.constants';

function b64url(value: object): string {
  return Buffer.from(JSON.stringify(value)).toString('base64url');
}

function legacyAiRs256Shell(expired = false): string {
  const now = Math.floor(Date.now() / 1000);
  return `${b64url({ alg: 'RS256', typ: 'JWT' })}.${b64url({
    serviceId: 'shop-assistant',
    iss: 'ai-microservice',
    iat: now,
    exp: expired ? now - 10 : now + 600,
  })}.not-a-real-signature`;
}

function hs256Shell(): string {
  const now = Math.floor(Date.now() / 1000);
  return `${b64url({ alg: 'HS256', typ: 'JWT' })}.${b64url({
    serviceId: 'shop-assistant',
    iss: 'ai-microservice',
    iat: now,
    exp: now + 600,
  })}.fakesig`;
}

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
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext(`Bearer ${legacyAiRs256Shell()}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });

  it('rejects expired legacy token', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(
      guard.canActivate(makeContext(`Bearer ${legacyAiRs256Shell(true)}`)),
    ).rejects.toThrow(UnauthorizedException);
  });

  it('rejects HS256 tokens', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(makeContext(`Bearer ${hs256Shell()}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });
});
