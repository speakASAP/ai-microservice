/**
 * ServiceAuthGuard authorization tests.
 *
 * Only Auth-minted RS256 (via verifyAuthToken) is accepted. Legacy ai-issued
 * RS256/HS256 tokens are rejected with zero fallback. Local JwtUtil mint is
 * deleted — fixtures are unsigned/self-labelled JWT shells that Auth JWKS
 * verification must refuse.
 */

import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ServiceAuthGuard } from './service-auth.guard';
import { IS_PUBLIC_KEY } from './public.decorator';
import { ROLES_KEY } from '../auth/roles.decorator';
import { AI_INVOKE_ROLES, AI_OPERATOR_ROLES } from '../auth/roles.constants';

function b64url(value: object): string {
  return Buffer.from(JSON.stringify(value)).toString('base64url');
}

/** Unsigned JWT shell labelled RS256 — Auth JWKS verify must reject. */
function legacyAiRs256Shell(serviceId = 'runlayer'): string {
  return `${b64url({ alg: 'RS256', typ: 'JWT' })}.${b64url({
    serviceId,
    iss: 'ai-microservice',
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 600,
  })}.not-a-real-signature`;
}

/** Unsigned JWT shell labelled HS256 — must never be accepted. */
function hs256Shell(serviceId = 'runlayer'): string {
  return `${b64url({ alg: 'HS256', typ: 'JWT' })}.${b64url({
    serviceId,
    iss: 'ai-microservice',
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 600,
  })}.fakesig`;
}

function contextFor(authorization?: string): ExecutionContext {
  return {
    switchToHttp: () => ({
      getRequest: () => ({ headers: authorization ? { authorization } : {} }),
    }),
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

describe('ServiceAuthGuard', () => {
  const original = { ...process.env };

  afterEach(() => {
    process.env = { ...original };
  });

  it('denies an undecorated route rather than falling back to a default role set', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(undefined));
    await expect(guard.canActivate(contextFor('Bearer x.y.z'))).rejects.toThrow(
      'Route has no authorization policy',
    );
  });

  it('allows a @Public route with no credential', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(undefined, true));
    await expect(guard.canActivate(contextFor())).resolves.toBe(true);
  });

  it('rejects a missing Authorization header on a protected route', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor())).rejects.toThrow('Missing service token');
  });

  it('rejects a legacy ai-issued RS256 token', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${legacyAiRs256Shell()}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });

  it('rejects a legacy ai-issued token on an operator route', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_OPERATOR_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${legacyAiRs256Shell()}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });

  it('rejects HS256 tokens', async () => {
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${hs256Shell()}`))).rejects.toThrow(
      UnauthorizedException,
    );
  });

  it('lets an Auth-minted invoke principal through after JWKS verify', async () => {
    const verifier = require('../auth/jwt-verifier');
    const spy = jest.spyOn(verifier, 'verifyAuthToken').mockResolvedValue({
      sub: 'svc-runlayer--ai-microservice',
      roles: ['internal:ai-microservice:invoke'],
      serviceName: 'runlayer',
    });
    const header = Buffer.from(JSON.stringify({ alg: 'RS256', kid: 'test' })).toString('base64url');
    const payload = Buffer.from(JSON.stringify({ sub: 'x' })).toString('base64url');
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(
      guard.canActivate(contextFor(`Bearer ${header}.${payload}.sig`)),
    ).resolves.toBe(true);
    spy.mockRestore();
  });

  it('denies an Auth-minted invoke principal on an operator route', async () => {
    const verifier = require('../auth/jwt-verifier');
    const spy = jest.spyOn(verifier, 'verifyAuthToken').mockResolvedValue({
      sub: 'svc-runlayer--ai-microservice',
      roles: ['internal:ai-microservice:invoke'],
    });
    const header = Buffer.from(JSON.stringify({ alg: 'RS256', kid: 'test' })).toString('base64url');
    const payload = Buffer.from(JSON.stringify({ sub: 'x' })).toString('base64url');
    const guard = new ServiceAuthGuard(reflectorFor(AI_OPERATOR_ROLES));
    await expect(
      guard.canActivate(contextFor(`Bearer ${header}.${payload}.sig`)),
    ).rejects.toThrow('Insufficient role');
    spy.mockRestore();
  });
});
