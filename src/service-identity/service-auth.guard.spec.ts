/**
 * ServiceAuthGuard authorization tests.
 *
 * Auth-minted RS256 (kid present) is enforced with roles. Legacy ai-issued
 * tokens (no kid) are accepted only while ALLOW_LEGACY_AI_ISSUED is open, and
 * only for invoke-tier routes — never operator.
 */

import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { generateKeyPairSync } from 'crypto';
import { JwtUtil } from './jwt.util';
import { ServiceAuthGuard } from './service-auth.guard';
import { IS_PUBLIC_KEY } from './public.decorator';
import { ROLES_KEY } from '../auth/roles.decorator';
import { AI_INVOKE_ROLES, AI_OPERATOR_ROLES } from '../auth/roles.constants';

const HS_SECRET = 'shared-secret-for-tests';

const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

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

  it('accepts a legacy ai-issued RS256 token on an invoke route while legacy is open', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.ALLOW_LEGACY_AI_ISSUED = 'true';
    process.env.ALLOW_HS256_FALLBACK = 'false';
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${token}`))).resolves.toBe(true);
  });

  it('accepts a legacy ai-issued token on an operator route during the dual window', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.ALLOW_LEGACY_AI_ISSUED = 'true';
    process.env.ALLOW_HS256_FALLBACK = 'false';
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const guard = new ServiceAuthGuard(reflectorFor(AI_OPERATOR_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${token}`))).resolves.toBe(true);
  });

  it('rejects legacy tokens once ALLOW_LEGACY_AI_ISSUED is closed', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.ALLOW_LEGACY_AI_ISSUED = 'false';
    const token = JwtUtil.signRS256('runlayer', privateKey);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${token}`))).rejects.toThrow(
      'AI-issued service tokens are no longer accepted',
    );
  });

  it('still accepts HS256 only while ALLOW_HS256_FALLBACK is open', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.JWT_SECRET = HS_SECRET;
    process.env.ALLOW_LEGACY_AI_ISSUED = 'true';
    process.env.ALLOW_HS256_FALLBACK = 'true';
    const token = JwtUtil.sign('runlayer', HS_SECRET);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${token}`))).resolves.toBe(true);
  });

  it('rejects HS256 once ALLOW_HS256_FALLBACK is closed', async () => {
    process.env.JWT_PUBLIC_KEY = publicKey;
    process.env.JWT_SECRET = HS_SECRET;
    process.env.ALLOW_LEGACY_AI_ISSUED = 'true';
    process.env.ALLOW_HS256_FALLBACK = 'false';
    const token = JwtUtil.sign('runlayer', HS_SECRET);
    const guard = new ServiceAuthGuard(reflectorFor(AI_INVOKE_ROLES));
    await expect(guard.canActivate(contextFor(`Bearer ${token}`))).rejects.toThrow(
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
