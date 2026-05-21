import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ServiceAuthGuard } from '../../src/service-identity/service-auth.guard';
import { JwtUtil } from '../../src/service-identity/jwt.util';

const SECRET = 'test-secret-at-least-32-chars-long!!';

function makeContext(authHeader: string | undefined, isPublic = false): ExecutionContext {
  const request = { headers: { authorization: authHeader }, path: '/task/draft', method: 'POST' } as never;
  return {
    switchToHttp: () => ({ getRequest: () => request }),
    getHandler: () => ({}),
    getClass: () => ({}),
    _isPublic: isPublic,
  } as unknown as ExecutionContext;
}

describe('ServiceAuthGuard', () => {
  let guard: ServiceAuthGuard;
  let reflector: Reflector;

  beforeEach(() => {
    process.env.JWT_SECRET = SECRET;
    reflector = { getAllAndOverride: jest.fn() } as unknown as Reflector;
    guard = new ServiceAuthGuard(reflector);
  });

  afterEach(() => {
    delete process.env.JWT_SECRET;
  });

  it('allows @Public() routes without token', () => {
    (reflector.getAllAndOverride as jest.Mock).mockReturnValue(true);
    expect(guard.canActivate(makeContext(undefined, true))).toBe(true);
  });

  it('rejects missing Authorization header', () => {
    (reflector.getAllAndOverride as jest.Mock).mockReturnValue(false);
    expect(() => guard.canActivate(makeContext(undefined))).toThrow(UnauthorizedException);
  });

  it('rejects non-Bearer scheme', () => {
    (reflector.getAllAndOverride as jest.Mock).mockReturnValue(false);
    expect(() => guard.canActivate(makeContext('Basic abc123'))).toThrow(UnauthorizedException);
  });

  it('rejects invalid token', () => {
    (reflector.getAllAndOverride as jest.Mock).mockReturnValue(false);
    expect(() => guard.canActivate(makeContext('Bearer garbage.token.here'))).toThrow(
      UnauthorizedException,
    );
  });

  it('allows valid token and attaches serviceId to request', () => {
    (reflector.getAllAndOverride as jest.Mock).mockReturnValue(false);
    const token = JwtUtil.sign('shop-assistant', SECRET);
    const ctx = makeContext(`Bearer ${token}`);
    expect(guard.canActivate(ctx)).toBe(true);
    const req = ctx.switchToHttp().getRequest<{ serviceId: string }>();
    expect(req.serviceId).toBe('shop-assistant');
  });

  it('rejects expired token', () => {
    (reflector.getAllAndOverride as jest.Mock).mockReturnValue(false);
    const token = JwtUtil.sign('shop-assistant', SECRET, -1);
    expect(() => guard.canActivate(makeContext(`Bearer ${token}`))).toThrow(UnauthorizedException);
  });
});
