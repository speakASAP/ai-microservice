import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { IS_PUBLIC_KEY } from './public.decorator';
import { JwtUtil } from './jwt.util';

interface ServiceRequest {
  headers: Record<string, string | undefined>;
  serviceId?: string;
}

@Injectable()
export class ServiceAuthGuard implements CanActivate {
  constructor(private readonly reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);
    if (isPublic) return true;

    const request = context.switchToHttp().getRequest<ServiceRequest>();
    const authHeader = request.headers['authorization'];
    if (!authHeader?.startsWith('Bearer ')) {
      throw new UnauthorizedException('Missing service token');
    }

    const token = authHeader.slice(7);
    const publicKey = process.env.JWT_PUBLIC_KEY;
    const secret = process.env.JWT_SECRET;

    // RS256 is the target scheme: only ai-microservice holds the private key,
    // so a compromise of any caller cannot be used to mint tokens.
    if (publicKey) {
      try {
        const payload = JwtUtil.verifyRS256(token, publicKey);
        request.serviceId = payload.serviceId;
        return true;
      } catch (err: unknown) {
        // Fall through to HS256 only while the migration window is open.
        if (!this.hs256FallbackEnabled() || !secret) {
          const message = err instanceof Error ? err.message : 'Invalid token';
          throw new UnauthorizedException(message);
        }
      }
    }

    // Legacy HS256 path. Retained so callers still holding shared-secret tokens
    // keep working until every one of them has been re-minted as RS256; remove
    // by setting ALLOW_HS256_FALLBACK=false once that is done.
    if (!this.hs256FallbackEnabled()) {
      throw new UnauthorizedException('HS256 service tokens are no longer accepted');
    }
    if (!secret) throw new UnauthorizedException('JWT_SECRET not configured');

    try {
      const payload = JwtUtil.verify(token, secret);
      request.serviceId = payload.serviceId;
      return true;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Invalid token';
      throw new UnauthorizedException(message);
    }
  }

  /** Defaults to true so an unconfigured deploy cannot lock every caller out. */
  private hs256FallbackEnabled(): boolean {
    return process.env.ALLOW_HS256_FALLBACK !== 'false';
  }
}
