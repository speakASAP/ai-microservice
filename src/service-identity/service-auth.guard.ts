import {
  CanActivate,
  ExecutionContext,
  Injectable,
  Logger,
  UnauthorizedException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { IS_PUBLIC_KEY } from './public.decorator';
import { ROLES_KEY } from '../auth/roles.decorator';
import { AI_LEGACY_EFFECTIVE_ROLES } from '../auth/roles.constants';
import { verifyAuthToken } from '../auth/jwt-verifier';
import { JwtUtil } from './jwt.util';

interface ServiceRequest {
  headers: Record<string, string | undefined>;
  serviceId?: string;
}

function tokenHeader(token: string): { alg?: string; kid?: string } {
  const [header] = token.split('.');
  if (!header) return {};
  try {
    const padded = header.replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(Buffer.from(padded, 'base64').toString()) as { alg?: string; kid?: string };
  } catch {
    return {};
  }
}

@Injectable()
export class ServiceAuthGuard implements CanActivate {
  private readonly logger = new Logger(ServiceAuthGuard.name);

  constructor(private readonly reflector: Reflector) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);
    if (isPublic) return true;

    const request = context.switchToHttp().getRequest<ServiceRequest>();
    const handler = `${context.getClass().name}.${context.getHandler().name}`;

    const meta = this.reflector.getAllAndOverride<{ roles?: readonly string[] }>(ROLES_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);
    const requiredRoles = meta?.roles ?? [];
    if (requiredRoles.length === 0) {
      this.logger.error(
        `Denied ${handler}: route carries neither @Roles nor @Public. ` +
          'Every route must declare its role set in src/auth/roles.constants.ts.',
      );
      throw new UnauthorizedException('Route has no authorization policy');
    }

    const authHeader = request.headers['authorization'];
    if (!authHeader?.startsWith('Bearer ')) {
      throw new UnauthorizedException('Missing service token');
    }
    const token = authHeader.slice(7);
    const head = tokenHeader(token);

    // Auth-issued RS256 (kid present) — fail closed; never fall through to legacy.
    if (head.kid) {
      const payload = await verifyAuthToken(token);
      const roles = Array.isArray(payload.roles) ? payload.roles : [];
      if (!roles.some((r) => requiredRoles.includes(r))) {
        this.logger.error(
          `Denied ${payload.sub ?? 'unknown'} on ${handler}: has [${roles.join(', ')}], needs one of [${requiredRoles.join(', ')}]`,
        );
        throw new UnauthorizedException('Insufficient role');
      }
      request.serviceId = payload.serviceName ?? payload.sub;
      return true;
    }

    // Legacy ai-issued tokens (self-signed RS256 / shared-secret HS256). Migration
    // window only — close with ALLOW_LEGACY_AI_ISSUED=false after every caller
    // holds an Auth-minted principal. Acceptance proof is an authenticated Auth
    // call, not Secret sync.
    if (!this.legacyAiIssuedEnabled()) {
      throw new UnauthorizedException(
        'AI-issued service tokens are no longer accepted; use an Auth-minted RS256 principal',
      );
    }

    const publicKey = process.env.JWT_PUBLIC_KEY;
    const secret = process.env.JWT_SECRET;

    if (head.alg === 'RS256' && publicKey) {
      try {
        const payload = JwtUtil.verifyRS256(token, publicKey);
        return this.acceptLegacy(request, handler, requiredRoles, payload.serviceId, 'RS256');
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Invalid token';
        throw new UnauthorizedException(message);
      }
    }

    if (this.hs256FallbackEnabled() && secret) {
      try {
        const payload = JwtUtil.verify(token, secret);
        return this.acceptLegacy(request, handler, requiredRoles, payload.serviceId, 'HS256');
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Invalid token';
        throw new UnauthorizedException(message);
      }
    }

    throw new UnauthorizedException('Unsupported service token; Auth-minted RS256 required');
  }

  private acceptLegacy(
    request: ServiceRequest,
    handler: string,
    requiredRoles: readonly string[],
    serviceId: string,
    alg: string,
  ): boolean {
    this.logger.warn(
      `Legacy ai-issued ${alg} service token accepted for serviceId=${serviceId} on ${handler}; ` +
        'migrate to Auth-minted svc-<caller>--ai-microservice@internal.alfares.cz',
    );
    const roles = [...AI_LEGACY_EFFECTIVE_ROLES];
    if (!roles.some((r) => requiredRoles.includes(r))) {
      this.logger.error(
        `Denied legacy ${serviceId} on ${handler}: legacy path only grants [${roles.join(', ')}]`,
      );
      throw new UnauthorizedException('Insufficient role');
    }
    request.serviceId = serviceId;
    return true;
  }

  /** Defaults to true so an unconfigured deploy cannot lock every caller out mid-migration. */
  private legacyAiIssuedEnabled(): boolean {
    return process.env.ALLOW_LEGACY_AI_ISSUED !== 'false';
  }

  /** HS256 shared-secret path. Live pod already has this closed (=false). */
  private hs256FallbackEnabled(): boolean {
    return process.env.ALLOW_HS256_FALLBACK !== 'false';
  }
}
