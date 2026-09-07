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
import { verifyAuthToken } from '../auth/jwt-verifier';

interface ServiceRequest {
  headers: Record<string, string | undefined>;
  serviceId?: string;
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

    // Auth-minted RS256 only — verifyAuthToken fails closed (alg, kid, JWKS, sig, exp).
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
}
