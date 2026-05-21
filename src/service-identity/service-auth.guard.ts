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
    const secret = process.env.JWT_SECRET;
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
}
