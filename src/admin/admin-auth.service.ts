import { ForbiddenException, Injectable, UnauthorizedException } from '@nestjs/common';
import { Request } from 'express';

export const ADMIN_SESSION_COOKIE = 'ai_admin_token';

export interface AdminUser {
  id?: string;
  email?: string;
  userType?: string;
  type?: string;
  roles?: string[];
}

@Injectable()
export class AdminAuthService {
  private readonly allowedRoles = new Set([
    'global:superadmin',
    'global:platform_admin',
    'global:admin',
    'app:ai-microservice:admin',
    'app:ai:admin',
  ]);

  getAuthLoginUrl(): string {
    const publicAuthUrl = process.env.AUTH_PUBLIC_URL || 'https://auth.alfares.cz';
    const returnUrl = process.env.AI_ADMIN_AUTH_RETURN_URL || 'https://ai.alfares.cz/admin/session';
    const url = new URL('/login', publicAuthUrl);
    url.searchParams.set('return_url', returnUrl);
    return url.toString();
  }

  getTokenFromRequest(req: Request): string | null {
    const authHeader = req.headers.authorization;
    if (typeof authHeader === 'string' && authHeader.startsWith('Bearer ')) {
      return authHeader.slice(7).trim() || null;
    }

    return this.parseCookies(req.headers.cookie || '')[ADMIN_SESSION_COOKIE] || null;
  }

  async requireAdminFromRequest(req: Request): Promise<AdminUser> {
    const token = this.getTokenFromRequest(req);
    if (!token) throw new UnauthorizedException('Admin authentication required');
    return this.requireAdminToken(token);
  }

  async requireAdminToken(token: string): Promise<AdminUser> {
    const user = await this.validateToken(token);
    if (!this.isAdmin(user)) {
      throw new ForbiddenException('Admin role required');
    }
    return user;
  }

  getCookieMaxAgeMs(token: string): number {
    const exp = this.decodeExp(token);
    if (!exp) return 12 * 60 * 60 * 1000;
    const remainingMs = exp * 1000 - Date.now();
    return Math.max(0, Math.min(remainingMs, 12 * 60 * 60 * 1000));
  }

  private async validateToken(token: string): Promise<AdminUser> {
    const authServiceUrl = process.env.AUTH_SERVICE_URL || 'http://auth-microservice.statex-apps.svc.cluster.local:3370';
    const response = await fetch(`${authServiceUrl.replace(/\/$/, '')}/auth/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });

    if (!response.ok) {
      throw new UnauthorizedException('Invalid admin token');
    }

    const payload = (await response.json()) as { valid?: boolean; user?: AdminUser };
    if (!payload.valid || !payload.user) {
      throw new UnauthorizedException('Invalid admin token');
    }

    return payload.user;
  }

  private isAdmin(user: AdminUser): boolean {
    if (user.userType === 'admin' || user.type === 'admin') return true;
    return (user.roles || []).some((role) => this.allowedRoles.has(role));
  }

  private parseCookies(cookieHeader: string): Record<string, string> {
    return cookieHeader
      .split(';')
      .map((part) => part.trim())
      .filter(Boolean)
      .reduce<Record<string, string>>((cookies, part) => {
        const separator = part.indexOf('=');
        if (separator === -1) return cookies;
        const key = decodeURIComponent(part.slice(0, separator).trim());
        const value = decodeURIComponent(part.slice(separator + 1).trim());
        cookies[key] = value;
        return cookies;
      }, {});
  }

  private decodeExp(token: string): number | null {
    try {
      const payload = JSON.parse(Buffer.from(token.split('.')[1] || '', 'base64url').toString('utf8')) as {
        exp?: number;
      };
      return typeof payload.exp === 'number' ? payload.exp : null;
    } catch {
      return null;
    }
  }
}
