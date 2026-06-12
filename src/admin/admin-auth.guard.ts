import { CanActivate, ExecutionContext, Injectable } from '@nestjs/common';
import { AdminAuthService } from './admin-auth.service';

@Injectable()
export class AdminAuthGuard implements CanActivate {
  constructor(private readonly adminAuth: AdminAuthService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    request.adminUser = await this.adminAuth.requireAdminFromRequest(request);
    return true;
  }
}
