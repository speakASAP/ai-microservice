/**
 * Required roles for an endpoint (OR logic). Use with ServiceAuthGuard.
 *
 * A route with no @Roles and no @Public is denied — see ServiceAuthGuard.
 */

import { SetMetadata } from '@nestjs/common';

export const ROLES_KEY = 'roles';

export const Roles = (...roles: readonly string[]) => SetMetadata(ROLES_KEY, { roles });
