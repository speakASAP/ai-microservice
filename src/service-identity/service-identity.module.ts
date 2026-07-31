import { Module } from '@nestjs/common';
import { APP_GUARD } from '@nestjs/core';
import { ServiceAuthGuard } from './service-auth.guard';

@Module({
  providers: [
    ServiceAuthGuard,
    {
      provide: APP_GUARD,
      useExisting: ServiceAuthGuard,
    },
  ],
  // Exported so other modules can apply the same guard instance explicitly
  // via `@UseGuards(ServiceAuthGuard)` instead of relying only on its
  // implicit global (APP_GUARD) registration.
  exports: [ServiceAuthGuard],
})
export class ServiceIdentityModule {}
