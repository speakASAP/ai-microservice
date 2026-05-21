import { Controller, Get } from '@nestjs/common';
import { Public } from './service-identity/public.decorator';

@Controller('health')
export class HealthController {
  @Get()
  @Public()
  check(): { status: string } {
    return { status: 'ok' };
  }
}
