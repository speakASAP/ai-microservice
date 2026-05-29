import { Controller, Get } from '@nestjs/common';
import { Public } from './service-identity/public.decorator';
import { parseOrThrow } from './contracts/parse-or-throw';
import { HealthResponseSchema } from './contracts/http-responses.contract';

@Controller('health')
export class HealthController {
  @Get()
  @Public()
  check() {
    return parseOrThrow(
      HealthResponseSchema,
      { status: 'ok', service: 'ai-microservice' },
      'health.check',
    );
  }
}
