import { Controller, Get, Query, UseGuards } from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { AdminAuthGuard } from './admin-auth.guard';
import { ADMIN_MODEL_CATALOG } from './admin-models';

@Public()
@UseGuards(AdminAuthGuard)
@Controller('admin/api')
export class AdminMetaController {
  @Get('models')
  models() {
    return { items: ADMIN_MODEL_CATALOG };
  }

  @Get('logs')
  async logs(
    @Query('service') service?: string,
    @Query('level') level?: string,
    @Query('range') range = '24h',
    @Query('limit') limit = '100',
    @Query('q') q?: string,
  ) {
    const loggingUrl = process.env.LOGGING_SERVICE_URL || 'http://logging-microservice.statex-apps.svc.cluster.local:3367';
    const params = new URLSearchParams();
    if (service) params.set('service', service);
    if (level) params.set('level', level);
    params.set('limit', String(Math.min(Math.max(Number(limit) || 100, 1), 500)));

    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - rangeToMs(range));
    params.set('startDate', startDate.toISOString());
    params.set('endDate', endDate.toISOString());

    const response = await fetch(`${loggingUrl.replace(/\/$/, '')}/api/logs/query?${params.toString()}`);
    if (!response.ok) {
      return { items: [], count: 0, error: `Logging service returned ${response.status}` };
    }

    const payload = (await response.json()) as { data?: unknown[]; count?: number };
    const needle = String(q || '').trim().toLowerCase();
    const items = (payload.data || [])
      .filter((item) => !needle || JSON.stringify(item).toLowerCase().includes(needle))
      .slice(0, Math.min(Math.max(Number(limit) || 100, 1), 500));

    return { items, count: items.length };
  }
}

function rangeToMs(range: string): number {
  if (range === '1h') return 60 * 60 * 1000;
  if (range === '6h') return 6 * 60 * 60 * 1000;
  if (range === '7d') return 7 * 24 * 60 * 60 * 1000;
  return 24 * 60 * 60 * 1000;
}
