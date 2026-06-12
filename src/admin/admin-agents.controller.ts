import { Body, Controller, Delete, Get, HttpCode, Param, Post, Put, Query, UseGuards } from '@nestjs/common';
import { AdminAgentsService, AiAgentPayload } from './admin-agents.service';
import { AiAgentModelTier, AiAgentStatus } from '../database/entities/ai-agent.entity';
import { Public } from '../service-identity/public.decorator';
import { AdminAuthGuard } from './admin-auth.guard';

@Public()
@UseGuards(AdminAuthGuard)
@Controller('admin/api/agents')
export class AdminAgentsController {
  constructor(private readonly adminAgents: AdminAgentsService) {}

  @Get()
  async list(
    @Query('q') q?: string,
    @Query('status') status?: AiAgentStatus,
    @Query('modelTier') modelTier?: AiAgentModelTier,
    @Query('serviceScope') serviceScope?: string,
  ) {
    const items = await this.adminAgents.list({
      q: q?.trim() || undefined,
      status: status || undefined,
      modelTier: modelTier || undefined,
      serviceScope: serviceScope?.trim() || undefined,
    });
    return { items };
  }

  @Get(':id')
  get(@Param('id') id: string) {
    return this.adminAgents.get(id);
  }

  @Post()
  create(@Body() body: AiAgentPayload) {
    return this.adminAgents.create(body);
  }

  @Put(':id')
  update(@Param('id') id: string, @Body() body: AiAgentPayload) {
    return this.adminAgents.update(id, body);
  }

  @Delete(':id')
  @HttpCode(204)
  async remove(@Param('id') id: string) {
    await this.adminAgents.remove(id);
  }
}
