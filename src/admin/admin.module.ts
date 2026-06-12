import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AiAgent } from '../database/entities/ai-agent.entity';
import { AdminAgentsController } from './admin-agents.controller';
import { AdminAgentsService } from './admin-agents.service';
import { AdminAuthGuard } from './admin-auth.guard';
import { AdminAuthService } from './admin-auth.service';
import { AdminFrontendController } from './admin-frontend.controller';

@Module({
  imports: [TypeOrmModule.forFeature([AiAgent])],
  controllers: [AdminAgentsController, AdminFrontendController],
  providers: [AdminAgentsService, AdminAuthService, AdminAuthGuard],
})
export class AdminModule {}
