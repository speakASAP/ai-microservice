import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AiAgent } from '../database/entities/ai-agent.entity';
import { AdminAgentsController } from './admin-agents.controller';
import { AdminAgentsService } from './admin-agents.service';
import { AdminFrontendController } from './admin-frontend.controller';

@Module({
  imports: [TypeOrmModule.forFeature([AiAgent])],
  controllers: [AdminAgentsController, AdminFrontendController],
  providers: [AdminAgentsService],
})
export class AdminModule {}
