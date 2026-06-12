import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AiController } from './ai.controller';
import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';
import { AiAgent } from '../database/entities/ai-agent.entity';

@Module({
  imports: [TypeOrmModule.forFeature([AiAgent])],
  controllers: [AiController],
  providers: [AiService, LoggingClient],
  exports: [AiService],
})
export class AiModule {}
