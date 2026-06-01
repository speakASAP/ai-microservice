import { Module } from '@nestjs/common';
import { AiController } from './ai.controller';
import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';

@Module({
  controllers: [AiController],
  providers: [AiService, LoggingClient],
  exports: [AiService],
})
export class AiModule {}
