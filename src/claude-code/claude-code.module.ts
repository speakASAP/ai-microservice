import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { RabbitModule } from '@golevelup/nestjs-rabbitmq';
import { ClaudeCodeController } from './claude-code.controller';
import { ClaudeCodeService } from './claude-code.service';
import { ClaudeCodeConsumer } from './claude-code.consumer';
import { LoggingClient } from './logging.client';
import { ClaudeCodeJob } from '../database/entities/claude-code-job.entity';
import { getRabbitMQConfig } from '../config/rabbitmq.config';

/**
 * Claude Code Module - encapsulates async job execution system with smart retry and logging.
 * Imports TypeORM entity, RabbitMQ, and registers controller + providers.
 */
@Module({
  imports: [
    TypeOrmModule.forFeature([ClaudeCodeJob]),
    RabbitModule.forRoot(getRabbitMQConfig()),
  ],
  controllers: [ClaudeCodeController],
  providers: [ClaudeCodeService, ClaudeCodeConsumer, LoggingClient],
  exports: [ClaudeCodeService],
})
export class ClaudeCodeModule {}
