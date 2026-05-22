import { Module } from '@nestjs/common';
import { APP_INTERCEPTOR } from '@nestjs/core';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ClaudeCodeModule } from './claude-code/claude-code.module';
import { ClaudeCodeJob } from './database/entities/claude-code-job.entity';
import { InferenceLog } from './database/entities/inference-log.entity';
import { VoiceModule } from './voice/voice.module';
import { TaskModule } from './task/task.module';
import { AiModule } from './ai/ai.module';
import { ServiceIdentityModule } from './service-identity/service-identity.module';
import { InferenceLogInterceptor } from './service-identity/inference-log.interceptor';
import { HealthController } from './health.controller';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.POSTGRES_HOST || 'localhost',
      port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
      username: process.env.POSTGRES_USER || 'postgres',
      password: process.env.POSTGRES_PASSWORD,
      database: process.env.POSTGRES_DB || 'ai_microservice',
      entities: [ClaudeCodeJob, InferenceLog],
      synchronize: process.env.NODE_ENV !== 'production',
      logging: process.env.DEBUG_SQL === 'true',
    }),
    TypeOrmModule.forFeature([InferenceLog]),
    ServiceIdentityModule,
    ClaudeCodeModule,
    VoiceModule,
    TaskModule,
    AiModule,
  ],
  controllers: [HealthController],
  providers: [
    {
      provide: APP_INTERCEPTOR,
      useClass: InferenceLogInterceptor,
    },
  ],
})
export class AppModule {}
