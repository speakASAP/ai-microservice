import { Module } from '@nestjs/common';
import { APP_INTERCEPTOR } from '@nestjs/core';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ClaudeCodeModule } from './claude-code/claude-code.module';
import { ClaudeCodeJob } from './database/entities/claude-code-job.entity';
import { InferenceLog } from './database/entities/inference-log.entity';
import { AiAgent } from './database/entities/ai-agent.entity';
import { VoiceModule } from './voice/voice.module';
import { TaskModule } from './task/task.module';
import { AiModule } from './ai/ai.module';
import { EmailTriageModule } from './email-triage/email-triage.module';
import { ShopAssistantModule } from './shop-assistant/shop-assistant.module';
import { AdminModule } from './admin/admin.module';
import { TeacherAssistantModule } from './teacher-assistant/teacher-assistant.module';
import { ServiceIdentityModule } from './service-identity/service-identity.module';
import { InferenceLogInterceptor } from './service-identity/inference-log.interceptor';
import { HealthController } from './health.controller';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      url: process.env.DATABASE_URL,
      host: process.env.DATABASE_URL ? undefined : (process.env.DB_HOST || process.env.POSTGRES_HOST || 'localhost'),
      port: process.env.DATABASE_URL ? undefined : parseInt(process.env.DB_PORT || process.env.POSTGRES_PORT || '5432', 10),
      username: process.env.DATABASE_URL ? undefined : (process.env.DB_USER || process.env.POSTGRES_USER || 'postgres'),
      password: process.env.DATABASE_URL ? undefined : (process.env.DB_PASSWORD || process.env.POSTGRES_PASSWORD),
      database: process.env.DATABASE_URL ? undefined : (process.env.DB_NAME || process.env.POSTGRES_DB || 'ai_microservice'),
      entities: [ClaudeCodeJob, InferenceLog, AiAgent],
      synchronize: process.env.NODE_ENV !== 'production' || process.env.DB_SYNC === 'true' || process.env.DB_AUTO_CREATE === 'true',
      logging: process.env.DEBUG_SQL === 'true',
    }),
    TypeOrmModule.forFeature([InferenceLog]),
    ServiceIdentityModule,
    ClaudeCodeModule,
    VoiceModule,
    TaskModule,
    AiModule,
    EmailTriageModule,
    ShopAssistantModule,
    AdminModule,
    TeacherAssistantModule,
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
