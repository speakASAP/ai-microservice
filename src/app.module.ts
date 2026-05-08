import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ClaudeCodeModule } from './claude-code/claude-code.module';
import { ClaudeCodeJob } from './database/entities/claude-code-job.entity';
import { VoiceModule } from './voice/voice.module';

/**
 * Root application module.
 * Initializes database connection and registers all feature modules.
 */
@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.POSTGRES_HOST || 'localhost',
      port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
      username: process.env.POSTGRES_USER || 'postgres',
      password: process.env.POSTGRES_PASSWORD,
      database: process.env.POSTGRES_DB || 'ai_microservice',
      entities: [ClaudeCodeJob],
      synchronize: process.env.NODE_ENV !== 'production',
      logging: process.env.DEBUG_SQL === 'true',
    }),
    ClaudeCodeModule,
    VoiceModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
