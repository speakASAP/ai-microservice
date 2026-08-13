import { Module } from '@nestjs/common';
import { ServiceIdentityModule } from '../service-identity/service-identity.module';
import { LlmClient } from './llm.client';
import { GenerateService } from './generate.service';
import { ValidateService } from './validate.service';
import { AnalyzeService } from './analyze.service';
import { TeacherAssistantController } from './teacher-assistant.controller';

@Module({
  imports: [ServiceIdentityModule],
  controllers: [TeacherAssistantController],
  // No `exports`: nothing outside this module imports LlmClient. Re-add it the
  // day something does, not speculatively.
  providers: [LlmClient, GenerateService, ValidateService, AnalyzeService],
})
export class TeacherAssistantModule {}
