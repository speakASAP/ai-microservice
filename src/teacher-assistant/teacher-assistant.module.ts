import { Module } from '@nestjs/common';
import { ServiceIdentityModule } from '../service-identity/service-identity.module';
import { LlmClient } from './llm.client';
import { GenerateService } from './generate.service';
import { ValidateService } from './validate.service';
import { TeacherAssistantController } from './teacher-assistant.controller';

@Module({
  imports: [ServiceIdentityModule],
  controllers: [TeacherAssistantController],
  providers: [LlmClient, GenerateService, ValidateService],
  exports: [LlmClient],
})
export class TeacherAssistantModule {}
