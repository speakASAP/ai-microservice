import { Module } from '@nestjs/common';
import { LlmClient } from './llm.client';
import { GenerateService } from './generate.service';
import { ValidateService } from './validate.service';

@Module({
  providers: [LlmClient, GenerateService, ValidateService],
  exports: [LlmClient],
})
export class TeacherAssistantModule {}
