import { Module } from '@nestjs/common';
import { LlmClient } from './llm.client';
import { GenerateService } from './generate.service';

@Module({
  providers: [LlmClient, GenerateService],
  exports: [LlmClient],
})
export class TeacherAssistantModule {}
