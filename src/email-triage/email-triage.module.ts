import { Module } from '@nestjs/common';
import { EmailTriageController } from './email-triage.controller';
import { EmailTriageService } from './email-triage.service';
import { AiModule } from '../ai/ai.module';

@Module({
  imports: [AiModule],
  controllers: [EmailTriageController],
  providers: [EmailTriageService],
  exports: [EmailTriageService],
})
export class EmailTriageModule {}
