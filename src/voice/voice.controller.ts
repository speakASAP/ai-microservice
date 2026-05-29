import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { VoiceService } from './voice.service';
import {
  TranscribeRequestSchema,
  TranscribeResponseSchema,
  ZodValidationPipe,
  parseOrThrow,
} from '../contracts';
import type { TranscribeRequestInput, TranscribeResponse } from '../contracts';

@Controller('voice')
export class VoiceController {
  constructor(private readonly voiceService: VoiceService) {}

  @Post('transcribe')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(TranscribeRequestSchema))
  async transcribe(@Body() dto: TranscribeRequestInput): Promise<TranscribeResponse> {
    const result = await this.voiceService.transcribe(dto);
    return parseOrThrow(TranscribeResponseSchema, result, 'voice.transcribe.response');
  }
}
