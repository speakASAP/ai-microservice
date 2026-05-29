import { Controller, Post, Body, HttpCode } from '@nestjs/common';
import { VoiceService } from './voice.service';
import { TranscribeDto } from './dto/transcribe.dto';
import { parseOrThrow, TranscribeResponseSchema } from '../contracts';
import type { TranscribeResponse } from '../contracts';

@Controller('voice')
export class VoiceController {
  constructor(private readonly voiceService: VoiceService) {}

  @Post('transcribe')
  @HttpCode(200)
  async transcribe(@Body() dto: TranscribeDto): Promise<TranscribeResponse> {
    const result = await this.voiceService.transcribe(dto);
    return parseOrThrow(TranscribeResponseSchema, result, 'voice.transcribe.response');
  }
}
