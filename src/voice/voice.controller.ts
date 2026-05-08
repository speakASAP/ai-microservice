import { Controller, Post, Body, HttpCode } from '@nestjs/common';
import { VoiceService } from './voice.service';
import { TranscribeDto } from './dto/transcribe.dto';

@Controller('voice')
export class VoiceController {
  constructor(private readonly voiceService: VoiceService) {}

  @Post('transcribe')
  @HttpCode(200)
  async transcribe(@Body() dto: TranscribeDto): Promise<{ transcript: string }> {
    return this.voiceService.transcribe(dto);
  }
}
