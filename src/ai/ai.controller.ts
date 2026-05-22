import { Controller, Post, Body, HttpCode } from '@nestjs/common';
import { AiService } from './ai.service';
import { CompleteRequestDto } from './dto/complete-request.dto';
import type { AiCompleteResult } from './ai.service';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Post('complete')
  @HttpCode(200)
  async complete(@Body() dto: CompleteRequestDto): Promise<AiCompleteResult> {
    return this.aiService.complete(dto);
  }
}
