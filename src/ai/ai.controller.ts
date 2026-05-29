import { Controller, Post, Body, HttpCode } from '@nestjs/common';
import { AiService } from './ai.service';
import { CompleteRequestDto } from './dto/complete-request.dto';
import { parseOrThrow, AiCompleteResponseSchema } from '../contracts';
import type { AiCompleteResponse } from '../contracts';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Post('complete')
  @HttpCode(200)
  async complete(@Body() dto: CompleteRequestDto): Promise<AiCompleteResponse> {
    const result = await this.aiService.complete(dto);
    return parseOrThrow(AiCompleteResponseSchema, result, 'ai.complete.response');
  }
}
