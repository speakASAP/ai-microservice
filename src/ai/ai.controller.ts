import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { AiService } from './ai.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow } from '../contracts/parse-or-throw';
import { AiCompleteRequestSchema, AiCompleteResponseSchema } from '../contracts';
import type { AiCompleteRequestInput } from '../contracts';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Post('complete')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(AiCompleteRequestSchema))
  async complete(@Body() body: AiCompleteRequestInput) {
    const result = await this.aiService.complete(body);
    return parseOrThrow(AiCompleteResponseSchema, result, 'ai.complete.response');
  }
}
