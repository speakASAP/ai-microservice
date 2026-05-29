import { Controller, Post, Get, Body, Param, HttpCode, HttpStatus, HttpException, UsePipes } from '@nestjs/common';
import { ClaudeCodeService } from './claude-code.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow, ExecuteCodeRequestSchema, JobEnqueueResponseSchema, JobStatusResponseSchema, NotFoundResponseSchema } from '../contracts';
import type { ExecuteCodeRequestInput } from '../contracts';

@Controller('ai/claude-code-execute')
export class ClaudeCodeController {
  constructor(private service: ClaudeCodeService) {}

  @Post()
  @HttpCode(201)
  @UsePipes(new ZodValidationPipe(ExecuteCodeRequestSchema))
  async executeCode(@Body() body: ExecuteCodeRequestInput) {
    const result = await this.service.enqueueJob(body);
    return parseOrThrow(JobEnqueueResponseSchema, result, 'claude-code.execute.response');
  }

  @Get(':jobId')
  async getStatus(@Param('jobId') jobId: string) {
    const status = await this.service.getJobStatus(jobId);
    if (!status) {
      const body = parseOrThrow(NotFoundResponseSchema, { error: 'Job not found' }, 'claude-code.status.not-found');
      throw new HttpException(body, HttpStatus.NOT_FOUND);
    }
    return parseOrThrow(JobStatusResponseSchema, status, 'claude-code.status.response');
  }
}
