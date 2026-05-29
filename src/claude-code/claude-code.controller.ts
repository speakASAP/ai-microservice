import { Controller, Post, Get, Body, Param, HttpCode } from '@nestjs/common';
import { ClaudeCodeService } from './claude-code.service';
import { ExecuteCodeDto } from './dto/execute-code.dto';
import { parseOrThrow, JobEnqueueResponseSchema, JobStatusResponseSchema } from '../contracts';

@Controller('ai/claude-code-execute')
export class ClaudeCodeController {
  constructor(private service: ClaudeCodeService) {}

  @Post()
  @HttpCode(201)
  async executeCode(@Body() dto: ExecuteCodeDto) {
    const result = await this.service.enqueueJob(dto);
    return parseOrThrow(JobEnqueueResponseSchema, result, 'claude-code.enqueue.response');
  }

  @Get(':jobId')
  async getStatus(@Param('jobId') jobId: string) {
    const status = await this.service.getJobStatus(jobId);
    if (!status) {
      return { error: 'Job not found' };
    }
    return parseOrThrow(JobStatusResponseSchema, status, 'claude-code.status.response');
  }
}
