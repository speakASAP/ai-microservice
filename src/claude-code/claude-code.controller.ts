import { Controller, Post, Get, Body, Param, HttpCode } from '@nestjs/common';
import { ClaudeCodeService } from './claude-code.service';
import { ExecuteCodeDto } from './dto/execute-code.dto';
import { JobEnqueueResponseDto } from './dto/job-enqueue-response.dto';
import { JobStatusResponseDto } from './dto/job-status-response.dto';

/**
 * REST endpoints for Claude Code job execution.
 * POST /ai/claude-code-execute - Submit a code execution job
 * GET /ai/claude-code-execute/:jobId - Poll job status
 */
@Controller('ai/claude-code-execute')
export class ClaudeCodeController {
  constructor(private service: ClaudeCodeService) {}

  /**
   * Submit a code execution job.
   * Returns job ID and initial status (always 'queued').
   */
  @Post()
  @HttpCode(201)
  async executeCode(@Body() dto: ExecuteCodeDto): Promise<JobEnqueueResponseDto> {
    return this.service.enqueueJob(dto);
  }

  /**
   * Poll job status and results by job ID.
   * Returns full job details if found, error object if not.
   */
  @Get(':jobId')
  async getStatus(@Param('jobId') jobId: string): Promise<JobStatusResponseDto | { error: string }> {
    const status = await this.service.getJobStatus(jobId);
    if (!status) {
      return { error: 'Job not found' };
    }
    return status;
  }
}
