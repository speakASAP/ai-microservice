import { Test, TestingModule } from '@nestjs/testing';
import { HttpException, HttpStatus } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { ClaudeCodeController } from '../../src/claude-code/claude-code.controller';
import { ClaudeCodeService } from '../../src/claude-code/claude-code.service';
import type { ExecuteCodeRequestInput } from '../../src/contracts';
import { JobEnqueueResponseDto } from '../../src/claude-code/dto/job-enqueue-response.dto';
import { JobStatusResponseDto } from '../../src/claude-code/dto/job-status-response.dto';
import { JobStatus } from '../../src/claude-code/job-status.enum';

describe('ClaudeCodeController', () => {
  let controller: ClaudeCodeController;
  let mockService: any;

  beforeEach(async () => {
    mockService = {
      enqueueJob: jest.fn(),
      getJobStatus: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [ClaudeCodeController],
      providers: [
        {
          provide: ClaudeCodeService,
          useValue: mockService,
        },
      ],
    }).compile();

    controller = module.get<ClaudeCodeController>(ClaudeCodeController);
  });

  describe('POST /ai/claude-code-execute', () => {
    it('should return job_id with status queued', async () => {
      const taskId = randomUUID();
      const dto: ExecuteCodeRequestInput = {
        taskId,
        repoPath: '/home/ssf/Documents/Github/beauty',
        branch: 'feat/test',
        instructions: 'Add endpoint',
        expectedOutcome: 'Endpoint returns 200',
      };

      const jobId = randomUUID();
      const response: JobEnqueueResponseDto = {
        jobId,
        taskId,
        status: 'queued' as JobStatus,
        createdAt: new Date(),
        implementationProvider: 'codex',
        intentChecksum: 'checksum-123',
        lifecycleStage: 'queued',
        statusDetail: 'Job accepted and queued for implementation execution.',
        auditSummary: 'provider=codex status=queued intent_checksum=checksum-123 validation=not_run',
      };

      mockService.enqueueJob.mockResolvedValue(response);

      const result = await controller.executeCode(dto);

      expect(result.jobId).toBe(jobId);
      expect(result.status).toBe('queued');
      expect(result.implementationProvider).toBe('codex');
      expect(result.intentChecksum).toBe('checksum-123');
      expect(result.lifecycleStage).toBe('queued');
      expect(result.auditSummary).toContain('provider=codex');
      expect(mockService.enqueueJob).toHaveBeenCalledWith(dto);
    });
  });

  describe('GET /ai/claude-code-execute/:jobId', () => {
    it('should return job status when found', async () => {
      const statusJobId = randomUUID();
      const response: JobStatusResponseDto = {
        jobId: statusJobId,
        taskId: randomUUID(),
        status: 'executing' as JobStatus,
        startedAt: new Date(),
        implementationProvider: 'claude-code',
        intentChecksum: 'checksum-456',
        lifecycleStage: 'executing',
        statusDetail: 'Job is executing with claude-code.',
        outputSummary: 'stdout_lines=0 stderr_lines=0 git_diff_lines=0',
        auditSummary: 'provider=claude-code status=executing intent_checksum=checksum-456 validation=not_run',
      };

      mockService.getJobStatus.mockResolvedValue(response);

      const result = await controller.getStatus(statusJobId) as JobStatusResponseDto;

      expect(result.jobId).toBe(statusJobId);
      expect(result.status).toBe('executing');
      expect(result.implementationProvider).toBe('claude-code');
      expect(result.intentChecksum).toBe('checksum-456');
      expect(result.lifecycleStage).toBe('executing');
      expect(result.outputSummary).toContain('stdout_lines=0');
      expect(mockService.getJobStatus).toHaveBeenCalledWith(statusJobId);
    });

    it('should throw HttpException with 404 when job not found', async () => {
      mockService.getJobStatus.mockResolvedValue(null);

      await expect(controller.getStatus('nonexistent')).rejects.toThrow(HttpException);
      await expect(controller.getStatus('nonexistent')).rejects.toMatchObject({
        status: HttpStatus.NOT_FOUND,
      });
      expect(mockService.getJobStatus).toHaveBeenCalledWith('nonexistent');
    });
  });
});
