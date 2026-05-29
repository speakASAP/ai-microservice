import { Test, TestingModule } from '@nestjs/testing';
import { randomUUID } from 'crypto';
import { ClaudeCodeController } from '../../src/claude-code/claude-code.controller';
import { ClaudeCodeService } from '../../src/claude-code/claude-code.service';
import { ExecuteCodeDto } from '../../src/claude-code/dto/execute-code.dto';
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
      const dto: ExecuteCodeDto = {
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
      };

      mockService.enqueueJob.mockResolvedValue(response);

      const result = await controller.executeCode(dto);

      expect(result.jobId).toBe(jobId);
      expect(result.status).toBe('queued');
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
      };

      mockService.getJobStatus.mockResolvedValue(response);

      const result = await controller.getStatus(statusJobId) as JobStatusResponseDto;

      expect(result.jobId).toBe(statusJobId);
      expect(result.status).toBe('executing');
      expect(mockService.getJobStatus).toHaveBeenCalledWith(statusJobId);
    });

    it('should return error object when job not found', async () => {
      mockService.getJobStatus.mockResolvedValue(null);

      const result = await controller.getStatus('nonexistent');

      expect(result).toEqual({ error: 'Job not found' });
      expect(mockService.getJobStatus).toHaveBeenCalledWith('nonexistent');
    });
  });
});
