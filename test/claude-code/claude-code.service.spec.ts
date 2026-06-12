import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { BadRequestException } from '@nestjs/common';
import { AmqpConnection } from '@golevelup/nestjs-rabbitmq';
import { ClaudeCodeService } from '../../src/claude-code/claude-code.service';
import { ClaudeCodeJob } from '../../src/database/entities/claude-code-job.entity';
import type { ExecuteCodeRequestInput } from '../../src/contracts';
import { JobStatus } from '../../src/claude-code/job-status.enum';
import { randomUUID } from 'crypto';

describe('ClaudeCodeService', () => {
  let service: ClaudeCodeService;
  let mockRepository: any;
  let mockAmqpConnection: any;

  beforeEach(async () => {
    mockRepository = {
      save: jest.fn(),
      findOne: jest.fn(),
      create: jest.fn((data) => data),
      update: jest.fn(),
    };

    mockAmqpConnection = {
      publish: jest.fn().mockResolvedValue(undefined),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ClaudeCodeService,
        {
          provide: getRepositoryToken(ClaudeCodeJob),
          useValue: mockRepository,
        },
        {
          provide: AmqpConnection,
          useValue: mockAmqpConnection,
        },
      ],
    }).compile();

    service = module.get<ClaudeCodeService>(ClaudeCodeService);
  });

  describe('enqueueJob', () => {
    it('should create and save a job record with status=queued', async () => {
      const taskId = randomUUID();
      const dto: ExecuteCodeRequestInput = {
        taskId,
        repoPath: '/home/ssf/Documents/Github/beauty',
        branch: 'feat/test',
        instructions: 'Add health check',
        expectedOutcome: 'GET /health returns 200',
        timeoutSeconds: 300,
      };

      const mockJobEntity = {
        jobId: 'job-123',
        taskId: dto.taskId,
        repoPath: dto.repoPath,
        branch: dto.branch,
        instructions: dto.instructions,
        expectedOutcome: dto.expectedOutcome,
        timeoutSeconds: dto.timeoutSeconds,
        status: 'queued',
        createdAt: new Date(),
      };

      mockRepository.create.mockReturnValue(mockJobEntity);
      mockRepository.save.mockResolvedValue(mockJobEntity);

      const result = await service.enqueueJob(dto);

      expect(result.jobId).toBe('job-123');
      expect(result.status).toBe('queued');
      expect(result.taskId).toBe(taskId);
      expect(mockRepository.create).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'queued',
          repoPath: dto.repoPath,
          taskId: dto.taskId,
        }),
      );
      expect(mockRepository.save).toHaveBeenCalledWith(mockJobEntity);
      expect(mockAmqpConnection.publish).toHaveBeenCalledWith(
        'claude-code-exchange',
        'claude-code.execute',
        expect.objectContaining({
          taskId: dto.taskId,
          repoPath: dto.repoPath,
          branch: dto.branch,
          instructions: dto.instructions,
        }),
      );
    });

    it('should use default timeout if not provided', async () => {
      const taskId = randomUUID();
      const dto: ExecuteCodeRequestInput = {
        taskId,
        repoPath: '/home/ssf/Documents/Github/test',
        branch: 'main',
        instructions: 'Test command',
      };

      const mockJobEntity = {
        jobId: 'job-456',
        ...dto,
        timeoutSeconds: 300,
        status: 'queued',
        createdAt: new Date(),
      };

      mockRepository.create.mockReturnValue(mockJobEntity);
      mockRepository.save.mockResolvedValue(mockJobEntity);

      const result = await service.enqueueJob(dto);

      expect(result.jobId).toBe('job-456');
      expect(mockRepository.create).toHaveBeenCalledWith(
        expect.objectContaining({
          timeoutSeconds: 300,
        }),
      );
    });
  });

  describe('getJobStatus', () => {
    it('should return job details by job_id', async () => {
      const jobId = 'job-123';
      const mockJob = {
        jobId,
        taskId: randomUUID(),
        status: 'executing' as JobStatus,
        startedAt: new Date(),
        completedAt: null,
        exitCode: null,
        stdout: null,
        stderr: null,
        gitDiff: null,
        validationPassed: null,
        validationOutput: null,
      };

      mockRepository.findOne.mockResolvedValue(mockJob);

      const result = await service.getJobStatus(jobId);

      expect(result).not.toBeNull();
      expect(result?.jobId).toBe(jobId);
      expect(result?.status).toBe('executing');
      expect(mockRepository.findOne).toHaveBeenCalledWith({
        where: { jobId },
      });
    });

    it('should return null if job not found', async () => {
      const jobId = 'nonexistent-job';
      mockRepository.findOne.mockResolvedValue(null);

      const result = await service.getJobStatus(jobId);

      expect(result).toBeNull();
      expect(mockRepository.findOne).toHaveBeenCalledWith({
        where: { jobId },
      });
    });

    it('should return all job status fields', async () => {
      const jobId = 'job-456';
      const completedAt = new Date();
      const mockJob = {
        jobId,
        taskId: randomUUID(),
        status: 'success',
        startedAt: new Date(),
        completedAt,
        exitCode: 0,
        stdout: 'Success output',
        stderr: '',
        gitDiff: 'diff content',
        validationPassed: true,
        validationOutput: 'Validation passed',
      };

      mockRepository.findOne.mockResolvedValue(mockJob);

      const result = await service.getJobStatus(jobId);

      expect(result).toEqual(expect.objectContaining({
        jobId: mockJob.jobId,
        taskId: mockJob.taskId,
        status: mockJob.status,
        startedAt: mockJob.startedAt,
        completedAt: mockJob.completedAt,
        exitCode: mockJob.exitCode,
        stdout: mockJob.stdout,
        stderr: mockJob.stderr,
        gitDiff: mockJob.gitDiff,
        validationPassed: mockJob.validationPassed,
        validationOutput: mockJob.validationOutput,
        lifecycleStage: 'completed',
        statusDetail: 'Job completed successfully with claude-code.',
        outputSummary: expect.stringContaining('stdout_lines=1'),
        validationSummary: expect.stringContaining('validation=passed'),
        auditSummary: expect.stringContaining('status=success'),
      }));
    });
  });

  describe('updateJobExecution', () => {
    it('should update job execution results', async () => {
      const jobId = 'job-789';
      const updateData = {
        status: 'success' as JobStatus,
        exitCode: 0,
        stdout: 'Command executed',
        completedAt: new Date(),
      };

      mockRepository.findOne.mockResolvedValue({
        jobId,
        status: 'executing' as JobStatus,
      });
      mockRepository.update.mockResolvedValue({ affected: 1 });

      await service.updateJobExecution(jobId, updateData);

      expect(mockRepository.update).toHaveBeenCalledWith(
        { jobId },
        expect.objectContaining({
          ...updateData,
          lifecycleStage: 'completed',
          statusDetail: expect.stringContaining('completed successfully'),
          outputSummary: expect.stringContaining('stdout_lines=1'),
          auditSummary: expect.stringContaining('status=success'),
          lastObservedAt: expect.any(Date),
        }),
      );
    });

    it('should update job status with valid transition', async () => {
      const jobId = 'job-123';
      mockRepository.findOne.mockResolvedValue({
        jobId,
        status: 'queued' as JobStatus,
      });

      await service.updateJobExecution(jobId, { status: 'executing' as JobStatus });

      expect(mockRepository.update).toHaveBeenCalledWith(
        { jobId },
        expect.objectContaining({ status: 'executing' }),
      );
    });

    it('should reject invalid status transition', async () => {
      const jobId = 'job-123';
      mockRepository.findOne.mockResolvedValue({
        jobId,
        status: 'success' as JobStatus,
      });

      await expect(
        service.updateJobExecution(jobId, { status: 'executing' as JobStatus }),
      ).rejects.toThrow(BadRequestException);
    });
  });

  describe('enqueueJob error handling', () => {
    it('should persist job even if RabbitMQ publish fails', async () => {
      const dto: ExecuteCodeRequestInput = {
        taskId: randomUUID(),
        repoPath: '/home/ssf/Documents/Github/beauty',
        branch: 'feat/test',
        instructions: 'Add endpoint',
      };

      const savedJob = {
        jobId: 'job-456',
        taskId: dto.taskId,
        repoPath: dto.repoPath,
        branch: dto.branch,
        instructions: dto.instructions,
        status: 'queued' as JobStatus,
        createdAt: new Date(),
      };

      mockRepository.save.mockResolvedValue(savedJob);
      mockAmqpConnection.publish.mockRejectedValue(new Error('AMQP down'));

      const result = await service.enqueueJob(dto);

      expect(result.jobId).toBe('job-456');
      expect(result.status).toBe('queued');
      expect(mockRepository.save).toHaveBeenCalled();
    });
  });

  describe('markJobRetrying', () => {
    it('updates job to retrying status and appends error history', async () => {
      const jobId = `job-${randomUUID()}`;
      mockRepository.findOne.mockResolvedValue({
        jobId,
        status: 'executing',
        retryCount: 0,
        errorHistory: null,
      });
      mockRepository.update.mockResolvedValue(undefined);

      await service.markJobRetrying(jobId, {
        retryCount: 1,
        nextRetryAt: new Date(Date.now() + 30_000),
        lastError: 'ETIMEDOUT: git fetch failed',
      });

      expect(mockRepository.update).toHaveBeenCalledWith(
        { jobId },
        expect.objectContaining({
          status: 'retrying',
          retryCount: 1,
          errorHistory: expect.arrayContaining([
            expect.objectContaining({ attempt: 1, error: 'ETIMEDOUT: git fetch failed' }),
          ]),
        }),
      );
    });

    it('returns silently when job not found', async () => {
      mockRepository.findOne.mockResolvedValue(null);
      await expect(
        service.markJobRetrying('job-unknown', {
          retryCount: 1,
          nextRetryAt: new Date(),
          lastError: 'error',
        }),
      ).resolves.not.toThrow();
      expect(mockRepository.update).not.toHaveBeenCalled();
    });
  });

  describe('isValidStatusTransition (updated for retrying)', () => {
    it('allows executing → retrying', async () => {
      const jobId = `job-${randomUUID()}`;
      mockRepository.findOne.mockResolvedValue({ jobId, status: 'executing' });
      mockRepository.update.mockResolvedValue(undefined);
      await expect(
        service.updateJobExecution(jobId, { status: 'retrying' as JobStatus }),
      ).resolves.not.toThrow();
    });

    it('allows retrying → executing', async () => {
      const jobId = `job-${randomUUID()}`;
      mockRepository.findOne.mockResolvedValue({ jobId, status: 'retrying' });
      mockRepository.update.mockResolvedValue(undefined);
      await expect(
        service.updateJobExecution(jobId, { status: 'executing' as JobStatus }),
      ).resolves.not.toThrow();
    });

    it('rejects retrying → success (must go through executing)', async () => {
      const jobId = `job-${randomUUID()}`;
      mockRepository.findOne.mockResolvedValue({ jobId, status: 'retrying' });
      await expect(
        service.updateJobExecution(jobId, { status: 'success' as JobStatus }),
      ).rejects.toThrow(BadRequestException);
    });
  });
});
