import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import request from 'supertest';
import { getRepositoryToken } from '@nestjs/typeorm';
import { AmqpConnection } from '@golevelup/nestjs-rabbitmq';
import { ClaudeCodeController } from '../../src/claude-code/claude-code.controller';
import { ClaudeCodeService } from '../../src/claude-code/claude-code.service';
import { ClaudeCodeConsumer } from '../../src/claude-code/claude-code.consumer';
import { ClaudeCodeJob } from '../../src/database/entities/claude-code-job.entity';
import { randomUUID } from 'crypto';

describe('ClaudeCode E2E', () => {
  let app: INestApplication;
  let mockRepo: any;
  let mockAmqp: any;

  beforeAll(async () => {
    const jobStore = new Map<string, any>();

    mockRepo = {
      create: jest.fn((data) => ({ ...data, createdAt: new Date(), updatedAt: new Date() })),
      save: jest.fn(async (entity) => {
        jobStore.set(entity.jobId, entity);
        return entity;
      }),
      findOne: jest.fn(async ({ where: { jobId } }) => jobStore.get(jobId) || null),
      update: jest.fn(async ({ jobId }, data) => {
        const existing = jobStore.get(jobId);
        if (existing) jobStore.set(jobId, { ...existing, ...data });
      }),
    };

    mockAmqp = {
      publish: jest.fn().mockResolvedValue(undefined),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [ClaudeCodeController],
      providers: [
        ClaudeCodeService,
        ClaudeCodeConsumer,
        { provide: getRepositoryToken(ClaudeCodeJob), useValue: mockRepo },
        { provide: AmqpConnection, useValue: mockAmqp },
      ],
    }).compile();

    app = module.createNestApplication();
    app.useGlobalPipes(new ValidationPipe({ whitelist: true }));
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  describe('POST /ai/claude-code-execute', () => {
    it('returns jobId and status=queued', async () => {
      const res = await request(app.getHttpServer())
        .post('/ai/claude-code-execute')
        .send({
          taskId: randomUUID(),
          repoPath: '/home/ssf/Documents/Github/beauty',
          branch: 'main',
          instructions: 'List all files in src/ directory',
          expectedOutcome: 'File listing complete',
        })
        .expect(201);

      expect(res.body.jobId).toBeDefined();
      expect(res.body.status).toBe('queued');
      expect(res.body.taskId).toBeDefined();
      expect(res.body.createdAt).toBeDefined();
    });

    it('rejects request missing required fields', async () => {
      await request(app.getHttpServer())
        .post('/ai/claude-code-execute')
        .send({ repoPath: '/home/ssf/Documents/Github/beauty' })
        .expect(400);
    });
  });

  describe('GET /ai/claude-code-execute/:jobId', () => {
    it('returns job status when found', async () => {
      const postRes = await request(app.getHttpServer())
        .post('/ai/claude-code-execute')
        .send({
          taskId: randomUUID(),
          repoPath: '/home/ssf/Documents/Github/beauty',
          branch: 'main',
          instructions: 'List files',
          expectedOutcome: 'Done',
        })
        .expect(201);

      const jobId = postRes.body.jobId;
      const getRes = await request(app.getHttpServer())
        .get(`/ai/claude-code-execute/${jobId}`)
        .expect(200);

      expect(getRes.body.jobId).toBe(jobId);
      expect(['queued', 'executing', 'success', 'failed']).toContain(getRes.body.status);
    });

    it('returns error object for unknown jobId', async () => {
      const res = await request(app.getHttpServer())
        .get('/ai/claude-code-execute/job-nonexistent-id')
        .expect(200);

      expect(res.body.error).toBe('Job not found');
    });
  });

  describe('RabbitMQ publish', () => {
    it('publishes message to claude-code-exchange on enqueue', async () => {
      mockAmqp.publish.mockClear();

      await request(app.getHttpServer())
        .post('/ai/claude-code-execute')
        .send({
          taskId: randomUUID(),
          repoPath: '/home/ssf/Documents/Github/beauty',
          branch: 'main',
          instructions: 'Check TypeScript errors',
          expectedOutcome: 'No errors',
        })
        .expect(201);

      expect(mockAmqp.publish).toHaveBeenCalledWith(
        'claude-code-exchange',
        'claude-code.execute',
        expect.objectContaining({ repoPath: '/home/ssf/Documents/Github/beauty' }),
      );
    });
  });
});
