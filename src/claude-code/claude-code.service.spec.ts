import { ClaudeCodeService } from './claude-code.service';
import { JobStatusResponseSchema } from '../contracts';

describe('ClaudeCodeService', () => {
  const repository = {
    findOne: jest.fn(),
    create: jest.fn(),
    save: jest.fn(),
    update: jest.fn(),
  };

  const amqpConnection = {
    publish: jest.fn(),
  };

  let service: ClaudeCodeService;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new ClaudeCodeService(repository as any, amqpConnection as any);
  });

  it('normalizes queued job null result fields for the status contract', async () => {
    repository.findOne.mockResolvedValue({
      jobId: '550e8400-e29b-41d4-a716-446655440000',
      taskId: '550e8400-e29b-41d4-a716-446655440001',
      status: 'queued',
      startedAt: null,
      completedAt: null,
      exitCode: null,
      stdout: null,
      stderr: null,
      gitDiff: null,
      validationPassed: null,
      validationOutput: null,
      implementationProvider: 'codex',
    });

    const status = await service.getJobStatus('550e8400-e29b-41d4-a716-446655440000');

    expect(status).toMatchObject({
      jobId: '550e8400-e29b-41d4-a716-446655440000',
      taskId: '550e8400-e29b-41d4-a716-446655440001',
      status: 'queued',
      implementationProvider: 'codex',
    });
    expect(JobStatusResponseSchema.safeParse(status).success).toBe(true);
  });
});
