import { Test, TestingModule } from '@nestjs/testing';
import { TaskService } from '../../src/task/task.service';
import { InternalServerErrorException, BadRequestException } from '@nestjs/common';

const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('TaskService', () => {
  let service: TaskService;

  beforeEach(async () => {
    process.env.LITELLM_BASE_URL = 'http://litellm:4000';
    const module: TestingModule = await Test.createTestingModule({
      providers: [TaskService],
    }).compile();
    service = module.get<TaskService>(TaskService);
    jest.clearAllMocks();
  });

  afterEach(() => {
    delete process.env.LITELLM_BASE_URL;
  });

  it('throws BadRequestException when both transcript and textNote are absent', async () => {
    await expect(service.draftTask({})).rejects.toThrow(BadRequestException);
  });

  it('returns structured draft from LiteLLM JSON response', async () => {
    const aiJson = { title: 'Natřít okna', description: 'Pomozte natřít okna ve třídě 2A.', priority: 'normal' };
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: JSON.stringify(aiJson) } }],
      }),
    });
    const result = await service.draftTask({ transcript: 'Potřebujeme natřít okna ve třídě 2A', language: 'cs' });
    expect(result.title).toBe('Natřít okna');
    expect(result.description).toContain('2A');
    expect(result.priority).toBe('normal');
    expect(result.modelTier).toBe('smart');
  });

  it('falls back gracefully and returns raw text when LiteLLM returns invalid JSON', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'not json' } }],
      }),
    });
    const result = await service.draftTask({ transcript: 'Opravte lavice', language: 'cs' });
    expect(result.title).toBeTruthy();
    expect(result.description).toBeTruthy();
  });

  it('throws InternalServerErrorException when LITELLM_BASE_URL not set', async () => {
    delete process.env.LITELLM_BASE_URL;
    await expect(service.draftTask({ transcript: 'test' })).rejects.toThrow(InternalServerErrorException);
  });
});
