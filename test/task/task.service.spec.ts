import { Test, TestingModule } from '@nestjs/testing';
import { BadRequestException } from '@nestjs/common';
import { TaskService } from '../../src/task/task.service';
import { AiService } from '../../src/ai/ai.service';

const mockAiComplete = jest.fn();

const aiServiceMock = {
  complete: mockAiComplete,
};

describe('TaskService', () => {
  let service: TaskService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        TaskService,
        { provide: AiService, useValue: aiServiceMock },
      ],
    }).compile();
    service = module.get<TaskService>(TaskService);
    jest.clearAllMocks();
  });

  it('throws BadRequestException when both transcript and textNote are absent', async () => {
    await expect(service.draftTask({})).rejects.toThrow(BadRequestException);
  });

  it('returns structured draft from AI JSON response', async () => {
    const aiJson = { title: 'Natřít okna', description: 'Pomozte natřít okna ve třídě 2A.', priority: 'normal' };
    mockAiComplete.mockResolvedValue({ text: JSON.stringify(aiJson), model_used: 'claude-sonnet-4-6', inputTokens: 10, outputTokens: 20, token_usage_estimate: 30 });

    const result = await service.draftTask({ transcript: 'Potřebujeme natřít okna ve třídě 2A', language: 'cs' });
    expect(result.title).toBe('Natřít okna');
    expect(result.description).toContain('2A');
    expect(result.priority).toBe('normal');
    expect(result.modelTier).toBe('smart');
  });

  it('falls back gracefully and returns raw text when AI returns invalid JSON', async () => {
    mockAiComplete.mockResolvedValue({ text: 'not json', model_used: 'claude-sonnet-4-6', inputTokens: 5, outputTokens: 5, token_usage_estimate: 10 });

    const result = await service.draftTask({ transcript: 'Opravte lavice', language: 'cs' });
    expect(result.title).toBeTruthy();
    expect(result.description).toBeTruthy();
  });
});
