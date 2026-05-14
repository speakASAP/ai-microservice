import { Test, TestingModule } from '@nestjs/testing';
import { TaskController } from '../../src/task/task.controller';
import { TaskService } from '../../src/task/task.service';
import { BadRequestException } from '@nestjs/common';

const mockDraftTask = jest.fn();

describe('TaskController', () => {
  let controller: TaskController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [TaskController],
      providers: [{ provide: TaskService, useValue: { draftTask: mockDraftTask } }],
    }).compile();
    controller = module.get<TaskController>(TaskController);
    jest.clearAllMocks();
  });

  it('returns TaskDraftResponse on valid input', async () => {
    mockDraftTask.mockResolvedValue({
      title: 'Natřít okna',
      description: 'Pomozte natřít okna.',
      priority: 'normal',
      modelTier: 'smart',
    });
    const result = await controller.draft({ transcript: 'Natřít okna ve třídě' });
    expect(result.title).toBe('Natřít okna');
  });

  it('propagates BadRequestException from service', async () => {
    mockDraftTask.mockRejectedValue(new BadRequestException('transcript required'));
    await expect(controller.draft({})).rejects.toThrow(BadRequestException);
  });
});
