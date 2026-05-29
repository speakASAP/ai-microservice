import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { TaskService } from './task.service';
import {
  TaskDraftRequestSchema,
  TaskDraftResponseSchema,
  ZodValidationPipe,
  parseOrThrow,
} from '../contracts';
import type { TaskDraftRequestInput, TaskDraftResponse } from '../contracts';

@Controller('task')
export class TaskController {
  constructor(private readonly taskService: TaskService) {}

  @Post('draft')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(TaskDraftRequestSchema))
  async draft(@Body() dto: TaskDraftRequestInput): Promise<TaskDraftResponse> {
    const result = await this.taskService.draftTask(dto);
    return parseOrThrow(TaskDraftResponseSchema, result, 'task.draft.response');
  }
}
