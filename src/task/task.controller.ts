import { Controller, Post, Body, HttpCode } from '@nestjs/common';
import { TaskService } from './task.service';
import { TaskDraftDto } from './dto/task-draft.dto';
import { parseOrThrow, TaskDraftResponseSchema } from '../contracts';
import type { TaskDraftResponse } from '../contracts';

@Controller('task')
export class TaskController {
  constructor(private readonly taskService: TaskService) {}

  @Post('draft')
  @HttpCode(200)
  async draft(@Body() dto: TaskDraftDto): Promise<TaskDraftResponse> {
    const result = await this.taskService.draftTask(dto);
    return parseOrThrow(TaskDraftResponseSchema, result, 'task.draft.response');
  }
}
