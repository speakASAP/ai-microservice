import { Controller, Post, Body, HttpCode } from '@nestjs/common';
import { TaskService } from './task.service';
import { TaskDraftDto } from './dto/task-draft.dto';
import type { TaskDraftResponse } from './dto/task-draft-response.dto';

@Controller('task')
export class TaskController {
  constructor(private readonly taskService: TaskService) {}

  @Post('draft')
  @HttpCode(200)
  async draft(@Body() dto: TaskDraftDto): Promise<TaskDraftResponse> {
    return this.taskService.draftTask(dto);
  }
}
