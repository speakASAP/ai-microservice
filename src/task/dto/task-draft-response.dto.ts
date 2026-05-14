export type TaskPriority = 'low' | 'normal' | 'high';

export interface TaskDraftResponse {
  title: string;
  description: string;
  priority: TaskPriority;
  deadline?: string;
  modelTier: string;
}
