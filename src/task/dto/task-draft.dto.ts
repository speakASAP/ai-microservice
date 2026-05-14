import { IsString, IsOptional } from 'class-validator';

export class TaskDraftDto {
  @IsOptional()
  @IsString()
  transcript?: string;

  @IsOptional()
  @IsString()
  textNote?: string;

  @IsOptional()
  @IsString()
  language?: string;
}
