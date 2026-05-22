import { IsString, IsOptional, IsNumber, IsObject } from 'class-validator';

export class CompleteRequestDto {
  @IsString()
  model_tier!: string;

  @IsOptional()
  @IsString()
  system_prompt?: string;

  @IsString()
  user_prompt!: string;

  @IsOptional()
  @IsObject()
  output_schema?: Record<string, unknown>;

  @IsOptional()
  @IsNumber()
  max_tokens?: number;

  @IsOptional()
  @IsString()
  correlation_id?: string;
}
