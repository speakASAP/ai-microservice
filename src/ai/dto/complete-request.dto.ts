import { IsString, IsOptional, IsNumber, IsObject, IsEnum, IsNotEmpty } from 'class-validator';

const MODEL_TIERS = ['free', 'cheap', 'smart', 'premium'] as const;

export class CompleteRequestDto {
  @IsOptional()
  @IsString()
  schemaVersion?: string;

  @IsEnum(MODEL_TIERS)
  model_tier!: 'free' | 'cheap' | 'smart' | 'premium';

  @IsOptional()
  @IsString()
  system_prompt?: string;

  @IsString()
  @IsNotEmpty()
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
