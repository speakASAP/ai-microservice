import { IsString, IsOptional } from 'class-validator';

export class TranscribeDto {
  @IsString()
  fileKey!: string;

  @IsOptional()
  @IsString()
  language?: string;
}
