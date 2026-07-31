import { Type } from 'class-transformer';
import {
  IsArray,
  IsIn,
  IsInt,
  IsNotEmpty,
  IsString,
  Min,
  ValidateNested,
} from 'class-validator';
import { CefrLevel, DrillTemplate, GenerateDrillRequest } from '../contracts';
import { CEFR_LEVELS, TopicDto } from './common.dto';

/**
 * `implements GenerateDrillRequest` so any drift between this DTO and the
 * shared contract in contracts.ts becomes a compile error, not a silent
 * runtime divergence.
 */
export class GenerateDrillRequestDto implements GenerateDrillRequest {
  @IsString()
  @IsNotEmpty()
  languageCode!: string;

  @IsString()
  @IsNotEmpty()
  materialLanguage!: string;

  @IsIn([...CEFR_LEVELS, null])
  level!: CefrLevel | null;

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => TopicDto)
  topics!: { slug: string; title: string; focus?: string }[];

  @IsString()
  @IsNotEmpty()
  instructions!: string;

  @IsInt()
  @Min(1)
  count!: number;

  @IsArray()
  @IsString({ each: true })
  knownVocabulary!: string[];

  @IsInt()
  @Min(0)
  maxNewWordsPerSentence!: number;

  @IsArray()
  @IsString({ each: true })
  exampleItems!: DrillTemplate[];

  @IsArray()
  @IsString({ each: true })
  avoidTexts!: string[];

  @IsString()
  @IsNotEmpty()
  correlationId!: string;
}
