import { Type } from 'class-transformer';
import {
  IsArray,
  IsIn,
  IsInt,
  IsNotEmpty,
  IsOptional,
  IsString,
  Min,
  ValidateNested,
} from 'class-validator';
import { CefrLevel, DrillTemplate, GenerateDrillRequest } from '../contracts';

const CEFR_LEVELS: CefrLevel[] = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'];

/** Matches the inline `{ slug, title, focus? }` topic shape in GenerateDrillRequest. */
class TopicDto {
  @IsString()
  @IsNotEmpty()
  slug!: string;

  @IsString()
  @IsNotEmpty()
  title!: string;

  @IsOptional()
  @IsString()
  focus?: string;
}

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
