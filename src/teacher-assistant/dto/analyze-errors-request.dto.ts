import { Type } from 'class-transformer';
import {
  ArrayMinSize,
  IsArray,
  IsBoolean,
  IsInt,
  IsNotEmpty,
  IsString,
  Min,
  ValidateIf,
  ValidateNested,
} from 'class-validator';
import { AnalyzeErrorsRequest, AnalyzeFailure } from '../contracts';

/**
 * `implements AnalyzeFailure` so any drift between this DTO and the shared
 * contract in contracts.ts becomes a compile error, not a silent runtime
 * divergence.
 */
export class AnalyzeFailureDto implements AnalyzeFailure {
  @IsString()
  @IsNotEmpty()
  answer!: string;

  @IsString()
  @IsNotEmpty()
  sentence!: string;

  // `string | null` in the contract: a REQUIRED key whose value may be null.
  // @IsOptional() would be wrong here — it skips validation for a missing key
  // too, silently accepting `undefined` where the contract guarantees `null`.
  // @ValidateIf runs @IsString() for every value except exactly `null`, which
  // correctly rejects a missing key while still accepting null. Mirrors
  // ValidateDrillRequestDto's `hint` field.
  @ValidateIf((o) => o.prompt !== null)
  @IsString()
  prompt!: string | null;

  @IsArray()
  @IsString({ each: true })
  wrongAttempts!: string[];

  @IsBoolean()
  revealed!: boolean;

  @IsInt()
  @Min(1)
  mistakeCount!: number;
}

/**
 * `implements AnalyzeErrorsRequest` so any drift between this DTO and the
 * shared contract in contracts.ts becomes a compile error, not a silent
 * runtime divergence.
 */
export class AnalyzeErrorsRequestDto implements AnalyzeErrorsRequest {
  @IsString()
  @IsNotEmpty()
  languageCode!: string;

  @IsString()
  @IsNotEmpty()
  materialLanguage!: string;

  // `string | null` in the contract: a REQUIRED key whose value may be null.
  // Same reasoning as AnalyzeFailureDto.prompt above.
  @ValidateIf((o) => o.level !== null)
  @IsString()
  level!: string | null;

  // An empty allow-list would make every cluster invalid — the model would
  // have nothing valid to file a gap under.
  @IsArray()
  @ArrayMinSize(1)
  @IsString({ each: true })
  allowedTopicSlugs!: string[];

  // An empty failure list means the caller should have short-circuited
  // before spending a model call.
  @IsArray()
  @ArrayMinSize(1)
  @ValidateNested({ each: true })
  @Type(() => AnalyzeFailureDto)
  failures!: AnalyzeFailure[];

  @IsString()
  @IsNotEmpty()
  correlationId!: string;
}
