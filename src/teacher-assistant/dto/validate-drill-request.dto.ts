import { Type } from 'class-transformer';
import {
  IsArray,
  IsIn,
  IsInt,
  IsNotEmpty,
  IsString,
  Min,
  ValidateIf,
  ValidateNested,
} from 'class-validator';
import { CefrLevel, DrillBlank, DrillTemplate, ValidateDrillRequest } from '../contracts';
import { CEFR_LEVELS, TopicDto } from './common.dto';

class DrillBlankDto implements DrillBlank {
  @IsInt()
  @Min(0)
  index!: number;

  @IsString()
  prompt!: string;

  @IsString()
  answer!: string;

  @IsArray()
  @IsString({ each: true })
  alternatives!: string[];
}

/** Matches the inline item shape in ValidateDrillRequest — NOT the DrillItemDTO contract. */
class ValidateItemDto {
  @IsInt()
  itemRef!: number;

  @IsString()
  @IsNotEmpty()
  template!: DrillTemplate;

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => DrillBlankDto)
  blanks!: DrillBlank[];

  // hint is `string | null` in the contract: a REQUIRED key whose value may
  // be null. @IsOptional() would be wrong here — it skips validation for a
  // missing key too, silently accepting `undefined` where the contract
  // guarantees `null`. @ValidateIf runs @IsString() for every value except
  // exactly `null`, which correctly rejects a missing key while still
  // accepting null.
  @ValidateIf((o) => o.hint !== null)
  @IsString()
  hint!: string | null;
}

/**
 * `implements ValidateDrillRequest` so any drift between this DTO and the
 * shared contract in contracts.ts becomes a compile error, not a silent
 * runtime divergence.
 */
export class ValidateDrillRequestDto implements ValidateDrillRequest {
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

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => ValidateItemDto)
  items!: { itemRef: number; template: DrillTemplate; blanks: DrillBlank[]; hint: string | null }[];

  @IsString()
  @IsNotEmpty()
  correlationId!: string;
}
