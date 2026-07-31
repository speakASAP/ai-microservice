import { IsNotEmpty, IsOptional, IsString } from 'class-validator';
import { CefrLevel } from '../contracts';

/** Single definition shared by every teacher-assistant DTO — two copies of this
 *  list in two request DTOs could drift silently. */
export const CEFR_LEVELS: CefrLevel[] = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'];

/**
 * Matches the inline `{ slug, title, focus? }` topic shape used by both
 * GenerateDrillRequest and ValidateDrillRequest.
 */
export class TopicDto {
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
