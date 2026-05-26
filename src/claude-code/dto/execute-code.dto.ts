import { IsString, IsUUID, IsNumber, IsOptional, Min, Max, IsNotEmpty, IsIn } from 'class-validator';

/**
 * Request DTO for executing code in a repository.
 * Validates instructions and execution parameters before submission to job queue.
 */
export class ExecuteCodeDto {
  /** Unique identifier linking this code execution to a parent task */
  @IsUUID()
  declare taskId: string;

  /** Absolute path to the repository directory */
  @IsString()
  @IsNotEmpty()
  declare repoPath: string;

  /** Git branch or commit SHA to checkout */
  @IsString()
  @IsNotEmpty()
  declare branch: string;

  /** Shell commands or script to execute in the repository */
  @IsString()
  @IsNotEmpty()
  declare instructions: string;

  /** Human-readable description of expected behavior after execution */
  @IsString()
  @IsOptional()
  expectedOutcome?: string;

  /** Execution timeout in seconds. Range: 10–3600. Default: 300 (5 min) */
  @IsNumber()
  @Min(10)
  @Max(3600)
  @IsOptional()
  timeoutSeconds?: number;

  /** Optional shell script to validate execution success (exit code 0 = pass) */
  @IsString()
  @IsOptional()
  validationScript?: string;

  /**
   * `code` — git worktree + `claude code` (default).
   * `print` — `claude --print` in repoPath (planning / semantic validation; no code edits).
   */
  @IsString()
  @IsIn(['code', 'print'])
  @IsOptional()
  executionMode?: 'code' | 'print';

  /** Claude model for print mode (e.g. claude-sonnet-4-6). Defaults to CC_PRINT_MODEL env. */
  @IsString()
  @IsOptional()
  model?: string;
}
