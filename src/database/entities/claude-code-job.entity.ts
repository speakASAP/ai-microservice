import { Entity, PrimaryColumn, Column, CreateDateColumn, UpdateDateColumn, Index } from 'typeorm';

@Entity('claude_code_jobs')
@Index(['taskId'])
@Index(['status'])
@Index(['createdAt'], { sort: 'DESC' })
export class ClaudeCodeJob {
  // UUID format (36 chars)
  @PrimaryColumn('varchar', { length: 36 })
  jobId: string;

  // Must be valid UUID
  @Column('uuid')
  taskId: string;

  @Column('text')
  repoPath: string;

  @Column('varchar')
  branch: string;

  @Column('text')
  instructions: string;

  @Column('text', { nullable: true })
  expectedOutcome: string;

  @Column('integer', { default: 300 })
  timeoutSeconds: number;

  @Column('text', { nullable: true })
  validationScript: string;

  @Column('varchar', { default: 'queued' })
  status: 'queued' | 'executing' | 'success' | 'failed' | 'timeout';

  @Column('timestamptz', { nullable: true })
  startedAt: Date;

  @Column('timestamptz', { nullable: true })
  completedAt: Date;

  @Column('integer', { nullable: true })
  exitCode: number;

  @Column('text', { nullable: true })
  stdout: string;

  @Column('text', { nullable: true })
  stderr: string;

  @Column('text', { nullable: true })
  gitDiff: string;

  @Column('boolean', { nullable: true })
  validationPassed: boolean;

  @Column('text', { nullable: true })
  validationOutput: string;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}
