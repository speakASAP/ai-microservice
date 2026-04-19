import { Entity, PrimaryColumn, Column, CreateDateColumn, UpdateDateColumn, Index } from 'typeorm';

@Entity('claude_code_jobs')
@Index('idx_taskId', ['taskId'])
@Index('idx_status', ['status'])
@Index('idx_createdAt', ['createdAt'])
export class ClaudeCodeJob {
  // UUID format (36 chars)
  @PrimaryColumn('varchar', { length: 36 })
  jobId!: string;

  // Must be valid UUID
  @Column('uuid')
  taskId!: string;

  @Column('text')
  repoPath!: string;

  @Column('varchar')
  branch!: string;

  @Column('text')
  instructions!: string;

  @Column('text', { nullable: true })
  expectedOutcome?: string;

  @Column('integer', { default: 300 })
  timeoutSeconds!: number;

  @Column('text', { nullable: true })
  validationScript?: string;

  @Column('varchar', { default: 'queued' })
  status!: 'queued' | 'executing' | 'success' | 'failed' | 'timeout' | 'retrying';

  @Column('timestamptz', { nullable: true })
  startedAt?: Date;

  @Column('timestamptz', { nullable: true })
  completedAt?: Date;

  @Column('integer', { nullable: true })
  exitCode?: number;

  @Column('text', { nullable: true })
  stdout?: string;

  @Column('text', { nullable: true })
  stderr?: string;

  @Column('text', { nullable: true })
  gitDiff?: string;

  @Column('boolean', { nullable: true })
  validationPassed?: boolean;

  @Column('text', { nullable: true })
  validationOutput?: string;

  @CreateDateColumn()
  createdAt!: Date;

  @UpdateDateColumn()
  updatedAt!: Date;

  @Column('integer', { default: 0 })
  retryCount!: number;

  @Column('integer', { default: 3 })
  maxRetries!: number;

  @Column('timestamptz', { nullable: true })
  nextRetryAt?: Date;

  @Column('timestamptz', { nullable: true })
  lastErrorAt?: Date;

  @Column('jsonb', { nullable: true })
  errorHistory?: Array<{ attempt: number; error: string; timestamp: string }>;
}
