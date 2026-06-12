import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, Index } from 'typeorm';

@Entity('inference_logs')
@Index('idx_inference_service_id', ['serviceId'])
@Index('idx_inference_business_id', ['businessId'])
@Index('idx_inference_created_at', ['createdAt'])
export class InferenceLog {
  @PrimaryGeneratedColumn('uuid')
  id!: string;

  @Column('varchar', { length: 100 })
  serviceId!: string;

  @Column('varchar', { length: 200 })
  endpoint!: string;

  @Column('varchar', { length: 20, nullable: true })
  modelTier?: string;

  @Column('varchar', { name: 'business_id', length: 128, nullable: true })
  businessId?: string;

  @Column('varchar', { name: 'model_used', length: 128, nullable: true })
  modelUsed?: string;

  @Column('integer', { name: 'input_tokens', nullable: true })
  inputTokens?: number;

  @Column('integer', { name: 'output_tokens', nullable: true })
  outputTokens?: number;

  @Column('integer', { name: 'token_usage_estimate', nullable: true })
  tokenUsageEstimate?: number;

  @Column('numeric', { name: 'estimated_cost_usd', precision: 12, scale: 8, nullable: true })
  estimatedCostUsd?: string;

  @Column('integer', { nullable: true })
  durationMs?: number;

  @Column('integer', { nullable: true })
  statusCode?: number;

  @CreateDateColumn()
  createdAt!: Date;
}
