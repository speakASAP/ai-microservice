import { Column, CreateDateColumn, Entity, Index, PrimaryGeneratedColumn, UpdateDateColumn } from 'typeorm';

export type AiAgentStatus = 'draft' | 'active' | 'disabled';
export type AiAgentModelTier = 'free' | 'cheap' | 'smart' | 'premium';

@Entity('ai_agents')
@Index('idx_ai_agents_slug', ['slug'], { unique: true })
@Index('idx_ai_agents_status', ['status'])
@Index('idx_ai_agents_service_scope', ['serviceScope'])
@Index('idx_ai_agents_updated_at', ['updatedAt'])
export class AiAgent {
  @PrimaryGeneratedColumn('uuid')
  id!: string;

  @Column('varchar', { length: 140 })
  name!: string;

  @Column('varchar', { length: 160 })
  slug!: string;

  @Column('text', { nullable: true })
  description?: string | null;

  @Column('varchar', { length: 24, default: 'draft' })
  status!: AiAgentStatus;

  @Column('varchar', { length: 120, name: 'service_scope' })
  serviceScope!: string;

  @Column('varchar', { length: 240, name: 'route_path', nullable: true })
  routePath?: string | null;

  @Column('varchar', { length: 24, name: 'model_tier', default: 'free' })
  modelTier!: AiAgentModelTier;

  @Column('varchar', { length: 180, name: 'provider_model', nullable: true })
  providerModel?: string | null;

  @Column('numeric', { precision: 4, scale: 2, default: 0.2 })
  temperature!: string;

  @Column('integer', { name: 'max_tokens', default: 1000 })
  maxTokens!: number;

  @Column('text', { name: 'system_prompt', default: '' })
  systemPrompt!: string;

  @Column('text', { name: 'user_prompt_template', default: '' })
  userPromptTemplate!: string;

  @Column('jsonb', { name: 'output_schema', nullable: true })
  outputSchema?: Record<string, unknown> | null;

  @Column('jsonb', { nullable: true })
  metadata?: Record<string, unknown> | null;

  @Column('text', { array: true, default: () => "'{}'" })
  tags!: string[];

  @CreateDateColumn({ name: 'created_at' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at' })
  updatedAt!: Date;
}
