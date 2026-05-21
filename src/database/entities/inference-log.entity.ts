import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, Index } from 'typeorm';

@Entity('inference_logs')
@Index('idx_inference_service_id', ['serviceId'])
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

  @Column('integer', { nullable: true })
  durationMs?: number;

  @Column('integer', { nullable: true })
  statusCode?: number;

  @CreateDateColumn()
  createdAt!: Date;
}
