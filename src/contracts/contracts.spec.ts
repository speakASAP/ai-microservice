import { AiCompleteRequestSchema, AiCompleteResponseSchema } from './ai-complete.contract';
import { ClaudeCodeJobStatusSchema, JobEnqueueResponseSchema, JobStatusResponseSchema } from './claude-code-job.contract';
import { TaskDraftRequestSchema, TaskDraftResponseSchema } from './task.contract';
import { TranscribeRequestSchema, TranscribeResponseSchema } from './voice.contract';
import { EmailDecideRequestSchema, EmailClassifyResponseSchema, EmailIngestRequestSchema } from './email-triage.contract';
import {
  ShopSearchRequestSchema,
  ShopTranscribeRequestSchema,
  ShopTranscribeResponseSchema,
  ShopRefineQueryResponseSchema,
  ShopSearchResponseSchema,
  ShopPresentationResponseSchema,
  ShopComparePricesResponseSchema,
  ShopExtractLocationResponseSchema,
} from './shop-assistant.contract';
import { HealthResponseSchema } from './http-responses.contract';
import { parseOrThrow } from './parse-or-throw';
import { ContractViolationError } from './contract-violation.error';

describe('AiCompleteRequestSchema', () => {
  it('accepts valid request', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'free', user_prompt: 'hello' });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });

  it('rejects unknown model_tier', () => {
    expect(AiCompleteRequestSchema.safeParse({ model_tier: 'turbo', user_prompt: 'hi' }).success).toBe(false);
  });

  it('rejects empty user_prompt', () => {
    expect(AiCompleteRequestSchema.safeParse({ model_tier: 'free', user_prompt: '' }).success).toBe(false);
  });

  it('accepts all valid tiers', () => {
    for (const tier of ['free', 'cheap', 'smart', 'premium']) {
      expect(AiCompleteRequestSchema.safeParse({ model_tier: tier, user_prompt: 'hi' }).success).toBe(true);
    }
  });

  it('schemaVersion defaults to 1.0 when omitted', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'free', user_prompt: 'hi' });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });

  it('rejects non-1.0 schemaVersion', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'free', user_prompt: 'hi', schemaVersion: '2.0' });
    expect(result.success).toBe(false);
  });
});

describe('AiCompleteResponseSchema', () => {
  it('accepts valid response', () => {
    const result = AiCompleteResponseSchema.safeParse({
      text: 'hi', model_used: 'sonnet', tier_used: 'smart', model_resolved: true,
    });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });

  it('passes through extra fields', () => {
    const result = AiCompleteResponseSchema.safeParse({
      text: 'hi', model_used: 'sonnet', tier_used: 'smart', model_resolved: true, extra_field: 'x',
    });
    expect(result.success).toBe(true);
    expect((result.data as any).extra_field).toBe('x');
  });

  // The response that broke cv-tuning: a tier name in model_used with nothing marking
  // it as unresolved. Consumers must be able to tell a served model from a stand-in.
  it('requires model_resolved so a tier name cannot pose as a model id', () => {
    const result = AiCompleteResponseSchema.safeParse({ text: 'hi', model_used: 'smart' });
    expect(result.success).toBe(false);
  });

  it('carries tier_used separately from model_used', () => {
    const result = AiCompleteResponseSchema.safeParse({
      text: 'hi', model_used: 'smart', tier_used: 'smart', model_resolved: false,
    });
    expect(result.success).toBe(true);
    expect(result.data!.model_resolved).toBe(false);
    expect(result.data!.tier_used).toBe('smart');
  });
});

describe('ClaudeCodeJobStatusSchema', () => {
  it('accepts all valid statuses', () => {
    for (const s of ['queued', 'executing', 'success', 'failed', 'timeout', 'retrying']) {
      expect(ClaudeCodeJobStatusSchema.safeParse(s).success).toBe(true);
    }
  });

  it('rejects unknown status', () => {
    expect(ClaudeCodeJobStatusSchema.safeParse('pending').success).toBe(false);
  });
});

describe('JobEnqueueResponseSchema', () => {
  it('accepts valid response', () => {
    const result = JobEnqueueResponseSchema.safeParse({
      jobId: '550e8400-e29b-41d4-a716-446655440000',
      taskId: '550e8400-e29b-41d4-a716-446655440001',
      status: 'queued',
      createdAt: new Date(),
    });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });

  it('rejects invalid jobId', () => {
    expect(JobEnqueueResponseSchema.safeParse({ jobId: 'not-a-uuid', taskId: '550e8400-e29b-41d4-a716-446655440001', status: 'queued', createdAt: new Date() }).success).toBe(false);
  });
});

describe('TaskDraftRequestSchema', () => {
  it('accepts valid request with transcript', () => {
    expect(TaskDraftRequestSchema.safeParse({ transcript: 'buy milk' }).success).toBe(true);
  });

  it('accepts optional fields', () => {
    expect(TaskDraftRequestSchema.safeParse({ textNote: 'some note', language: 'en' }).success).toBe(true);
  });
});

describe('TaskDraftResponseSchema', () => {
  it('accepts valid response', () => {
    const result = TaskDraftResponseSchema.safeParse({
      title: 'Buy milk',
      description: 'Go to store',
      priority: 'normal',
      modelTier: 'free',
    });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });

  it('rejects invalid priority', () => {
    expect(TaskDraftResponseSchema.safeParse({ title: 'x', description: 'y', priority: 'urgent', modelTier: 'free' }).success).toBe(false);
  });
});

describe('TranscribeRequestSchema', () => {
  it('accepts valid file key', () => {
    expect(TranscribeRequestSchema.safeParse({ fileKey: 'audio/file.mp3' }).success).toBe(true);
  });

  it('rejects empty fileKey', () => {
    expect(TranscribeRequestSchema.safeParse({ fileKey: '' }).success).toBe(false);
  });
});

describe('TranscribeResponseSchema', () => {
  it('accepts valid transcript', () => {
    const result = TranscribeResponseSchema.safeParse({ transcript: 'hello world' });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });
});

describe('EmailIngestRequestSchema', () => {
  it('accepts empty object (all optional)', () => {
    expect(EmailIngestRequestSchema.safeParse({}).success).toBe(true);
  });

  it('accepts full email payload', () => {
    expect(EmailIngestRequestSchema.safeParse({ message_id: 'abc', from: 'a@b.com', subject: 'Hi', body: 'Hello' }).success).toBe(true);
  });

  it('adds schemaVersion default', () => {
    const result = EmailIngestRequestSchema.safeParse({});
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });
});

describe('EmailDecideRequestSchema', () => {
  it('accepts valid request', () => {
    expect(EmailDecideRequestSchema.safeParse({ intent: 'support', confidence: 0.9 }).success).toBe(true);
  });

  it('accepts string confidence', () => {
    expect(EmailDecideRequestSchema.safeParse({ intent: 'support', confidence: '0.9' }).success).toBe(true);
  });

  it('rejects empty intent', () => {
    expect(EmailDecideRequestSchema.safeParse({ intent: '', confidence: 0.9 }).success).toBe(false);
  });

  it('rejects missing confidence', () => {
    expect(EmailDecideRequestSchema.safeParse({ intent: 'support' }).success).toBe(false);
  });
});

describe('ShopSearchRequestSchema', () => {
  it('accepts valid query', () => {
    expect(ShopSearchRequestSchema.safeParse({ query_text: 'blue shoes' }).success).toBe(true);
  });

  it('rejects empty query', () => {
    expect(ShopSearchRequestSchema.safeParse({ query_text: '' }).success).toBe(false);
  });

  it('accepts optional limit', () => {
    expect(ShopSearchRequestSchema.safeParse({ query_text: 'shoes', limit: 10 }).success).toBe(true);
  });
});

describe('ShopTranscribeRequestSchema', () => {
  it('accepts valid voice_file_url', () => {
    expect(ShopTranscribeRequestSchema.safeParse({ voice_file_url: 'https://cdn.example.com/audio.mp3' }).success).toBe(true);
  });

  it('rejects empty url', () => {
    expect(ShopTranscribeRequestSchema.safeParse({ voice_file_url: '' }).success).toBe(false);
  });
});

describe('parseOrThrow', () => {
  it('returns parsed data on success', () => {
    const result = parseOrThrow(ShopSearchRequestSchema, { query_text: 'test' }, 'ctx');
    expect(result.query_text).toBe('test');
  });

  it('throws ContractViolationError on failure', () => {
    expect(() => parseOrThrow(ShopSearchRequestSchema, { query_text: '' }, 'test.ctx')).toThrow(ContractViolationError);
  });

  it('ContractViolationError carries context string', () => {
    try {
      parseOrThrow(ShopSearchRequestSchema, {}, 'my.context');
      fail('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ContractViolationError);
      expect((e as ContractViolationError).context).toBe('my.context');
    }
  });

  it('ContractViolationError carries issues array', () => {
    try {
      parseOrThrow(AiCompleteRequestSchema, { model_tier: 'bad' }, 'ai.ctx');
    } catch (e) {
      expect((e as ContractViolationError).issues.length).toBeGreaterThan(0);
    }
  });

  it('error message includes context', () => {
    try {
      parseOrThrow(ShopSearchRequestSchema, {}, 'search.ctx');
    } catch (e) {
      expect((e as Error).message).toContain('search.ctx');
    }
  });
});

describe('ContractViolationError', () => {
  it('name is ContractViolationError', () => {
    const err = new ContractViolationError('ctx', []);
    expect(err.name).toBe('ContractViolationError');
    expect(err.context).toBe('ctx');
    expect(err.issues).toEqual([]);
  });

  it('is instanceof Error', () => {
    const err = new ContractViolationError('ctx', []);
    expect(err).toBeInstanceOf(Error);
  });
});

describe('ShopTranscribeResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopTranscribeResponseSchema.safeParse({ transcript: 'hello' }).success).toBe(true);
  });
  it('rejects missing transcript', () => {
    expect(ShopTranscribeResponseSchema.safeParse({}).success).toBe(false);
  });
  it('adds schemaVersion default', () => {
    const r = ShopTranscribeResponseSchema.safeParse({ transcript: 'x' });
    expect(r.success && r.data.schemaVersion).toBe('1.0');
  });
});

describe('ShopRefineQueryResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopRefineQueryResponseSchema.safeParse({ query_text: 'shoes', refined_params: {} }).success).toBe(true);
  });
  it('rejects missing query_text', () => {
    expect(ShopRefineQueryResponseSchema.safeParse({ refined_params: {} }).success).toBe(false);
  });
});

describe('ShopSearchResponseSchema', () => {
  it('accepts valid items array', () => {
    expect(ShopSearchResponseSchema.safeParse({
      items: [{ title: 'Shoe', url: 'http://x.com', position: 1 }],
    }).success).toBe(true);
  });
  it('accepts empty items array', () => {
    expect(ShopSearchResponseSchema.safeParse({ items: [] }).success).toBe(true);
  });
  it('rejects item missing required position', () => {
    expect(ShopSearchResponseSchema.safeParse({
      items: [{ title: 'x', url: 'http://x.com' }],
    }).success).toBe(false);
  });
});

describe('ShopPresentationResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopPresentationResponseSchema.safeParse({ formatted_content: 'Here are results...' }).success).toBe(true);
  });
  it('rejects missing formatted_content', () => {
    expect(ShopPresentationResponseSchema.safeParse({}).success).toBe(false);
  });
});

describe('ShopComparePricesResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopComparePricesResponseSchema.safeParse({ summary: 'Best pick is X' }).success).toBe(true);
  });
  it('rejects missing summary', () => {
    expect(ShopComparePricesResponseSchema.safeParse({}).success).toBe(false);
  });
});

describe('ShopExtractLocationResponseSchema', () => {
  it('accepts valid response with region', () => {
    expect(ShopExtractLocationResponseSchema.safeParse({ region: 'Czech Republic', augmented_query: 'shoes Czech Republic' }).success).toBe(true);
  });
  it('accepts null values', () => {
    expect(ShopExtractLocationResponseSchema.safeParse({ region: null, augmented_query: null }).success).toBe(true);
  });
  it('rejects missing region field', () => {
    expect(ShopExtractLocationResponseSchema.safeParse({ augmented_query: null }).success).toBe(false);
  });
});

describe('HealthResponseSchema', () => {
  it('accepts valid health response', () => {
    const result = HealthResponseSchema.safeParse({ status: 'ok', service: 'ai-microservice' });
    expect(result.success).toBe(true);
  });

  it('rejects invalid status value', () => {
    const result = HealthResponseSchema.safeParse({ status: 'running', service: 'ai-microservice' });
    expect(result.success).toBe(false);
  });

  it('rejects missing service field', () => {
    const result = HealthResponseSchema.safeParse({ status: 'ok' });
    expect(result.success).toBe(false);
  });
});
