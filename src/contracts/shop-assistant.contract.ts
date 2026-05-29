import { z } from 'zod';

export const ShopTranscribeRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  voice_file_url: z.string().min(1),
});

export const ShopRefineQueryRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  user_text: z.string().min(1),
  previous_params: z.record(z.string(), z.unknown()).optional(),
  role: z.string().optional(),
  prompt_content: z.string().optional(),
  model: z.string().optional(),
});

export const ShopSearchRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  query_text: z.string().min(1),
  limit: z.number().int().positive().optional(),
});

export const ShopPresentationRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  results: z.array(z.record(z.string(), z.unknown())),
  query_text: z.string().min(1),
  role: z.string().optional(),
  prompt_content: z.string().optional(),
  model: z.string().optional(),
});

export const ShopComparePricesRequestSchema = ShopPresentationRequestSchema.extend({
  priority_order: z.array(z.string()).optional(),
});

export const ShopExtractLocationRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  user_text: z.string().min(1),
  query_text: z.string().min(1),
  role: z.string().optional(),
  prompt_content: z.string().optional(),
  model: z.string().optional(),
  priority_order: z.array(z.string()).optional(),
});

export type ShopTranscribeRequest = z.infer<typeof ShopTranscribeRequestSchema>;
export type ShopRefineQueryRequest = z.infer<typeof ShopRefineQueryRequestSchema>;
export type ShopSearchRequest = z.infer<typeof ShopSearchRequestSchema>;
