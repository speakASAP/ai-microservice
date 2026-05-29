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
export type ShopPresentationRequest = z.infer<typeof ShopPresentationRequestSchema>;
export type ShopComparePricesRequest = z.infer<typeof ShopComparePricesRequestSchema>;
export type ShopExtractLocationRequest = z.infer<typeof ShopExtractLocationRequestSchema>;

export const ShopTranscribeResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  transcript: z.string(),
}).passthrough();

export const ShopRefineQueryResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  query_text: z.string(),
  refined_params: z.record(z.string(), z.unknown()),
}).passthrough();

const ShopSearchItemSchema = z.object({
  title: z.string(),
  url: z.string(),
  price: z.unknown().optional(),
  source: z.string().optional(),
  position: z.number().int().nonnegative(),
  snippet: z.string().optional(),
}).passthrough();

export const ShopSearchResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  items: z.array(ShopSearchItemSchema),
}).passthrough();

export const ShopPresentationResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  formatted_content: z.string(),
}).passthrough();

export const ShopComparePricesResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  summary: z.string(),
}).passthrough();

export const ShopExtractLocationResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  region: z.string().nullable(),
  augmented_query: z.string().nullable(),
}).passthrough();

export type ShopTranscribeResponse = z.infer<typeof ShopTranscribeResponseSchema>;
export type ShopRefineQueryResponse = z.infer<typeof ShopRefineQueryResponseSchema>;
export type ShopSearchResponse = z.infer<typeof ShopSearchResponseSchema>;
export type ShopPresentationResponse = z.infer<typeof ShopPresentationResponseSchema>;
export type ShopComparePricesResponse = z.infer<typeof ShopComparePricesResponseSchema>;
export type ShopExtractLocationResponse = z.infer<typeof ShopExtractLocationResponseSchema>;
