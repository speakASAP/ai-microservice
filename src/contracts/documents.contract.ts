import { z } from 'zod';

/**
 * How the text was obtained. Callers persist this: a fact graph built from `ocr` carries
 * recognition risk that one built from `pdf-text` does not, and collapsing the two would
 * hide which documents were read by a recogniser rather than parsed exactly.
 */
export const DOCUMENT_ENGINES = ['plain-text', 'pdf-text', 'docx', 'ocr'] as const;
export type DocumentEngine = (typeof DOCUMENT_ENGINES)[number];

/** Decoded size ceiling. Large enough for a scanned multi-page CV, small enough that one request cannot exhaust the pod. */
export const MAX_DOCUMENT_BYTES = 20 * 1024 * 1024;

export const ExtractDocumentRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  filename: z.string().min(1),
  mimeType: z.string().min(1),
  contentBase64: z.string().min(1),
  /** Tesseract language codes, e.g. `eng` or `eng+ces`. Only consulted when OCR runs. */
  ocrLanguage: z
    .string()
    .regex(/^[a-z]{3}(?:\+[a-z]{3})*$/, 'expected tesseract language codes such as "eng" or "eng+ces"')
    .optional(),
  /**
   * Set false to make a scan fail loudly instead of being recognised. A caller that needs
   * exact text (a contract, an invoice total) must not silently receive OCR output.
   */
  allowOcr: z.boolean().default(true),
});

export const ExtractDocumentResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  text: z.string(),
  engine: z.enum(DOCUMENT_ENGINES),
  ocrUsed: z.boolean(),
  /** Pages parsed or recognised. 0 when the format has no page structure. */
  pages: z.number().int().nonnegative(),
  chars: z.number().int().nonnegative(),
});

export type ExtractDocumentRequest = z.infer<typeof ExtractDocumentRequestSchema>;
export type ExtractDocumentRequestInput = z.input<typeof ExtractDocumentRequestSchema>;
export type ExtractDocumentResponse = z.infer<typeof ExtractDocumentResponseSchema>;
