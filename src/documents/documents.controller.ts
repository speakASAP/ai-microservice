import { Body, Controller, HttpCode, Post, UsePipes } from '@nestjs/common';
import { DocumentExtractorService } from './document-extractor.service';
import {
  ExtractDocumentRequestSchema,
  ExtractDocumentResponseSchema,
  ZodValidationPipe,
  parseOrThrow,
} from '../contracts';
import type { ExtractDocumentRequestInput, ExtractDocumentResponse } from '../contracts';

/**
 * Shared document reading for the whole ecosystem.
 *
 * Deliberately NOT a model call: PDF, DOCX and plain text are parsed exactly, and OCR runs
 * locally through tesseract. Every service that needs text out of a file uses this instead
 * of carrying its own parser stack and its own OCR system packages.
 */
@Controller('documents')
export class DocumentsController {
  constructor(private readonly extractor: DocumentExtractorService) {}

  @Post('extract')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(ExtractDocumentRequestSchema))
  async extract(@Body() dto: ExtractDocumentRequestInput): Promise<ExtractDocumentResponse> {
    const result = await this.extractor.extract(ExtractDocumentRequestSchema.parse(dto));
    return parseOrThrow(ExtractDocumentResponseSchema, result, 'documents.extract.response');
  }
}
