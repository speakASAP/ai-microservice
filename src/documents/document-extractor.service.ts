import { BadRequestException, Inject, Injectable, Logger, Optional } from '@nestjs/common';
import { spawn } from 'child_process';
import { mkdtemp, readdir, rm, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  DocumentEngine,
  ExtractDocumentRequest,
  ExtractDocumentResponse,
  MAX_DOCUMENT_BYTES,
} from '../contracts';

export const PDF_PARSER = 'AI_DOC_PDF_PARSER';
export const DOCX_PARSER = 'AI_DOC_DOCX_PARSER';
export const COMMAND_RUNNER = 'AI_DOC_COMMAND_RUNNER';

export type PdfParser = (buffer: Buffer) => Promise<{ text: string; numpages?: number }>;
export type DocxParser = (input: { buffer: Buffer }) => Promise<{ value: string }>;
export type CommandRunner = (command: string, args: string[]) => Promise<string>;

export const PDF_MIME = 'application/pdf';
export const DOCX_MIME = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
const TEXT_MIMES = ['text/plain', 'text/markdown', 'text/csv'];
const IMAGE_MIMES = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/webp', 'image/bmp'];

const DEFAULT_OCR_LANGUAGE = process.env.OCR_DEFAULT_LANGUAGE ?? 'eng';
/** Rasterisation density. Below ~300dpi tesseract accuracy on body text drops sharply. */
const OCR_DPI = '300';
const OCR_TIMEOUT_MS = 120_000;
/** A scan of a long document is a cost, not a feature; refuse rather than run for minutes. */
const MAX_OCR_PAGES = 30;

@Injectable()
export class DocumentExtractorService {
  private readonly logger = new Logger(DocumentExtractorService.name);

  constructor(
    @Optional() @Inject(PDF_PARSER) private readonly parsePdf?: PdfParser,
    @Optional() @Inject(DOCX_PARSER) private readonly parseDocx?: DocxParser,
    @Optional() @Inject(COMMAND_RUNNER) private readonly run: CommandRunner = defaultRunner,
  ) {}

  static isSupported(mimeType: string): boolean {
    return (
      mimeType === PDF_MIME ||
      mimeType === DOCX_MIME ||
      TEXT_MIMES.includes(mimeType) ||
      IMAGE_MIMES.includes(mimeType)
    );
  }

  async extract(dto: ExtractDocumentRequest): Promise<ExtractDocumentResponse> {
    const buffer = this.decode(dto.contentBase64);
    const mimeType = dto.mimeType.split(';')[0].trim().toLowerCase();

    if (!DocumentExtractorService.isSupported(mimeType)) {
      throw new BadRequestException(
        `unsupported document type ${mimeType}. Supported: PDF, DOCX, plain text, and PNG/JPEG/TIFF images.`,
      );
    }

    const language = dto.ocrLanguage ?? DEFAULT_OCR_LANGUAGE;
    const started = Date.now();
    const result = await this.extractByType(buffer, mimeType, dto, language);

    if (result.text.trim().length === 0) {
      // A scan parses "successfully" to an empty string. Returning it would let a caller
      // replace a real document with nothing and never learn that it did.
      throw new BadRequestException(
        'no text could be extracted from this document. If it is a scan or a photo, ' +
          (dto.allowOcr
            ? 'the image was too unclear to recognise - try a higher-resolution copy.'
            : 'retry with allowOcr enabled.'),
      );
    }

    this.logger.log(
      `extracted ${dto.filename} type=${mimeType} engine=${result.engine} ` +
        `ocr=${result.ocrUsed} pages=${result.pages} chars=${result.text.length} in ${Date.now() - started}ms`,
    );

    return {
      schemaVersion: '1.0',
      text: result.text,
      engine: result.engine,
      ocrUsed: result.ocrUsed,
      pages: result.pages,
      chars: result.text.length,
    };
  }

  private async extractByType(
    buffer: Buffer,
    mimeType: string,
    dto: ExtractDocumentRequest,
    language: string,
  ): Promise<{ text: string; engine: DocumentEngine; ocrUsed: boolean; pages: number }> {
    if (TEXT_MIMES.includes(mimeType)) {
      return { text: buffer.toString('utf8'), engine: 'plain-text', ocrUsed: false, pages: 0 };
    }

    if (mimeType === DOCX_MIME) {
      const parser = this.parseDocx ?? (await defaultDocxParser());
      const value = await this.guard('DOCX', () => parser({ buffer }));
      return { text: value.value, engine: 'docx', ocrUsed: false, pages: 0 };
    }

    if (IMAGE_MIMES.includes(mimeType)) {
      if (!dto.allowOcr) {
        throw new BadRequestException(
          'this document is an image and can only be read with OCR, which the caller disabled',
        );
      }
      const text = await this.ocrImageBuffer(buffer, mimeType, language);
      return { text, engine: 'ocr', ocrUsed: true, pages: 1 };
    }

    const parser = this.parsePdf ?? (await defaultPdfParser());
    const parsed = await this.guard('PDF', () => parser(buffer));

    // A born-digital PDF parses exactly; only a scan comes back empty, and that is the
    // single case worth paying for OCR.
    if (parsed.text.trim().length > 0) {
      return { text: parsed.text, engine: 'pdf-text', ocrUsed: false, pages: parsed.numpages ?? 0 };
    }

    if (!dto.allowOcr) {
      throw new BadRequestException(
        'this PDF contains no text layer and can only be read with OCR, which the caller disabled',
      );
    }

    const ocr = await this.ocrPdf(buffer, language);
    return { text: ocr.text, engine: 'ocr', ocrUsed: true, pages: ocr.pages };
  }

  /** Rasterises the PDF and recognises each page. Both tools are local; nothing leaves the pod. */
  private async ocrPdf(buffer: Buffer, language: string): Promise<{ text: string; pages: number }> {
    const dir = await mkdtemp(join(tmpdir(), 'ai-doc-ocr-'));
    try {
      const pdfPath = join(dir, 'input.pdf');
      await writeFile(pdfPath, buffer);
      await this.runTool('pdftoppm', ['-r', OCR_DPI, '-png', pdfPath, join(dir, 'page')]);

      const pages = (await readdir(dir))
        .filter((name) => name.startsWith('page') && name.endsWith('.png'))
        .sort();
      if (pages.length === 0) {
        throw new BadRequestException('the PDF could not be rasterised for OCR');
      }
      if (pages.length > MAX_OCR_PAGES) {
        throw new BadRequestException(
          `this document has ${pages.length} scanned pages; the limit is ${MAX_OCR_PAGES}. Split it and retry.`,
        );
      }

      const texts: string[] = [];
      for (const page of pages) {
        texts.push(await this.runTool('tesseract', [join(dir, page), 'stdout', '-l', language]));
      }

      return { text: texts.join('\n\n').trim(), pages: pages.length };
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  private async ocrImageBuffer(buffer: Buffer, mimeType: string, language: string): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), 'ai-doc-ocr-'));
    try {
      const path = join(dir, `input.${mimeType.split('/')[1]}`);
      await writeFile(path, buffer);
      return (await this.runTool('tesseract', [path, 'stdout', '-l', language])).trim();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  private async runTool(command: string, args: string[]): Promise<string> {
    try {
      return await this.run(command, args);
    } catch (cause) {
      if (cause instanceof BadRequestException) throw cause;
      const message = cause instanceof Error ? cause.message : String(cause);
      if (message.includes('ENOENT')) {
        // A missing binary is a deployment fault, not a bad document. Saying so stops it
        // being investigated as a corrupt upload.
        this.logger.error(`${command} is not installed in this image; OCR cannot run`);
        throw new BadRequestException(`OCR is unavailable: ${command} is not installed in this deployment`);
      }
      this.logger.error(`${command} failed: ${message.slice(0, 300)}`);
      throw new BadRequestException(`OCR failed while running ${command}: ${message.slice(0, 200)}`);
    }
  }

  private async guard<T>(label: string, work: () => Promise<T>): Promise<T> {
    try {
      return await work();
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      this.logger.error(`${label} parse failed: ${message.slice(0, 300)}`);
      throw new BadRequestException(`could not read the ${label}: ${message.slice(0, 200)}`);
    }
  }

  private decode(contentBase64: string): Buffer {
    const buffer = Buffer.from(contentBase64, 'base64');
    if (buffer.length === 0) {
      throw new BadRequestException('the document is empty');
    }
    if (buffer.length > MAX_DOCUMENT_BYTES) {
      throw new BadRequestException(
        `the document is ${buffer.length} bytes; the limit is ${MAX_DOCUMENT_BYTES}`,
      );
    }
    return buffer;
  }
}

async function defaultPdfParser(): Promise<PdfParser> {
  const mod = await import('pdf-parse');
  return (mod as unknown as { default?: PdfParser }).default ?? (mod as unknown as PdfParser);
}

async function defaultDocxParser(): Promise<DocxParser> {
  const mod = await import('mammoth');
  return (mod as unknown as { extractRawText: DocxParser }).extractRawText;
}

/** Spawns without a shell: document filenames and language codes never become shell syntax. */
const defaultRunner: CommandRunner = (command, args) =>
  new Promise<string>((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`${command} timed out after ${OCR_TIMEOUT_MS}ms`));
    }, OCR_TIMEOUT_MS);

    child.stdout.on('data', (chunk) => (stdout += chunk.toString()));
    child.stderr.on('data', (chunk) => (stderr += chunk.toString()));
    child.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(stdout);
      else reject(new Error(`${command} exited with ${code}: ${stderr.slice(0, 200)}`));
    });
  });
