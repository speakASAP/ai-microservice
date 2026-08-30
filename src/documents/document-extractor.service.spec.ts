import { BadRequestException } from '@nestjs/common';
import { DocumentExtractorService, DOCX_MIME, PDF_MIME } from './document-extractor.service';
import { ExtractDocumentRequestSchema } from '../contracts';

const request = (overrides: Record<string, unknown> = {}) =>
  ExtractDocumentRequestSchema.parse({
    filename: 'cv.pdf',
    mimeType: PDF_MIME,
    contentBase64: Buffer.from('%PDF-1.4 fake').toString('base64'),
    ...overrides,
  });

describe('DocumentExtractorService', () => {
  const pdfWithText = jest.fn(async () => ({ text: 'Jane Doe\nSenior Developer', numpages: 2 }));
  const scannedPdf = jest.fn(async () => ({ text: '   ', numpages: 1 }));
  const docx = jest.fn(async () => ({ value: 'Docx body text' }));

  beforeEach(() => jest.clearAllMocks());

  it('reads a born-digital PDF without paying for OCR', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());

    const result = await service.extract(request());

    expect(result.text).toContain('Senior Developer');
    expect(result.engine).toBe('pdf-text');
    expect(result.ocrUsed).toBe(false);
    expect(result.pages).toBe(2);
  });

  it('reports the character count so a caller can detect a near-empty read', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());

    const result = await service.extract(request());

    expect(result.chars).toBe(result.text.length);
  });

  it('falls back to OCR only when the PDF has no text layer', async () => {
    const run = jest.fn(async (command: string) =>
      command === 'tesseract' ? 'Recognised scan text' : '',
    );
    const service = new DocumentExtractorService(scannedPdf, docx, run);
    jest.spyOn(service as never, 'ocrPdf' as never).mockResolvedValue({
      text: 'Recognised scan text',
      pages: 1,
    } as never);

    const result = await service.extract(request());

    expect(result.engine).toBe('ocr');
    expect(result.ocrUsed).toBe(true);
    expect(result.text).toBe('Recognised scan text');
  });

  it('refuses to OCR when the caller demands exact text', async () => {
    const service = new DocumentExtractorService(scannedPdf, docx, jest.fn());

    await expect(service.extract(request({ allowOcr: false }))).rejects.toBeInstanceOf(
      BadRequestException,
    );
  });

  it('reads DOCX without OCR', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());

    const result = await service.extract(request({ mimeType: DOCX_MIME, filename: 'cv.docx' }));

    expect(result.engine).toBe('docx');
    expect(result.text).toBe('Docx body text');
  });

  it('returns plain text verbatim', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());

    const result = await service.extract(
      request({
        mimeType: 'text/plain',
        filename: 'cv.txt',
        contentBase64: Buffer.from('# CV\n\nline').toString('base64'),
      }),
    );

    expect(result.text).toBe('# CV\n\nline');
    expect(result.engine).toBe('plain-text');
  });

  it('rejects an unsupported type by name so the caller can tell the user what to send', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());

    await expect(
      service.extract(request({ mimeType: 'application/x-msdownload', filename: 'cv.exe' })),
    ).rejects.toThrow(/application\/x-msdownload/);
  });

  it('rejects an empty payload at the contract boundary', () => {
    expect(
      ExtractDocumentRequestSchema.safeParse({
        filename: 'cv.pdf',
        mimeType: PDF_MIME,
        contentBase64: '',
      }).success,
    ).toBe(false);
  });

  it('rejects base64 that decodes to no bytes rather than returning empty text', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());

    // Valid base64, zero bytes. Without this check it would parse "successfully" to ''.
    await expect(service.extract(request({ contentBase64: '====' }))).rejects.toThrow(/empty/);
    expect(pdfWithText).not.toHaveBeenCalled();
  });

  it('reports a missing OCR binary as a deployment fault, not a bad document', async () => {
    const run = jest.fn(async () => {
      throw new Error('spawn tesseract ENOENT');
    });
    const service = new DocumentExtractorService(pdfWithText, docx, run);

    await expect(
      service.extract(
        request({
          mimeType: 'image/png',
          filename: 'scan.png',
          contentBase64: Buffer.from('\x89PNG').toString('base64'),
        }),
      ),
    ).rejects.toThrow(/not installed in this deployment/);
  });

  it('never lets a document filename reach a shell', async () => {
    const run = jest.fn(async () => 'ok');
    const service = new DocumentExtractorService(pdfWithText, docx, run);

    await service.extract(
      request({
        mimeType: 'image/png',
        filename: 'cv; rm -rf /.png',
        contentBase64: Buffer.from('\x89PNG').toString('base64'),
      }),
    );

    // The path is built from a generated temp dir, never from the caller-supplied name.
    const [, args] = run.mock.calls[0] as unknown as [string, string[]];
    expect(args[0]).not.toContain('rm -rf');
  });

  it('rejects a document above the size ceiling before any parsing happens', async () => {
    const service = new DocumentExtractorService(pdfWithText, docx, jest.fn());
    const oversized = Buffer.alloc(21 * 1024 * 1024).toString('base64');

    await expect(service.extract(request({ contentBase64: oversized }))).rejects.toThrow(/limit is/);
    expect(pdfWithText).not.toHaveBeenCalled();
  });
});
