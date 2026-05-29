import { Injectable, InternalServerErrorException, BadRequestException } from '@nestjs/common';
import type { TranscribeRequestInput } from '../contracts';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const FormData = require('form-data') as typeof import('form-data');

@Injectable()
export class VoiceService {
  private readonly openrouterApiKey = process.env.OPENROUTER_API_KEY;
  private readonly storageEndpoint = process.env.STORAGE_ENDPOINT ?? 'http://minio-microservice.statex-apps.svc.cluster.local:9000';
  private readonly storageBucket = process.env.STORAGE_BUCKET ?? 'school-committee';
  private readonly storageAccessKey = process.env.STORAGE_ACCESS_KEY;
  private readonly storageSecretKey = process.env.STORAGE_SECRET_KEY;

  async transcribe(dto: TranscribeRequestInput): Promise<{ transcript: string }> {
    const audioBuffer = await this.fetchFromStorage(dto.fileKey);
    const transcript = await this.callWhisper(audioBuffer, dto.fileKey, dto.language);
    return { transcript };
  }

  private async fetchFromStorage(fileKey: string): Promise<Buffer> {
    const url = `${this.storageEndpoint}/${this.storageBucket}/${fileKey}`;
    const headers: Record<string, string> = {};

    if (this.storageAccessKey && this.storageSecretKey) {
      const authHeader = 'Basic ' + Buffer.from(`${this.storageAccessKey}:${this.storageSecretKey}`).toString('base64');
      headers['Authorization'] = authHeader;
    }

    const res = await fetch(url, { headers, signal: AbortSignal.timeout(30_000) });
    if (!res.ok) {
      throw new BadRequestException(`Storage file not found: ${fileKey}`);
    }
    return Buffer.from(await res.arrayBuffer());
  }

  private async callWhisper(audio: Buffer, fileKey: string, language?: string): Promise<string> {
    if (!this.openrouterApiKey) {
      throw new InternalServerErrorException('OPENROUTER_API_KEY is not configured');
    }

    const ext = fileKey.split('.').pop() ?? 'webm';
    const form = new FormData();
    form.append('file', audio, { filename: `audio.${ext}`, contentType: `audio/${ext}` });
    form.append('model', 'openai/whisper-1');
    if (language) form.append('language', language);

    const res = await fetch('https://openrouter.ai/api/v1/audio/transcriptions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.openrouterApiKey}`,
        ...form.getHeaders(),
      },
      body: form.getBuffer(),
      signal: AbortSignal.timeout(60_000),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new InternalServerErrorException(`Whisper API error: ${errText}`);
    }

    const body = await res.json() as { text?: string };
    return body.text ?? '';
  }
}
