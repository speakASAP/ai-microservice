import { Test, TestingModule } from '@nestjs/testing';
import { VoiceController } from '../../src/voice/voice.controller';
import { VoiceService } from '../../src/voice/voice.service';
import { TranscribeDto } from '../../src/voice/dto/transcribe.dto';

describe('VoiceController', () => {
  let controller: VoiceController;
  let mockService: { transcribe: jest.Mock };

  beforeEach(async () => {
    mockService = { transcribe: jest.fn() };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [VoiceController],
      providers: [{ provide: VoiceService, useValue: mockService }],
    }).compile();

    controller = module.get<VoiceController>(VoiceController);
  });

  describe('POST /voice/transcribe', () => {
    it('returns transcript for valid file key', async () => {
      const dto: TranscribeDto = { fileKey: 'uploads/voice-abc.webm', language: 'cs' };
      mockService.transcribe.mockResolvedValue({ transcript: 'Dobrý den' });

      const result = await controller.transcribe(dto);

      expect(result.transcript).toBe('Dobrý den');
      expect(mockService.transcribe).toHaveBeenCalledWith(dto);
    });

    it('returns transcript without optional language', async () => {
      const dto: TranscribeDto = { fileKey: 'uploads/voice-xyz.webm' };
      mockService.transcribe.mockResolvedValue({ transcript: 'Hello world' });

      const result = await controller.transcribe(dto);

      expect(result.transcript).toBe('Hello world');
    });

    it('propagates service errors', async () => {
      const dto: TranscribeDto = { fileKey: 'uploads/missing.webm' };
      mockService.transcribe.mockRejectedValue(new Error('File not found'));

      await expect(controller.transcribe(dto)).rejects.toThrow('File not found');
    });
  });
});
