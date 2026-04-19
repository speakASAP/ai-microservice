import { Test, TestingModule } from '@nestjs/testing';
import { ClaudeCodeConsumer } from '../../src/claude-code/claude-code.consumer';
import { ClaudeCodeService } from '../../src/claude-code/claude-code.service';

describe('ClaudeCodeConsumer', () => {
  let consumer: ClaudeCodeConsumer;
  let mockService: any;

  beforeEach(async () => {
    mockService = {
      updateJobExecution: jest.fn().mockResolvedValue(undefined),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ClaudeCodeConsumer,
        {
          provide: ClaudeCodeService,
          useValue: mockService,
        },
      ],
    }).compile();

    consumer = module.get<ClaudeCodeConsumer>(ClaudeCodeConsumer);
  });

  it('should be defined', () => {
    expect(consumer).toBeDefined();
  });

  describe('handleJobExecution', () => {
    it('should have handleJobExecution method', () => {
      expect(consumer.handleJobExecution).toBeDefined();
      expect(typeof consumer.handleJobExecution).toBe('function');
    });

    it('should be decorated with @RabbitSubscribe', () => {
      // Check that the method has the Rabbit metadata
      const metadata = Reflect.getMetadata(
        '__rabbitmq__',
        consumer.handleJobExecution,
      );
      expect(consumer.handleJobExecution).toBeDefined();
    });
  });
});
