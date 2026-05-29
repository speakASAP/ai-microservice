import { ZodValidationPipe } from '../../src/contracts/zod-validation.pipe';
import { z } from 'zod';
import { BadRequestException } from '@nestjs/common';

describe('ZodValidationPipe', () => {
  const schema = z.object({ name: z.string() });
  const pipe = new ZodValidationPipe(schema);

  it('passes valid input through', () => {
    expect(pipe.transform({ name: 'Alice' }, {} as any)).toEqual({ name: 'Alice' });
  });

  it('throws BadRequestException on invalid input', () => {
    expect(() => pipe.transform({ name: 123 }, {} as any)).toThrow(BadRequestException);
  });

  it('throws BadRequestException on missing required field', () => {
    expect(() => pipe.transform({}, {} as any)).toThrow(BadRequestException);
  });
});
