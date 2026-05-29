import { PipeTransform, BadRequestException } from '@nestjs/common';
import { ZodSchema, ZodError } from 'zod';

export class ZodValidationPipe implements PipeTransform {
  constructor(private readonly schema: ZodSchema) {}

  transform(value: unknown, metadata: any) {
    const result = this.schema.safeParse(value);
    if (!result.success) {
      throw new BadRequestException({
        error: 'Contract violation',
        details: result.error.issues.map((e) => ({
          path: e.path.join('.'),
          message: e.message,
        })),
      });
    }
    return result.data;
  }
}
