import { ZodSchema } from 'zod';
import { ContractViolationError } from './contract-violation.error';

export function parseOrThrow<T>(schema: ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    throw new ContractViolationError(context, result.error.issues);
  }
  return result.data;
}
