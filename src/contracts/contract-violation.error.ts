import { ZodIssue } from 'zod';

export class ContractViolationError extends Error {
  constructor(
    readonly context: string,
    readonly issues: ZodIssue[],
  ) {
    super(`contract_violation:${context}`);
    this.name = 'ContractViolationError';
  }
}
