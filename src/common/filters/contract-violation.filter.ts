import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  Injectable,
  Logger,
} from '@nestjs/common';
import { Response } from 'express';
import { ContractViolationError } from '../../contracts/contract-violation.error';

@Catch(ContractViolationError)
@Injectable()
export class ContractViolationFilter implements ExceptionFilter {
  private readonly logger = new Logger(ContractViolationFilter.name);

  catch(exception: ContractViolationError, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();

    const issuesSummary = exception.issues
      .map((i) => `[${i.path.join('.')}] ${i.message}`)
      .join('; ');

    this.logger.error(
      `Contract violation at "${exception.context}": ${issuesSummary}`,
    );

    response.status(500).json({
      error: 'contract_violation',
      context: exception.context,
      issues: exception.issues,
    });
  }
}
