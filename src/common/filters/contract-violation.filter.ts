import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  Injectable,
  Logger,
} from '@nestjs/common';
import { Response } from 'express';
import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';
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

    // Fire-and-forget escalation — never suppress the HTTP response waiting for it
    this.fireEscalation(exception.context, issuesSummary).catch(() => {});

    response.status(500).json({
      error: 'contract_violation',
      context: exception.context,
      issues: exception.issues,
    });
  }

  private fireEscalation(context: string, issuesSummary: string): Promise<void> {
    return new Promise((resolve) => {
      const notifUrl = process.env.NOTIFICATION_SERVICE_URL;
      const chatId = process.env.TELEGRAM_CHAT_ID;
      if (!notifUrl || !chatId) { resolve(); return; }

      let parsed: URL;
      try { parsed = new URL(`${notifUrl}/notifications/send`); } catch { resolve(); return; }

      const body = JSON.stringify({
        channel: 'telegram',
        type: 'custom',
        recipient: chatId,
        subject: `Contract violation: ${context}`,
        message: `Contract violation detected at boundary "${context}".\n\nIssues:\n${issuesSummary}\n\nThe flow has been halted. Please investigate immediately.`,
        templateData: { level: 'critical' },
        service: 'ai-microservice',
      });

      const transport = parsed.protocol === 'https:' ? https : http;
      const req = transport.request(
        { hostname: parsed.hostname, port: parsed.port, path: parsed.pathname, method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
          timeout: 5000 },
        () => resolve(),
      );
      req.on('error', () => resolve());
      req.on('timeout', () => { req.destroy(); resolve(); });
      req.write(body);
      req.end();
    });
  }
}
