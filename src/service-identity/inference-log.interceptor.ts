import {
  CallHandler,
  ExecutionContext,
  Injectable,
  NestInterceptor,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Observable, tap } from 'rxjs';
import { InferenceLog } from '../database/entities/inference-log.entity';

interface ServiceRequest {
  serviceId?: string;
  method: string;
  path: string;
}

interface ServiceResponse {
  statusCode: number;
}

@Injectable()
export class InferenceLogInterceptor implements NestInterceptor {
  constructor(
    @InjectRepository(InferenceLog)
    private readonly logRepo: Repository<InferenceLog>,
  ) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<unknown> {
    const request = context.switchToHttp().getRequest<ServiceRequest>();
    const start = Date.now();

    return next.handle().pipe(
      tap({
        next: () => this.persist(request, context, Date.now() - start),
        error: () => this.persist(request, context, Date.now() - start),
      }),
    );
  }

  private persist(
    request: ServiceRequest,
    context: ExecutionContext,
    durationMs: number,
  ): void {
    const serviceId = request.serviceId ?? 'anonymous';
    const endpoint = `${request.method} ${request.path}`;
    const response = context.switchToHttp().getResponse<ServiceResponse>();

    this.logRepo
      .save({
        serviceId,
        endpoint,
        durationMs,
        statusCode: response.statusCode,
      })
      .catch(() => {
        // Fire-and-forget — never block the response on logging
      });
  }
}
