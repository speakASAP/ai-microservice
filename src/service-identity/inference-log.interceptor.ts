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
  body?: {
    model_tier?: string;
    business_id?: string;
    businessId?: string;
  };
}

interface ServiceResponse {
  statusCode: number;
}

interface InferenceResponseBody {
  model_used?: string;
  inputTokens?: number;
  outputTokens?: number;
  token_usage_estimate?: number;
  estimated_cost_usd?: string | number;
  estimatedCostUsd?: string | number;
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
        next: (responseBody) => this.persist(request, context, Date.now() - start, responseBody),
        error: () => this.persist(request, context, Date.now() - start),
      }),
    );
  }

  private persist(
    request: ServiceRequest,
    context: ExecutionContext,
    durationMs: number,
    responseBody?: unknown,
  ): void {
    const serviceId = request.serviceId ?? 'anonymous';
    const endpoint = `${request.method} ${request.path}`;
    const response = context.switchToHttp().getResponse<ServiceResponse>();
    const inferenceResponse = this.asInferenceResponse(responseBody);

    this.logRepo
      .save({
        serviceId,
        endpoint,
        modelTier: request.body?.model_tier,
        businessId: request.body?.business_id ?? request.body?.businessId,
        modelUsed: inferenceResponse?.model_used,
        inputTokens: inferenceResponse?.inputTokens,
        outputTokens: inferenceResponse?.outputTokens,
        tokenUsageEstimate: inferenceResponse?.token_usage_estimate,
        estimatedCostUsd: this.normalizeCost(
          inferenceResponse?.estimated_cost_usd ?? inferenceResponse?.estimatedCostUsd,
        ),
        durationMs,
        statusCode: response.statusCode,
      })
      .catch(() => {
        // Fire-and-forget — never block the response on logging
      });
  }

  private asInferenceResponse(responseBody: unknown): InferenceResponseBody | undefined {
    if (responseBody === null || typeof responseBody !== 'object' || Array.isArray(responseBody)) {
      return undefined;
    }
    return responseBody as InferenceResponseBody;
  }

  private normalizeCost(cost: string | number | undefined): string | undefined {
    if (typeof cost === 'number' && Number.isFinite(cost)) {
      return cost.toFixed(8);
    }
    if (typeof cost === 'string' && cost.trim()) {
      return cost;
    }
    return undefined;
  }
}
