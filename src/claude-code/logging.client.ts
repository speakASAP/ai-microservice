import { Injectable, Optional } from "@nestjs/common";
import { Inject } from "@nestjs/common";

export const LOGGING_URL_TOKEN = 'LOGGING_URL';
export const LOGGING_FETCH_TOKEN = 'LOGGING_FETCH';

export type LogLevel = "info" | "warn" | "error" | "debug";

@Injectable()
export class LoggingClient {
  private readonly url: string;
  private readonly fetchFn: typeof fetch;

  constructor(
    @Optional() @Inject(LOGGING_URL_TOKEN) url?: string,
    @Optional() @Inject(LOGGING_FETCH_TOKEN) fetchFn?: typeof fetch,
  ) {
    this.url =
      url ??
      process.env.LOGGING_SERVICE_URL ??
      "http://logging-microservice:3367";
    this.fetchFn = fetchFn ?? fetch;
  }

  async log(
    level: LogLevel,
    message: string,
    metadata: Record<string, unknown> = {},
  ): Promise<void> {
    try {
      const response = await this.fetchFn(`${this.url}/api/logs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          service: "ai-microservice",
          level,
          message,
          timestamp: new Date().toISOString(),
          metadata,
        }),
        signal: AbortSignal.timeout(3000),
      });
      if (!response.ok) {
        // Silent — logging must never crash the service
      }
    } catch {
      // Silent — logging must never crash the service
    }
  }
}
