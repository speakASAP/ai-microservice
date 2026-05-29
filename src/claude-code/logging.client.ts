import { Injectable } from "@nestjs/common";

export type LogLevel = "info" | "warn" | "error" | "debug";

@Injectable()
export class LoggingClient {
  private readonly url: string;
  private readonly fetchFn: typeof fetch;

  constructor(
    url?: string,
    fetchFn?: typeof fetch,
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
