import { LoggingClient, LogLevel } from "../../src/claude-code/logging.client";

describe("LoggingClient", () => {
  let client: LoggingClient;
  let mockFetch: jest.Mock;

  beforeEach(() => {
    mockFetch = jest.fn().mockResolvedValue({ ok: true });
    client = new LoggingClient(
      "http://logging-microservice:3367",
      mockFetch as any,
    );
  });

  it("posts structured log entry to /api/logs", async () => {
    await client.log("info", "Test event", { jobId: "job-123" });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://logging-microservice:3367/api/logs",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: expect.stringContaining('"message":"Test event"'),
      }),
    );

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.service).toBe("ai-microservice");
    expect(body.level).toBe("info");
    expect(body.message).toBe("Test event");
    expect(body.metadata).toEqual({ jobId: "job-123" });
    expect(body.timestamp).toBeDefined();
  });

  it("silently swallows network errors without throwing", async () => {
    mockFetch.mockRejectedValue(new Error("Connection refused"));
    await expect(client.log("error", "Failed", {})).resolves.not.toThrow();
  });

  it("silently swallows non-ok HTTP responses without throwing", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 503 });
    await expect(client.log("warn", "Warning", {})).resolves.not.toThrow();
  });

  it("uses default URL from env when not provided", () => {
    const defaultClient = new LoggingClient();
    expect(defaultClient).toBeDefined();
  });
});
