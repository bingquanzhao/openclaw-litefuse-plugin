import http from "node:http";
import https from "node:https";
import { ExportResultCode } from "@opentelemetry/core";
import type { ExportResult } from "@opentelemetry/core";
import type { SpanExporter, ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { JsonTraceSerializer } from "@opentelemetry/otlp-transformer";

// Drop-in OTLP/HTTP+JSON SpanExporter that bypasses globalThis.fetch.
// Mirrors the rationale from the v3 directFetch: enterprise installs use
// global-agent to monkey-patch fetch with proxy routing; that breaks
// uploads to self-hosted Litefuse on a private network. node:http.request
// honors NO_PROXY-style env routing without intercept.

const httpAgent = new http.Agent({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

export interface DirectOtlpExporterOptions {
  url: string;
  headers: Record<string, string>;
  timeoutMillis?: number;
}

export class DirectOtlpHttpExporter implements SpanExporter {
  private readonly url: URL;
  private readonly headers: Record<string, string>;
  private readonly timeoutMillis: number;
  private shutdownOnce: Promise<void> | undefined;

  constructor(opts: DirectOtlpExporterOptions) {
    this.url = new URL(opts.url);
    this.headers = {
      "Content-Type": "application/json",
      ...opts.headers,
    };
    this.timeoutMillis = opts.timeoutMillis ?? 30000;
  }

  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    if (this.shutdownOnce) {
      resultCallback({ code: ExportResultCode.FAILED, error: new Error("Exporter shut down") });
      return;
    }
    if (spans.length === 0) {
      resultCallback({ code: ExportResultCode.SUCCESS });
      return;
    }

    let body: Uint8Array | undefined;
    try {
      body = JsonTraceSerializer.serializeRequest(spans);
    } catch (err) {
      resultCallback({ code: ExportResultCode.FAILED, error: err as Error });
      return;
    }
    if (!body) {
      resultCallback({ code: ExportResultCode.SUCCESS });
      return;
    }

    const isHttps = this.url.protocol === "https:";
    const lib = isHttps ? https : http;
    const agent = isHttps ? httpsAgent : httpAgent;

    const req = lib.request(
      this.url,
      {
        method: "POST",
        headers: {
          ...this.headers,
          "Content-Length": Buffer.byteLength(body as Uint8Array).toString(),
        },
        agent,
        timeout: this.timeoutMillis,
      },
      (res) => {
        // Drain so the socket can be reused via keepAlive
        res.resume();
        res.on("end", () => {
          const status = res.statusCode ?? 0;
          if (status >= 200 && status < 300) {
            resultCallback({ code: ExportResultCode.SUCCESS });
          } else {
            resultCallback({
              code: ExportResultCode.FAILED,
              error: new Error(`OTLP export failed: HTTP ${status}`),
            });
          }
        });
      }
    );

    req.on("error", (err) => {
      resultCallback({ code: ExportResultCode.FAILED, error: err });
    });
    req.on("timeout", () => {
      req.destroy(new Error(`OTLP export timeout after ${this.timeoutMillis}ms`));
    });

    req.write(body);
    req.end();
  }

  async shutdown(): Promise<void> {
    if (!this.shutdownOnce) {
      this.shutdownOnce = Promise.resolve().then(() => {
        httpAgent.destroy();
        httpsAgent.destroy();
      });
    }
    return this.shutdownOnce;
  }

  async forceFlush(): Promise<void> {
    // No buffer of our own — BatchSpanProcessor handles batching.
  }
}
