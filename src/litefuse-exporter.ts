import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { setLangfuseTracerProvider } from "@langfuse/tracing";
import { DirectOtlpHttpExporter } from "./otlp-exporter.js";
import { LANGFUSE_SDK_VERSION } from "@langfuse/core";

export interface TargetConfig {
  name?: string;
  publicKey: string;
  secretKey: string;
  baseUrl?: string;
  debug?: boolean;
}

export interface LitefusePluginConfig {
  publicKey?: string;
  secretKey?: string;
  baseUrl?: string;
  debug?: boolean;
  targets?: TargetConfig[];
  enabledHooks?: string[];
  tags?: string[];
  environment?: string;
  userId?: string;
}

export interface PluginApi {
  logger: {
    info: (msg: string) => void;
    error: (msg: string) => void;
    warn: (msg: string) => void;
    debug?: (msg: string) => void;
  };
}

const DEFAULT_BASE_URL = "https://litefuse.cloud";

function buildAuthHeader(publicKey: string, secretKey: string): string {
  return "Basic " + Buffer.from(`${publicKey}:${secretKey}`).toString("base64");
}

function makeProcessor(target: TargetConfig, environment: string): LangfuseSpanProcessor {
  const baseUrl = target.baseUrl || DEFAULT_BASE_URL;
  const exporter = new DirectOtlpHttpExporter({
    url: `${baseUrl.replace(/\/$/, "")}/api/public/otel/v1/traces`,
    headers: {
      Authorization: buildAuthHeader(target.publicKey, target.secretKey),
      "x-langfuse-sdk-name": "javascript",
      "x-langfuse-sdk-version": LANGFUSE_SDK_VERSION,
      "x-langfuse-public-key": target.publicKey,
    },
  });

  return new LangfuseSpanProcessor({
    publicKey: target.publicKey,
    secretKey: target.secretKey,
    baseUrl,
    environment,
    exporter,
    // Our spans go through @langfuse/tracing so they always pass the default
    // filter, but the override is cheap insurance against future scope changes.
    shouldExportSpan: () => true,
  });
}

export class LitefuseExporter {
  private readonly api: PluginApi;
  private readonly config: LitefusePluginConfig;
  private readonly provider: NodeTracerProvider;
  private readonly processors: LangfuseSpanProcessor[];

  constructor(api: PluginApi, config: LitefusePluginConfig) {
    this.api = api;
    this.config = config;

    const targets: TargetConfig[] = config.targets || [
      {
        publicKey: config.publicKey!,
        secretKey: config.secretKey!,
        baseUrl: config.baseUrl,
        debug: config.debug,
      },
    ];

    const environment = config.environment || "default";
    this.processors = [];
    for (const t of targets) {
      if (!t.publicKey || !t.secretKey) continue;
      const proc = makeProcessor(t, environment);
      this.processors.push(proc);
      api.logger.info(`[Litefuse] Target added: ${t.name || t.baseUrl || DEFAULT_BASE_URL}`);
    }

    this.provider = new NodeTracerProvider({ spanProcessors: this.processors });
    // Use Langfuse's own provider slot rather than registering as global OTel
    // provider — keeps us from clobbering whatever OpenClaw's host process
    // may already have set up.
    setLangfuseTracerProvider(this.provider);

    api.logger.info(
      `[Litefuse] Plugin initialized with ${this.processors.length} target(s) (env=${environment})`
    );
  }

  async forceFlush(): Promise<void> {
    await this.provider.forceFlush();
    if (this.config.debug) {
      this.api.logger.debug?.(`[Litefuse] Flushed ${this.processors.length} target(s)`);
    }
  }

  async shutdown(): Promise<void> {
    await this.provider.shutdown();
    this.api.logger.info(`[Litefuse] Plugin disposed (${this.processors.length} targets)`);
  }
}
