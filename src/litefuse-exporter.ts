import { Langfuse } from "langfuse";
import http from "node:http";
import https from "node:https";

// ---------------------------------------------------------------------------
// Direct fetch using node:http/https — completely bypasses global-agent proxy
// patching on globalThis.fetch. Same approach as @opentelemetry/otlp-exporter-base
// which uses http.request() directly to avoid proxy interference.
// ---------------------------------------------------------------------------
const directHttpAgent = new http.Agent({ keepAlive: true });
const directHttpsAgent = new https.Agent({ keepAlive: true });

function directFetch(url: string | URL, init: RequestInit = {}): Promise<Response> {
  const parsedUrl = new URL(typeof url === "string" ? url : url.toString());
  const isHttps = parsedUrl.protocol === "https:";
  const lib = isHttps ? https : http;
  const agent = isHttps ? directHttpsAgent : directHttpAgent;

  return new Promise((resolve, reject) => {
    const method = (init.method || "GET").toUpperCase();
    const headers: Record<string, string | number> = {};
    if (init.headers) {
      if (typeof (init.headers as any).forEach === "function") {
        (init.headers as any).forEach((v: string, k: string) => { headers[k] = v; });
      } else if (typeof init.headers === "object") {
        Object.assign(headers, init.headers);
      }
    }
    const body = init.body ?? null;
    if (body && !headers["content-length"] && typeof body === "string") {
      headers["content-length"] = Buffer.byteLength(body);
    }

    const req = lib.request(parsedUrl, { method, headers: headers as any, agent, timeout: 30000 }, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const buf = Buffer.concat(chunks);
        const text = buf.toString("utf-8");
        resolve({
          ok: (res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 300,
          status: res.statusCode ?? 0,
          statusText: res.statusMessage || "",
          headers: new Headers(
            Object.entries(res.headers)
              .filter(([, v]) => v != null)
              .map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : v!] as [string, string])
          ),
          text: () => Promise.resolve(text),
          json: () => Promise.resolve(JSON.parse(text)),
          body: null,
        } as any);
      });
    });

    req.on("error", reject);
    req.on("timeout", () => { req.destroy(new Error("Request timeout")); });
    if (init.signal) {
      init.signal.addEventListener("abort", () => req.destroy(new Error("Aborted")), { once: true });
    }
    if (body) { req.write(body); }
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface SpanData {
  name: string;
  type: string;
  startTime: number;
  endTime?: number;
  attributes?: Record<string, any>;
  input?: any;
  output?: any;
  traceId: string;
  spanId: string;
  parentSpanId?: string;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Extract text content from a parts-based message */
function extractTextFromParts(parts: any[]): string {
  if (!Array.isArray(parts)) return "";
  return parts
    .filter((p: any) => p.type === "text" || p.type === "reasoning")
    .map((p: any) => p.content || p.text || "")
    .join("\n");
}

/** Convert parts-based messages to Litefuse ChatMessage[] format: [{ role, content }] */
function toChatMessages(messages: any[]): { role: string; content: string }[] {
  if (!Array.isArray(messages)) return [];
  return messages.map((msg: any) => ({
    role: msg.role || "user",
    content: msg.parts ? extractTextFromParts(msg.parts) : (typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? "")),
  }));
}

/** Keys already extracted into input/output — strip from metadata to avoid duplication. */
const GENERATION_MESSAGE_KEYS = new Set([
  "gen_ai.input.messages",
  "gen_ai.output.messages",
  "gen_ai.system_instructions",
]);

function stripMessageAttrs(attrs?: Record<string, any>): Record<string, any> | undefined {
  if (!attrs) return undefined;
  const result: Record<string, any> = {};
  for (const [k, v] of Object.entries(attrs)) {
    if (!GENERATION_MESSAGE_KEYS.has(k)) result[k] = v;
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function parseInput(attributes?: Record<string, any>): any {
  if (!attributes) return undefined;
  const result: { role: string; content: string }[] = [];

  // System prompt as first message
  if (attributes["gen_ai.system_instructions"]) {
    try {
      const parsed = JSON.parse(attributes["gen_ai.system_instructions"]);
      const text = Array.isArray(parsed) ? extractTextFromParts(parsed) : String(parsed);
      if (text) result.push({ role: "system", content: text });
    } catch {
      result.push({ role: "system", content: String(attributes["gen_ai.system_instructions"]) });
    }
  }

  // History + user messages
  if (attributes["gen_ai.input.messages"]) {
    try {
      const parsed = JSON.parse(attributes["gen_ai.input.messages"]);
      result.push(...toChatMessages(parsed));
    } catch {
      // fallback: raw string
    }
  }

  return result.length > 0 ? result : undefined;
}

function parseOutput(attributes?: Record<string, any>): any {
  if (!attributes) return undefined;
  if (attributes["gen_ai.output.messages"]) {
    try {
      const parsed = JSON.parse(attributes["gen_ai.output.messages"]);
      return toChatMessages(parsed);
    } catch {
      // fallback
    }
  }
  return undefined;
}

function createLitefuseClient(target: TargetConfig): Langfuse {
  const client = new Langfuse({
    publicKey: target.publicKey,
    secretKey: target.secretKey,
    baseUrl: target.baseUrl || "https://litefuse.cloud",
    requestTimeout: 30000,
  });
  // Monkey-patch fetch to use node:http directly, bypassing global-agent proxy
  (client as any).fetch = directFetch;
  return client;
}

// ---------------------------------------------------------------------------
// SingleTargetExporter — one Litefuse client with its own trace/span maps
// ---------------------------------------------------------------------------

class SingleTargetExporter {
  private api: PluginApi;
  private label: string;
  private litefuse: Langfuse;
  private traceMap = new Map<string, any>();
  private spanMap = new Map<string, any>();
  private agentSpanMap = new Map<string, any>();
  private debug: boolean;
  private tags: string[];
  private environment: string;

  constructor(api: PluginApi, target: TargetConfig, label: string, tags: string[], environment: string) {
    this.api = api;
    this.label = label;
    this.litefuse = createLitefuseClient(target);
    this.debug = target.debug || false;
    this.tags = tags;
    this.environment = environment;
  }

  getOrCreateTrace(traceId: string, metadata: Record<string, any> = {}): any {
    let trace = this.traceMap.get(traceId);
    if (!trace) {
      trace = this.litefuse.trace({
        id: traceId,
        name: `openclaw-${traceId.slice(0, 8)}`,
        metadata: { ...metadata, source: "openclaw-plugin" },
        tags: this.tags,
        environment: this.environment,
      });
      this.traceMap.set(traceId, trace);
    }
    return trace;
  }

  updateTrace(traceId: string, updates: Record<string, any> = {}): void {
    const trace = this.traceMap.get(traceId);
    if (trace) {
      trace.update(updates);
    }
  }

  /** Get the parent to nest under: agent span if exists, otherwise trace */
  private _getParent(traceId: string): any {
    const agentSpan = this.agentSpanMap.get(traceId);
    if (agentSpan) return agentSpan;
    return this.traceMap.get(traceId);
  }

  async startSpan(spanData: SpanData, customSpanId?: string): Promise<void> {
    const trace = this.getOrCreateTrace(spanData.traceId, {
      channelId: spanData.attributes?.["openclaw.channel.id"],
    });
    const spanId = customSpanId || spanData.spanId;
    const startTime = new Date(spanData.startTime);

    if (spanData.type === "entry" || !spanData.parentSpanId) {
      // Skip creating entry events — they create flat siblings that break the graph.
      this.spanMap.set(spanId, { type: "event", obj: null, trace });
    } else if (spanData.type === "agent") {
      const span = trace.span({
        id: spanId,
        name: spanData.name,
        startTime,
        metadata: stripMessageAttrs(spanData.attributes),
        input: spanData.input,
        environment: this.environment,
      });
      this.spanMap.set(spanId, { type: "span", obj: span, trace });
      this.agentSpanMap.set(spanData.traceId, span);
    } else if (spanData.type === "model") {
      const parent = this._getParent(spanData.traceId);
      const generation = parent.generation({
        id: spanId,
        name: spanData.name,
        model: spanData.attributes?.["gen_ai.request.model"] || "unknown",
        startTime,
        metadata: stripMessageAttrs(spanData.attributes),
        input: parseInput(spanData.attributes),
        output: parseOutput(spanData.attributes),
        environment: this.environment,
      });
      this.spanMap.set(spanId, { type: "generation", obj: generation, trace });
    }
  }

  endSpanById(
    spanId: string,
    endTime: number,
    additionalAttrs?: Record<string, any>,
    output?: any,
    _input?: any,
  ): void {
    const spanInfo = this.spanMap.get(spanId);
    if (!spanInfo) return;

    if (spanInfo.type === "generation" && spanInfo.obj) {
      if (additionalAttrs) {
        spanInfo.obj.update({
          metadata: stripMessageAttrs(additionalAttrs),
          output: output || parseOutput(additionalAttrs),
        });
      }
      spanInfo.obj.end({ endTime: endTime ? new Date(endTime) : undefined });
    } else if (spanInfo.type === "span" && spanInfo.obj) {
      const updatePayload: Record<string, any> = {};
      if (additionalAttrs) updatePayload.metadata = stripMessageAttrs(additionalAttrs);
      if (output) updatePayload.output = output;
      if (_input) updatePayload.input = _input;
      if (Object.keys(updatePayload).length > 0) {
        spanInfo.obj.update(updatePayload);
      }
      spanInfo.obj.end({ endTime: endTime ? new Date(endTime) : undefined });
    }
    // event type: fire-and-forget
    this.spanMap.delete(spanId);
  }

  async export(spanData: SpanData): Promise<void> {
    const trace = this.getOrCreateTrace(spanData.traceId, {
      channelId: spanData.attributes?.["openclaw.channel.id"],
    });
    const parent = this._getParent(spanData.traceId);
    const startTime = new Date(spanData.startTime);

    if (spanData.type === "model") {
      const usageInput = spanData.attributes?.["gen_ai.usage.input_tokens"];
      const usageOutput = spanData.attributes?.["gen_ai.usage.output_tokens"];
      const cacheRead = spanData.attributes?.["gen_ai.usage.cache_read.input_tokens"];
      const cacheWrite = spanData.attributes?.["gen_ai.usage.cache_creation.input_tokens"];

      const endTime = spanData.endTime ? new Date(spanData.endTime) : new Date();

      // Pass Anthropic's raw token breakdown to Litefuse via usageDetails.
      // Keys match default-model-prices.json so Litefuse calculates cost correctly:
      //   input                          → $3/MTok
      //   output                         → $15/MTok
      //   cache_read_input_tokens        → $0.30/MTok
      //   cache_creation_input_tokens    → $3.75/MTok
      // Server auto-calculates "total" from all keys (IngestionService:1401-1407).
      const hasUsage = (usageInput !== undefined && usageInput > 0) ||
                       (usageOutput !== undefined && usageOutput > 0) ||
                       (cacheRead !== undefined && cacheRead > 0) ||
                       (cacheWrite !== undefined && cacheWrite > 0);
      const usageDetails: Record<string, number> | undefined = hasUsage ? {
        input: usageInput ?? 0,
        output: usageOutput ?? 0,
        cache_read_input_tokens: cacheRead ?? 0,
        cache_creation_input_tokens: cacheWrite ?? 0,
      } : undefined;

      const generation = parent.generation({
        id: spanData.spanId,
        name: spanData.name,
        model: spanData.attributes?.["gen_ai.request.model"] || "unknown",
        startTime,
        endTime,
        metadata: stripMessageAttrs(spanData.attributes),
        input: parseInput(spanData.attributes),
        output: parseOutput(spanData.attributes),
        environment: this.environment,
      });

      if (usageDetails) {
        generation.update({ usageDetails });
      }
      generation.end({ endTime });
    } else if (spanData.type === "tool") {
      // Use "tool-create" event type directly for TOOL observation type
      // TOOL type is required for Litefuse graph/DAG rendering
      (this.litefuse as any).enqueue("tool-create", {
        id: spanData.spanId,
        traceId: parent.traceId,
        parentObservationId: parent.observationId || undefined,
        name: spanData.name,
        startTime,
        endTime: new Date(),
        metadata: spanData.attributes,
        input: spanData.attributes?.["gen_ai.tool.call.arguments"],
        output: spanData.attributes?.["gen_ai.tool.call.result"],
        environment: this.environment,
      });
    } else {
      // session/gateway/default events: fire-and-forget
      trace.event({ id: spanData.spanId, name: spanData.name, startTime, metadata: spanData.attributes, environment: this.environment });
    }
  }

  endTrace(): void {
    this.traceMap.clear();
    this.spanMap.clear();
    this.agentSpanMap.clear();
  }

  async flush(): Promise<void> {
    await this.litefuse.flushAsync();
  }

  async dispose(): Promise<void> {
    await this.litefuse.flushAsync();
    await this.litefuse.shutdownAsync();
    this.traceMap.clear();
    this.spanMap.clear();
  }
}

// ---------------------------------------------------------------------------
// Multi-target facade — delegates every call to all SingleTargetExporters
// ---------------------------------------------------------------------------

export class LitefuseExporter {
  private api: PluginApi;
  private config: LitefusePluginConfig;
  private targets: SingleTargetExporter[] = [];

  constructor(api: PluginApi, config: LitefusePluginConfig) {
    this.api = api;
    this.config = config;

    // Build target list: support both single config and targets array
    const targetConfigs: TargetConfig[] = config.targets || [
      {
        publicKey: config.publicKey!,
        secretKey: config.secretKey!,
        baseUrl: config.baseUrl,
        debug: config.debug,
      },
    ];

    const tags = config.tags || ["openclaw"];
    const environment = config.environment || "default";

    for (const t of targetConfigs) {
      if (!t.publicKey || !t.secretKey) continue;
      const label = t.name || t.baseUrl || "default";
      this.targets.push(new SingleTargetExporter(api, t, label, tags, environment));
      api.logger.info(`[Litefuse] Target added: ${label}`);
    }

    api.logger.info(`[Litefuse] Plugin initialized with ${this.targets.length} target(s)`);
  }

  getOrCreateTrace(traceId: string, metadata?: Record<string, any>): any {
    return this.targets[0]?.getOrCreateTrace(traceId, metadata);
  }

  async startSpan(spanData: SpanData, customSpanId?: string): Promise<void> {
    await Promise.allSettled(this.targets.map(t => t.startSpan(spanData, customSpanId)));
  }

  updateTrace(traceId: string, updates: Record<string, any>): void {
    for (const t of this.targets) t.updateTrace(traceId, updates);
  }

  endSpanById(spanId: string, endTime: number, additionalAttrs?: Record<string, any>, output?: any, _input?: any): void {
    for (const t of this.targets) t.endSpanById(spanId, endTime, additionalAttrs, output, _input);
  }

  async export(spanData: SpanData): Promise<void> {
    await Promise.allSettled(this.targets.map(t => t.export(spanData)));
  }

  endTrace(): void {
    for (const t of this.targets) t.endTrace();
  }

  async flush(): Promise<void> {
    await Promise.allSettled(this.targets.map(t => t.flush()));
    if (this.config.debug) this.api.logger.debug?.(`[Litefuse] Flushed ${this.targets.length} target(s)`);
  }

  async dispose(): Promise<void> {
    await Promise.allSettled(this.targets.map(t => t.dispose()));
    this.api.logger.info(`[Litefuse] Plugin disposed (${this.targets.length} targets)`);
  }
}
