import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import {
  startObservation,
  type LangfuseSpan,
  type LangfuseAgent,
  type LangfuseGeneration,
  type LangfuseTool,
} from "@langfuse/tracing";
import { LangfuseOtelSpanAttributes } from "@langfuse/core";
import { LitefuseExporter } from "./litefuse-exporter.js";
import type { LitefusePluginConfig } from "./litefuse-exporter.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function safeClone<T>(value: T): T {
  if (typeof globalThis.structuredClone === "function") {
    return globalThis.structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const MAX_ATTR_LENGTH = 3_200_000;

function truncateAttr(value: string): string {
  return value.length > MAX_ATTR_LENGTH
    ? value.substring(0, MAX_ATTR_LENGTH)
    : value;
}

function toSpecParts(content: any): any[] {
  if (content === undefined || content === null) return [];

  if (typeof content === "string") {
    return [{ type: "text", content }];
  }

  if (Array.isArray(content)) {
    return content.map((item) => {
      if (typeof item === "string") {
        return { type: "text", content: item };
      }
      if (typeof item === "object" && item !== null) {
        const obj = item as Record<string, any>;
        if (obj.type === "toolCall" || obj.type === "tool_call" || obj.type === "function_call") {
          return {
            type: "tool_call",
            id: obj.id || obj.toolCallId || null,
            name: obj.name || obj.toolName || "",
            arguments: obj.arguments || obj.input || obj.params || null,
          };
        }
        if (obj.type === "toolResult" || obj.type === "tool_result" || obj.type === "tool_call_response") {
          const resp = obj.response ?? obj.result ?? obj.content ?? "";
          return {
            type: "tool_call_response",
            id: obj.id || obj.toolCallId || null,
            response: typeof resp === "string" ? resp : JSON.stringify(resp),
          };
        }
        if (obj.type === "text") {
          return { type: "text", content: String(obj.content ?? obj.text ?? "") };
        }
        if (obj.type === "thinking" || obj.type === "reasoning") {
          return { type: "reasoning", content: String(obj.content ?? obj.thinking ?? "") };
        }
        if (obj.type) return obj;
        return { type: "text", content: JSON.stringify(item) };
      }
      return { type: "text", content: String(item) };
    });
  }

  return [{ type: "text", content: JSON.stringify(content) }];
}

const ROLE_MAP: Record<string, string> = {
  toolResult: "tool",
  tool_result: "tool",
  function: "tool",
};

function buildInputMessages(historyMessages: any[], userPrompt?: string, systemPrompt?: string): any[] {
  const result: any[] = [];
  if (systemPrompt) {
    result.push({ role: "system", content: systemPrompt });
  }
  for (const msg of historyMessages) {
    const role = ROLE_MAP[msg.role] || msg.role;
    const parts = toSpecParts(msg.content);
    const textParts = parts.filter((p) => p.type === "text" || p.type === "reasoning");
    const content = textParts.length
      ? textParts.map((p) => p.content || "").join("\n")
      : JSON.stringify(parts);
    result.push({ role, content });
  }
  if (userPrompt) {
    result.push({ role: "user", content: userPrompt });
  }
  return result;
}

function buildOutputMessages(assistantTexts: string[], finishReason = "stop"): any[] {
  return assistantTexts.map((text) => ({
    role: "assistant",
    content: text,
    finish_reason: finishReason,
  }));
}

function normalizeChannelId(input: string): string {
  if (!input || input === "unknown") return "system/unknown";
  if (input.includes("/")) return input;
  if (/^agent[_:]/.test(input)) return `agent/${input.slice(6)}`;
  return `system/${input}`;
}

function resolveChannelId(ctx: Record<string, any>, eventFrom?: string): string {
  const raw =
    ctx.sessionKey ||
    ctx.channelId ||
    ctx.conversationId ||
    eventFrom ||
    "unknown";
  return normalizeChannelId(raw);
}

// ---------------------------------------------------------------------------
// Per-turn state
// ---------------------------------------------------------------------------

interface TurnContext {
  runId: string;
  turnId: string;
  channelId: string;
  originalChannelId: string;
  sessionId?: string;
  userInput?: any;
  lastOutput?: any;

  // The root span carries trace-level attributes (TRACE_NAME, TRACE_USER_ID,
  // TRACE_INPUT, etc.) and is the parent of all per-turn observations.
  rootSpan?: LangfuseSpan;
  rootSpanStartTime?: number;

  // The agent span (when an agent runs) — parent of tool/generation spans
  // produced during that agent's invocation.
  agentSpan?: LangfuseAgent;
  agentStartTime?: number;

  // Currently-open generation (linked to the pending llm_input/llm_output).
  // Per-runId state lives in llmObsByRunId for late-arriving llm_output.
  llmSpan?: LangfuseGeneration;
  llmStartTime?: number;
}

interface PendingToolCall {
  toolName: string;
  toolSpan: LangfuseTool;
  toolStartTime: number;
  toolInput: any;
  turnCtx: TurnContext;
  channelId: string;
}

interface LlmRunState {
  generation: LangfuseGeneration;
  startTime: number;
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

function activate(api: OpenClawPluginApi): void {
  const pluginConfig = (api.pluginConfig || {}) as Record<string, any>;

  if (!pluginConfig.targets && !pluginConfig.publicKey) {
    api.logger.error("[Litefuse] Missing required configuration: 'publicKey' or 'targets' must be provided");
    return;
  }
  if (!pluginConfig.targets && !pluginConfig.secretKey) {
    api.logger.error("[Litefuse] Missing required configuration: 'secretKey' or 'targets' must be provided");
    return;
  }

  const config: LitefusePluginConfig = {
    publicKey: pluginConfig.publicKey,
    secretKey: pluginConfig.secretKey,
    baseUrl: pluginConfig.baseUrl,
    debug: pluginConfig.debug || false,
    enabledHooks: pluginConfig.enabledHooks,
    targets: pluginConfig.targets,
    tags: pluginConfig.tags,
    environment: pluginConfig.environment,
    userId: pluginConfig.userId,
  };

  const exporter = new LitefuseExporter(api, config);
  const tags = config.tags || ["openclaw"];

  // -- Per-turn state ------------------------------------------------------
  const contextByChannelId = new Map<string, TurnContext>();
  const contextByRunId = new Map<string, TurnContext>();
  const llmObsByRunId = new Map<string, LlmRunState>();

  let lastUserChannelId: string | undefined;
  let lastUserTurnContext: TurnContext | undefined;
  let pendingToolCall: PendingToolCall | undefined;

  const openclawVersion =
    (api.config as any)?.meta?.lastTouchedVersion || (api as any).runtime?.version || "unknown";

  const shouldHookEnabled = (hookName: string): boolean => {
    if (!config.enabledHooks) return true;
    return config.enabledHooks.includes(hookName);
  };

  // -- Context lookup / linking --------------------------------------------
  const getContextByChannel = (channelId: string) => contextByChannelId.get(channelId);
  const getContextByRun = (runId: string) => contextByRunId.get(runId);
  const getOriginalChannelId = (runId: string): string | undefined => {
    const ctx = contextByRunId.get(runId);
    return ctx?.originalChannelId || ctx?.channelId;
  };

  const startTurn = (runId: string, channelId: string, originalChannelId?: string): TurnContext => {
    const ctx: TurnContext = {
      runId,
      turnId: runId,
      channelId,
      originalChannelId: originalChannelId || channelId,
    };
    contextByChannelId.set(channelId, ctx);
    contextByRunId.set(runId, ctx);
    return ctx;
  };

  const endTurn = (channelId: string): void => {
    const ctx = contextByChannelId.get(channelId);
    if (ctx) {
      contextByChannelId.delete(channelId);
      contextByRunId.delete(ctx.runId);
    }
  };

  const getOrCreateContext = (
    rawChannelId: string,
    runId?: string,
    hookName?: string
  ): { ctx: TurnContext; channelId: string; isNew: boolean } => {
    let channelId = rawChannelId;
    let activeCtx: TurnContext | undefined;

    const effectiveRunId = runId || getContextByChannel(rawChannelId)?.runId || `run-${Date.now()}`;

    // For agent events, prefer lastUserTurnContext (most recent user message)
    // over stale channelId lookups from previous conversations.
    if (rawChannelId.startsWith("agent/") && lastUserTurnContext) {
      activeCtx = lastUserTurnContext;
      channelId = lastUserChannelId || channelId;
      contextByChannelId.set(rawChannelId, activeCtx);
      contextByRunId.set(effectiveRunId, activeCtx);

      if (config.debug) {
        api.logger.info(
          `[Litefuse] LINKING agent to user context: hook=${hookName}, agentChannel=${rawChannelId}, userChannel=${channelId}`
        );
      }
    }

    if (!activeCtx) {
      activeCtx = getContextByChannel(rawChannelId);
    }

    if (rawChannelId.startsWith("agent/") && !activeCtx && effectiveRunId) {
      const originalChannelId = getOriginalChannelId(effectiveRunId);
      if (originalChannelId) {
        channelId = originalChannelId;
        activeCtx = getContextByChannel(originalChannelId) || activeCtx;
      }
    }

    if (!activeCtx) {
      activeCtx = getContextByRun(effectiveRunId);
    }

    // Fallback: link to last user trace for processing hooks. Handles
    // platforms (e.g. TUI) where hookCtx resolves to a different channelId
    // than message_received used.
    if (
      !activeCtx &&
      lastUserTurnContext &&
      hookName &&
      hookName !== "message_received" &&
      hookName !== "gateway_start"
    ) {
      activeCtx = lastUserTurnContext;
      channelId = lastUserChannelId || channelId;
      contextByChannelId.set(rawChannelId, activeCtx);
      contextByRunId.set(effectiveRunId, activeCtx);

      if (config.debug) {
        api.logger.info(
          `[Litefuse] FALLBACK LINKING to user context: hook=${hookName}, rawChannel=${rawChannelId}, userChannel=${channelId}`
        );
      }
    }

    let isNew = false;
    if (!activeCtx) {
      activeCtx = startTurn(effectiveRunId, channelId, rawChannelId !== channelId ? rawChannelId : undefined);
      isNew = true;

      if (config.debug) {
        api.logger.info(
          `[Litefuse] NEW TurnContext: hook=${hookName}, channelId=${channelId}, runId=${effectiveRunId}`
        );
      }
    } else if (config.debug) {
      api.logger.info(
        `[Litefuse] REUSING TurnContext: hook=${hookName}, channelId=${channelId}`
      );
    }

    return { ctx: activeCtx, channelId, isNew };
  };

  // -- Root span helpers ---------------------------------------------------
  const ensureRootSpan = (ctx: TurnContext, channelId: string, options: Record<string, any> = {}): void => {
    if (ctx.rootSpan) return;

    const now = Date.now();
    ctx.rootSpanStartTime = now;

    // Use the user message content as the initial trace-level input. Other
    // identity fields (TRACE_NAME, TRACE_USER_ID, TRACE_SESSION_ID) are set
    // immediately after creation since they depend on the generated traceId.
    const span = startObservation(
      "enter_openclaw_system",
      {
        input: ctx.userInput,
        environment: config.environment,
      },
      { startTime: new Date(now) }
    );

    const userId = options.userId ?? "unknown";
    const rawUserId = options.rawUserId;
    const traceUserId = config.userId && rawUserId
      ? `${config.userId}/${rawUserId}`
      : config.userId || rawUserId || userId;

    span.otelSpan.setAttribute(LangfuseOtelSpanAttributes.AS_ROOT, true);
    span.otelSpan.setAttribute(
      LangfuseOtelSpanAttributes.TRACE_NAME,
      `openclaw-${span.traceId.slice(0, 8)}`
    );
    if (traceUserId) {
      span.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, String(traceUserId));
    }
    if (ctx.sessionId) {
      span.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, ctx.sessionId);
    }
    if (tags.length) {
      span.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_TAGS, tags);
    }
    span.otelSpan.setAttributes({
      "gen_ai.operation.name": "enter",
      "gen_ai.user.id": String(userId),
      "openclaw.session.id": ctx.sessionId || channelId,
      "openclaw.run.id": ctx.runId,
      "openclaw.turn.id": ctx.turnId,
      "openclaw.message.role": options.role || "unknown",
      "openclaw.message.from": options.from || "unknown",
      "openclaw.version": openclawVersion,
      "openclaw.channel.id": channelId,
    });

    ctx.rootSpan = span;

    if (config.debug) {
      api.logger.info(`[Litefuse] Started root span: traceId=${span.traceId}, spanId=${span.id}`);
    }
  };

  const parentForChildren = (ctx: TurnContext): LangfuseSpan | LangfuseAgent | undefined => {
    return ctx.agentSpan || ctx.rootSpan;
  };

  // -- Hooks --------------------------------------------------------------

  api.on("gateway_stop", async () => {
    try {
      await exporter.forceFlush();
    } catch {
      // best-effort
    }
    await exporter.shutdown();
  });

  if (shouldHookEnabled("gateway_start")) {
    api.on("gateway_start", async (event: any) => {
      // One-shot event with no parent context — emit as an isolated span.
      const span = startObservation(
        "gateway_start",
        {
          metadata: { "gateway.port": event.port || 0 },
          environment: config.environment,
        },
        { asType: "event" }
      );
      span.otelSpan.setAttribute(LangfuseOtelSpanAttributes.AS_ROOT, true);
      span.otelSpan.setAttribute(
        LangfuseOtelSpanAttributes.TRACE_NAME,
        `gateway-${span.traceId.slice(0, 8)}`
      );
      if (tags.length) {
        span.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_TAGS, tags);
      }
      span.otelSpan.setAttribute("openclaw.version", openclawVersion);
      span.end();
    });
  }

  if (shouldHookEnabled("session_start")) {
    api.on("session_start", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx, event.sessionId);
      const { ctx, channelId, isNew } = getOrCreateContext(rawChannelId, undefined, "session_start");

      // Same rationale as v3: persist sessionId on the trace only when reusing
      // an existing user context, to avoid ghost traces with sessionId but no
      // userId.
      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
        if (!isNew && ctx.rootSpan) {
          ctx.rootSpan.otelSpan.setAttribute(
            LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
            event.sessionId
          );
        }
      }

      const parent = parentForChildren(ctx);
      const evtAttrs: any = {
        metadata: { "event.type": "session_start" },
        environment: config.environment,
      };
      const opts = { asType: "event" as const };
      const evt = parent
        ? parent.startObservation("session_start", evtAttrs, opts)
        : startObservation("session_start", evtAttrs, opts);
      if (event.sessionId) {
        evt.otelSpan.setAttribute("openclaw.session.id", event.sessionId);
      }
      evt.end();
    });
  }

  if (shouldHookEnabled("session_end")) {
    api.on("session_end", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx, event.sessionId);
      const { ctx, channelId, isNew } = getOrCreateContext(rawChannelId, undefined, "session_end");

      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
        if (!isNew && ctx.rootSpan) {
          ctx.rootSpan.otelSpan.setAttribute(
            LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
            event.sessionId
          );
        }
      }

      const parent = parentForChildren(ctx);
      const evtAttrs: any = {
        metadata: {
          "session.duration_ms": event.durationMs || 0,
          "session.message_count": event.messageCount || 0,
        },
        output: {
          messageCount: event.messageCount,
          durationMs: event.durationMs,
        },
        environment: config.environment,
      };
      const opts = { asType: "event" as const };
      const evt = parent
        ? parent.startObservation("session_end", evtAttrs, opts)
        : startObservation("session_end", evtAttrs, opts);
      if (event.sessionId) {
        evt.otelSpan.setAttribute("openclaw.session.id", event.sessionId);
      }
      evt.end();

      endTurn(channelId);
    });
  }

  if (shouldHookEnabled("message_received")) {
    api.on("message_received", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx, event.from || event.metadata?.senderId);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "message_received");

      let role = event.role;
      if (!role && event.from) role = "user";

      const isUserMessage = !rawChannelId.startsWith("agent/");

      if (isUserMessage) {
        if (!role) role = "user";
        lastUserChannelId = channelId;
        lastUserTurnContext = ctx;
        ctx.userInput = event.content;

        const rawUserId = event.from || event.metadata?.senderId;
        ensureRootSpan(ctx, channelId, {
          userId: rawUserId || "unknown",
          rawUserId,
          role,
          from: event.from,
        });

        if (ctx.rootSpan) {
          ctx.rootSpan.update({ input: event.content });
        }
      }
    });
  }

  if (shouldHookEnabled("message_sending")) {
    api.on("message_sending", async (event: any, hookCtx: any) => {
      if (lastUserTurnContext) {
        lastUserTurnContext.lastOutput = event.content;
      } else {
        const rawChannelId = resolveChannelId(hookCtx, event.to);
        const { ctx } = getOrCreateContext(rawChannelId, undefined, "message_sending");
        ctx.lastOutput = event.content;
      }
    });
  }

  if (shouldHookEnabled("message_sent")) {
    api.on("message_sent", async (event: any, hookCtx: any) => {
      if (event.content && event.success) {
        if (lastUserTurnContext) {
          lastUserTurnContext.lastOutput = event.content;
        } else {
          const rawChannelId = resolveChannelId(hookCtx, event.to);
          const { ctx } = getOrCreateContext(rawChannelId, undefined, "message_sent");
          ctx.lastOutput = event.content;
        }
      }
    });
  }

  if (shouldHookEnabled("llm_input")) {
    api.on("llm_input", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, event.runId, "llm_input");

      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
        if (ctx.rootSpan) {
          ctx.rootSpan.otelSpan.setAttribute(
            LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
            event.sessionId
          );
        }
      }

      // Trace-level userId from llm context, in case message_received didn't
      // set it (e.g., agent-only invocations).
      const rawLlmUserId = hookCtx.trigger || event.from || event.metadata?.senderId || undefined;
      const userId = config.userId && rawLlmUserId
        ? `${config.userId}/${rawLlmUserId}`
        : config.userId || rawLlmUserId;
      if (userId && ctx.rootSpan) {
        ctx.rootSpan.otelSpan.setAttribute(
          LangfuseOtelSpanAttributes.TRACE_USER_ID,
          String(userId)
        );
      }

      if (!ctx.userInput && event.prompt) {
        ctx.userInput = event.prompt;
      }

      const now = Date.now();
      const historyMsgs = event.historyMessages?.length
        ? event.historyMessages.map((msg: any) => safeClone(msg))
        : [];
      const inputMessages = buildInputMessages(historyMsgs, event.prompt, event.systemPrompt);

      // Truncate via the LangfuseOtelSpanAttributes path — the SDK
      // JSON.stringifies whatever we hand it, so pre-serialize and clamp.
      const serializedInput = truncateAttr(JSON.stringify(inputMessages));

      const parent = parentForChildren(ctx);
      const genName = `chat ${event.model || "unknown"}`;
      const genAttrs = {
        model: event.model || "unknown",
        input: inputMessages,
        environment: config.environment,
      } as const;
      const genOpts = { asType: "generation" as const, startTime: new Date(now) };

      const generation = parent
        ? parent.startObservation(genName, genAttrs, genOpts)
        : startObservation(genName, genAttrs, genOpts);

      generation.otelSpan.setAttributes({
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": event.provider,
        "gen_ai.request.model": event.model,
        "gen_ai.response.model": event.model,
        "openclaw.session.id": ctx.sessionId || channelId,
        "openclaw.run.id": ctx.runId,
        "openclaw.turn.id": ctx.turnId,
        "openclaw.channel.id": channelId,
        "openclaw.version": openclawVersion,
      });
      if (event.systemPrompt) {
        generation.otelSpan.setAttribute(
          "gen_ai.system_instructions",
          truncateAttr(JSON.stringify([{ type: "text", content: event.systemPrompt }]))
        );
      }
      generation.otelSpan.setAttribute("gen_ai.input.messages", serializedInput);

      ctx.llmSpan = generation;
      ctx.llmStartTime = now;
      if (event.runId) {
        llmObsByRunId.set(event.runId, { generation, startTime: now });
      }

      if (config.debug) {
        api.logger.info(
          `[Litefuse] LLM input started: ${event.provider}/${event.model}, runId=${event.runId}, spanId=${generation.id}`
        );
      }
    });
  }

  if (shouldHookEnabled("llm_output")) {
    api.on("llm_output", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx } = getOrCreateContext(rawChannelId, event.runId, "llm_output");

      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
      }

      const now = Date.now();

      // Resolve the generation: ctx → per-runId map. If neither exists
      // (llm_output without prior llm_input) we silently skip — there's
      // no anchor to end against.
      const runState = event.runId ? llmObsByRunId.get(event.runId) : undefined;
      const generation = ctx.llmSpan || runState?.generation;
      if (!generation) {
        if (config.debug) {
          api.logger.warn(`[Litefuse] llm_output with no matching llm_input: runId=${event.runId}`);
        }
        return;
      }

      // Output text: prefer assistantTexts, fall back to lastAssistant.content
      let outputTexts: string[] = [];
      if (event.assistantTexts?.length) {
        outputTexts = event.assistantTexts;
      } else if (event.lastAssistant?.content) {
        for (const part of event.lastAssistant.content) {
          if (part.type === "text" && part.text) {
            const text = part.text.replace(/^\[\[reply_to_current\]\]\s*/, "");
            if (text) outputTexts.push(text);
          }
        }
      }

      if (outputTexts.length) {
        const outputText = outputTexts.join("\n");
        ctx.lastOutput = outputText;
        if (lastUserTurnContext) {
          lastUserTurnContext.lastOutput = outputText;
        }
      }

      const lastAssistantUsage = (event.lastAssistant as any)?.usage;
      const inputTokens = event.usage?.input ?? lastAssistantUsage?.input ?? 0;
      const outputTokens = event.usage?.output ?? lastAssistantUsage?.output ?? 0;
      const cacheReadTokens = event.usage?.cacheRead ?? lastAssistantUsage?.cacheRead ?? 0;
      const cacheCreationTokens = event.usage?.cacheWrite ?? lastAssistantUsage?.cacheWrite ?? 0;
      const hasUsage =
        inputTokens > 0 || outputTokens > 0 || cacheReadTokens > 0 || cacheCreationTokens > 0;

      const stopReason =
        typeof event.lastAssistant?.stopReason === "string" ? event.lastAssistant.stopReason : undefined;
      const outputMessages = outputTexts.length
        ? buildOutputMessages(outputTexts, stopReason || "stop")
        : undefined;

      const updatePayload: any = {};
      if (outputMessages) {
        updatePayload.output = outputMessages;
      }
      if (hasUsage) {
        // Keys deliberately match Litefuse's default-model-prices.json so the
        // server can compute cost per cache tier.
        updatePayload.usageDetails = {
          input: inputTokens,
          output: outputTokens,
          cache_read_input_tokens: cacheReadTokens,
          cache_creation_input_tokens: cacheCreationTokens,
        };
      }
      generation.update(updatePayload);

      // Also stamp the OTel semantic-convention attributes that the v3
      // implementation provided — preserves any downstream code that reads
      // them off raw OTel spans.
      generation.otelSpan.setAttributes({
        "gen_ai.usage.input_tokens": inputTokens,
        "gen_ai.usage.output_tokens": outputTokens,
        "gen_ai.usage.total_tokens": inputTokens + outputTokens,
        "gen_ai.usage.cache_read.input_tokens": cacheReadTokens,
        "gen_ai.usage.cache_creation.input_tokens": cacheCreationTokens,
      });
      if (stopReason) {
        generation.otelSpan.setAttribute(
          "gen_ai.response.finish_reasons",
          JSON.stringify([stopReason])
        );
      }
      if (outputMessages) {
        generation.otelSpan.setAttribute(
          "gen_ai.output.messages",
          truncateAttr(JSON.stringify(outputMessages))
        );
      }

      generation.end(new Date(now));

      // Clear pointers
      if (ctx.llmSpan === generation) {
        ctx.llmSpan = undefined;
        ctx.llmStartTime = undefined;
      }
      if (event.runId) llmObsByRunId.delete(event.runId);

      if (config.debug) {
        const startTime = runState?.startTime || ctx.llmStartTime || now;
        api.logger.info(
          `[Litefuse] Exported LLM generation: ${event.provider}/${event.model}, duration=${now - startTime}ms`
        );
      }
    });
  }

  if (shouldHookEnabled("before_tool_call")) {
    api.on("before_tool_call", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "before_tool_call");

      const now = Date.now();
      const parent = parentForChildren(ctx);
      const toolAttrs: any = {
        input: event.params,
        environment: config.environment,
      };
      const toolOpts = { asType: "tool" as const, startTime: new Date(now) };
      const toolSpan = parent
        ? parent.startObservation(`execute_tool ${event.toolName}`, toolAttrs, toolOpts)
        : startObservation(`execute_tool ${event.toolName}`, toolAttrs, toolOpts);

      toolSpan.otelSpan.setAttributes({
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": event.toolName,
        "gen_ai.tool.type": "function",
        "openclaw.session.id": ctx.sessionId || channelId,
        "openclaw.run.id": ctx.runId,
        "openclaw.turn.id": ctx.turnId,
        "openclaw.channel.id": channelId,
        "openclaw.version": openclawVersion,
      });
      if (event.params !== undefined) {
        toolSpan.otelSpan.setAttribute(
          "gen_ai.tool.call.arguments",
          truncateAttr(typeof event.params === "string" ? event.params : JSON.stringify(event.params))
        );
      }

      pendingToolCall = {
        toolName: event.toolName,
        toolSpan,
        toolStartTime: now,
        toolInput: event.params,
        turnCtx: ctx,
        channelId,
      };

      if (config.debug) {
        api.logger.info(`[Litefuse] Tool call started: ${event.toolName}, spanId=${toolSpan.id}`);
      }
    });
  }

  if (shouldHookEnabled("after_tool_call")) {
    api.on("after_tool_call", async (event: any, _hookCtx: any) => {
      if (!pendingToolCall || pendingToolCall.toolName !== event.toolName) {
        return;
      }

      const { toolName, toolSpan, toolStartTime } = pendingToolCall;
      pendingToolCall = undefined;

      const now = Date.now();
      const durationMs = event.durationMs ?? now - toolStartTime;

      const updatePayload: any = {};
      if (event.error) {
        updatePayload.level = "ERROR";
        updatePayload.statusMessage = String(event.error);
      } else if (event.result !== undefined) {
        updatePayload.output = event.result;
      }
      toolSpan.update(updatePayload);

      toolSpan.otelSpan.setAttribute("tool.duration_ms", durationMs);
      if (event.error) {
        toolSpan.otelSpan.setAttribute("error.type", String(event.error));
      } else if (event.result !== undefined) {
        toolSpan.otelSpan.setAttribute(
          "gen_ai.tool.call.result",
          truncateAttr(typeof event.result === "string" ? event.result : JSON.stringify(event.result))
        );
      }

      toolSpan.end(new Date(now));

      if (config.debug) {
        api.logger.info(`[Litefuse] Exported tool span: ${toolName}, duration=${durationMs}ms`);
      }
    });
  }

  if (shouldHookEnabled("before_agent_start")) {
    api.on("before_agent_start", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const agentId = hookCtx.agentId || event.agentId || "openclaw";
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "before_agent_start");

      ensureRootSpan(ctx, channelId, {
        userId: hookCtx.trigger || "system",
        role: hookCtx.trigger || "system",
        from: agentId,
      });

      // Only first before_agent_start per turn gets a real agent span;
      // subsequent fires (nested agents) reuse the existing one to preserve
      // the v3 behavior of one agent observation per turn.
      if (ctx.agentSpan) return;

      const now = Date.now();
      ctx.agentStartTime = now;

      const agentInput = ctx.userInput || lastUserTurnContext?.userInput;
      const parent = ctx.rootSpan;
      const agentAttrs: any = {
        input: agentInput,
        environment: config.environment,
      };
      const agentOpts = { asType: "agent" as const, startTime: new Date(now) };
      const agentSpan = parent
        ? parent.startObservation(`invoke_agent ${agentId}`, agentAttrs, agentOpts)
        : startObservation(`invoke_agent ${agentId}`, agentAttrs, agentOpts);

      agentSpan.otelSpan.setAttributes({
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": "openclaw",
        "gen_ai.agent.id": agentId,
        "gen_ai.agent.name": agentId,
        "openclaw.session.id": ctx.sessionId || channelId,
        "openclaw.run.id": ctx.runId,
        "openclaw.turn.id": ctx.turnId,
        "openclaw.channel.id": channelId,
        "openclaw.version": openclawVersion,
      });

      ctx.agentSpan = agentSpan;

      if (config.debug) {
        api.logger.info(`[Litefuse] Started agent span: ${agentId}, spanId=${agentSpan.id}`);
      }
    });
  }

  if (shouldHookEnabled("agent_end")) {
    api.on("agent_end", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "agent_end");

      const pendingAgentSpan = ctx.agentSpan;
      ctx.agentSpan = undefined;
      ctx.agentStartTime = undefined;

      const agentEndAttrs: Record<string, any> | undefined = pendingAgentSpan
        ? {
            "agent.duration_ms": event.durationMs || 0,
            "agent.message_count": Array.isArray(event.messages) ? event.messages.length : 0,
            "agent.success": event.success ?? true,
            ...(event.error ? { "agent.error": event.error } : {}),
          }
        : undefined;

      // Snapshot — do NOT clear lastUserTurnContext yet. llm_output may fire
      // after agent_end and needs to find the user context.
      const savedLastUserTurn = lastUserTurnContext;
      const savedLastUserChannelId = lastUserChannelId;
      const originalChannelId = ctx.originalChannelId || savedLastUserChannelId || channelId;
      const rootCtx = savedLastUserTurn || ctx;

      if (rootCtx.rootSpan || pendingAgentSpan) {
        const root = rootCtx.rootSpan;
        const rootSessionId = ctx.sessionId || rootCtx.sessionId;
        const userInput = rootCtx.userInput;

        // Defer agent/root end so late llm_output / message_sent events can
        // still write to the trace before it's flushed.
        setTimeout(async () => {
          const finalOutput = ctx.lastOutput || rootCtx.lastOutput;

          if (pendingAgentSpan) {
            const agentUpdate: any = {};
            if (userInput !== undefined) agentUpdate.input = userInput;
            if (finalOutput !== undefined) agentUpdate.output = finalOutput;
            pendingAgentSpan.update(agentUpdate);
            if (agentEndAttrs) {
              pendingAgentSpan.otelSpan.setAttributes(agentEndAttrs);
              if (userInput !== undefined) {
                pendingAgentSpan.otelSpan.setAttribute(
                  "gen_ai.input.messages",
                  truncateAttr(JSON.stringify([{ role: "user", content: String(userInput) }]))
                );
              }
              if (finalOutput !== undefined) {
                pendingAgentSpan.otelSpan.setAttribute(
                  "gen_ai.output.messages",
                  truncateAttr(JSON.stringify(buildOutputMessages([
                    typeof finalOutput === "string" ? finalOutput : JSON.stringify(finalOutput),
                  ])))
                );
              }
            }
            pendingAgentSpan.end();
            if (config.debug) {
              api.logger.info(
                `[Litefuse] Ended agent span: spanId=${pendingAgentSpan.id}, duration=${event.durationMs}ms`
              );
            }
          }

          if (root && rootCtx.rootSpanStartTime) {
            const endTime = Date.now();
            // Final trace-level attributes on the root span. In OTel each
            // span is shipped once on end(), so set everything here.
            const rootUpdate: any = {};
            if (userInput !== undefined) rootUpdate.input = userInput;
            if (finalOutput !== undefined) rootUpdate.output = finalOutput;
            root.update(rootUpdate);

            if (userInput !== undefined) {
              root.otelSpan.setAttribute(
                LangfuseOtelSpanAttributes.TRACE_INPUT,
                truncateAttr(JSON.stringify(userInput))
              );
            }
            if (finalOutput !== undefined) {
              root.otelSpan.setAttribute(
                LangfuseOtelSpanAttributes.TRACE_OUTPUT,
                truncateAttr(JSON.stringify(finalOutput))
              );
              root.otelSpan.setAttribute(
                "gen_ai.output.messages",
                truncateAttr(JSON.stringify(buildOutputMessages([
                  typeof finalOutput === "string" ? finalOutput : JSON.stringify(finalOutput),
                ])))
              );
            }
            if (userInput !== undefined) {
              root.otelSpan.setAttribute(
                "gen_ai.input.messages",
                truncateAttr(JSON.stringify([{ role: "user", content: String(userInput) }]))
              );
            }
            if (rootSessionId) {
              root.otelSpan.setAttribute(
                LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
                rootSessionId
              );
              root.otelSpan.setAttribute("openclaw.session.id", rootSessionId);
            }
            root.otelSpan.setAttribute(
              "request.duration_ms",
              endTime - rootCtx.rootSpanStartTime
            );

            root.end(new Date(endTime));
            rootCtx.rootSpan = undefined;
            rootCtx.rootSpanStartTime = undefined;

            if (config.debug) {
              api.logger.info(
                `[Litefuse] Ended root span: spanId=${root.id}, duration=${endTime - (rootCtx.rootSpanStartTime || endTime)}ms, traceId=${root.traceId}`
              );
            }
          }

          // Clear global pointers (deferred — see comment above)
          lastUserChannelId = undefined;
          lastUserTurnContext = undefined;

          if (savedLastUserChannelId) endTurn(savedLastUserChannelId);
          if (originalChannelId && originalChannelId !== savedLastUserChannelId) {
            endTurn(originalChannelId);
          }
          if (rawChannelId !== originalChannelId && rawChannelId !== savedLastUserChannelId) {
            contextByChannelId.delete(rawChannelId);
          }

          // Let the OTel BatchSpanProcessor drain its in-memory queue before
          // forcing a flush. Mirrors the v3 500ms wait before flushAsync.
          await new Promise((r) => setTimeout(r, 500));
          await exporter.forceFlush();
        }, 200);
      } else {
        lastUserChannelId = undefined;
        lastUserTurnContext = undefined;
        if (savedLastUserChannelId) endTurn(savedLastUserChannelId);
        if (originalChannelId && originalChannelId !== savedLastUserChannelId) {
          endTurn(originalChannelId);
        }
        if (rawChannelId !== originalChannelId && rawChannelId !== savedLastUserChannelId) {
          contextByChannelId.delete(rawChannelId);
        }
        await exporter.forceFlush();
      }
    });
  }

  api.logger.info(
    `[Litefuse] Plugin activated (baseUrl: ${config.baseUrl || "litefuse.cloud"})`
  );
}

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------

const plugin = {
  id: "openclaw-litefuse-plugin",
  name: "OpenClaw Litefuse Plugin",
  version: "0.3.0",
  description: "Report OpenClaw AI agent execution traces to Litefuse (OTel-based)",

  configSchema: {
    type: "object",
    properties: {
      publicKey: {
        type: "string",
        default: "",
        description: "Litefuse public key (single target mode)",
      },
      secretKey: {
        type: "string",
        default: "",
        description: "Litefuse secret key (single target mode)",
      },
      baseUrl: {
        type: "string",
        default: "https://litefuse.cloud",
        description: "Litefuse server URL (use self-hosted URL for on-premise)",
      },
      targets: {
        type: "array",
        description: "Multiple Litefuse targets (overrides publicKey/secretKey)",
        items: {
          type: "object",
          properties: {
            name: { type: "string", description: "Target display name" },
            publicKey: { type: "string" },
            secretKey: { type: "string" },
            baseUrl: { type: "string" },
          },
          required: ["publicKey", "secretKey"],
        },
      },
      tags: {
        type: "array",
        items: { type: "string" },
        default: ["openclaw"],
        description: "Tags to attach to all Litefuse traces (e.g. instance name, team)",
      },
      environment: {
        type: "string",
        default: "default",
        description: "Litefuse environment label (e.g. production, staging, development)",
      },
      userId: {
        type: "string",
        description: "userId prefix on traces (e.g. 'alice' → 'alice/openclaw-tui')",
      },
      debug: {
        type: "boolean",
        default: false,
        description: "Enable debug logging",
      },
      enabledHooks: {
        type: "array",
        items: { type: "string" },
        description: "List of hooks to enable (if not set, all hooks are enabled)",
      },
    },
  },

  register(api: OpenClawPluginApi) {
    activate(api);
  },
};

export default plugin;
