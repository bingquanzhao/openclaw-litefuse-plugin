import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { LitefuseExporter } from "./litefuse-exporter.js";
import type { LitefusePluginConfig, TargetConfig } from "./litefuse-exporter.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateId(length = 16): string {
  const chars = "0123456789abcdef";
  let result = "";
  for (let i = 0; i < length; i++) {
    result += chars[Math.floor(Math.random() * chars.length)];
  }
  return result;
}

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

function formatSystemInstructions(systemPrompt: string): string {
  return truncateAttr(JSON.stringify([{ type: "text", content: systemPrompt }]));
}

const ROLE_MAP: Record<string, string> = {
  toolResult: "tool",
  tool_result: "tool",
  function: "tool",
};

function formatInputMessages(historyMessages: any[], userPrompt?: string): string {
  const result = [];
  for (const msg of historyMessages) {
    const role = ROLE_MAP[msg.role] || msg.role;
    result.push({ role, parts: toSpecParts(msg.content) });
  }
  if (userPrompt) {
    result.push({ role: "user", parts: [{ type: "text", content: userPrompt }] });
  }
  return truncateAttr(JSON.stringify(result));
}

function formatOutputMessages(assistantTexts: string[], finishReason = "stop"): string {
  return truncateAttr(
    JSON.stringify(
      assistantTexts.map((text) => ({
        role: "assistant",
        parts: [{ type: "text", content: text }],
        finish_reason: finishReason,
      }))
    )
  );
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
// Plugin implementation
// ---------------------------------------------------------------------------

interface TraceContext {
  traceId: string;
  rootSpanId: string;
  rootSpanStartTime?: number;
  runId: string;
  turnId: string;
  channelId: string;
  originalChannelId: string;
  sessionId?: string;
  userInput?: any;
  lastOutput?: any;
  agentSpanId?: string;
  agentStartTime?: number;
  llmStartTime?: number;
  llmSpanId?: string;
  llmSystemInstructions?: string;
  llmInputMessages?: string;
}

interface PendingToolCall {
  toolName: string;
  toolCallId: string;
  toolSpanId: string;
  toolStartTime: number;
  toolInput: any;
  traceContext: TraceContext;
  channelId: string;
}

function activate(api: OpenClawPluginApi): void {
  const pluginConfig = (api.pluginConfig || {}) as Record<string, any>;

  // When using targets array, publicKey/secretKey are optional at top level
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

  // -- Trace context management -------------------------------------------
  const contextByChannelId = new Map<string, TraceContext>();
  const contextByRunId = new Map<string, TraceContext>();

  let lastUserChannelId: string | undefined;
  let lastUserTraceContext: TraceContext | undefined;
  let pendingToolCall: PendingToolCall | undefined;

  let lastLlmSystemInstructions: string | undefined;
  let lastLlmInputMessages: string | undefined;
  let lastLlmStartTime: number | undefined;
  let lastLlmSpanId: string | undefined;

  // Per-runId LLM timing — survives context linking failures
  const llmTimingByRunId = new Map<string, { startTime: number; spanId: string; systemInstructions?: string; inputMessages?: string }>();

  const openclawVersion = (api.config as any)?.meta?.lastTouchedVersion || (api as any).runtime?.version || "unknown";

  const shouldHookEnabled = (hookName: string): boolean => {
    if (!config.enabledHooks) return true;
    return config.enabledHooks.includes(hookName);
  };

  const getContextByChannel = (channelId: string) => contextByChannelId.get(channelId);
  const getContextByRun = (runId: string) => contextByRunId.get(runId);

  const getOriginalChannelId = (runId: string): string | undefined => {
    const ctx = contextByRunId.get(runId);
    return ctx?.originalChannelId || ctx?.channelId;
  };

  const startTurn = (runId: string, channelId: string, originalChannelId?: string): TraceContext => {
    const traceId = generateId(32);
    const ctx: TraceContext = {
      traceId,
      rootSpanId: generateId(16),
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
  ): { ctx: TraceContext; channelId: string; isNew: boolean } => {
    let channelId = rawChannelId;
    let activeCtx: TraceContext | undefined;

    const effectiveRunId = runId || getContextByChannel(rawChannelId)?.runId || `run-${Date.now()}`;

    // For agent events, prefer lastUserTraceContext (most recent user message)
    // over stale channelId lookups from previous conversations.
    if (rawChannelId.startsWith("agent/") && lastUserTraceContext) {
      activeCtx = lastUserTraceContext;
      channelId = lastUserChannelId || channelId;
      contextByChannelId.set(rawChannelId, activeCtx);
      contextByRunId.set(effectiveRunId, activeCtx);

      if (config.debug) {
        api.logger.info(
          `[Litefuse] LINKING agent to user context: hook=${hookName}, agentChannel=${rawChannelId}, userChannel=${channelId}, traceId=${activeCtx.traceId}`
        );
      }
    }

    // Non-agent events or no lastUserTraceContext: use channelId/runId lookup
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

    // Fallback: link to last user trace for processing hooks.
    // Handles platforms (e.g. TUI) where hookCtx resolves to a different
    // channelId than the one used by message_received.
    if (
      !activeCtx &&
      lastUserTraceContext &&
      hookName &&
      hookName !== "message_received" &&
      hookName !== "gateway_start"
    ) {
      activeCtx = lastUserTraceContext;
      channelId = lastUserChannelId || channelId;
      contextByChannelId.set(rawChannelId, activeCtx);
      contextByRunId.set(effectiveRunId, activeCtx);

      if (config.debug) {
        api.logger.info(
          `[Litefuse] FALLBACK LINKING to user context: hook=${hookName}, rawChannel=${rawChannelId}, userChannel=${channelId}, traceId=${activeCtx.traceId}`
        );
      }
    }

    let isNew = false;
    if (!activeCtx) {
      activeCtx = startTurn(effectiveRunId, channelId, rawChannelId !== channelId ? rawChannelId : undefined);
      isNew = true;

      if (config.debug) {
        api.logger.info(
          `[Litefuse] NEW TraceContext: hook=${hookName}, channelId=${channelId}, runId=${effectiveRunId}, traceId=${activeCtx.traceId}`
        );
      }
    } else if (config.debug) {
      api.logger.info(
        `[Litefuse] REUSING TraceContext: hook=${hookName}, channelId=${channelId}, traceId=${activeCtx.traceId}`
      );
    }

    return { ctx: activeCtx, channelId, isNew };
  };

  const createSpan = (
    ctx: TraceContext,
    channelId: string,
    name: string,
    type: string,
    startTime: number,
    endTime: number,
    attributes: Record<string, any> = {},
    input?: any,
    output?: any,
    parentSpanId?: string
  ) => ({
    name,
    type,
    startTime,
    endTime,
    attributes: {
      ...attributes,
      "openclaw.version": openclawVersion,
      "openclaw.session.id": ctx.sessionId || channelId,
      "gen_ai.session.id": ctx.sessionId || channelId,
      "openclaw.run.id": ctx.runId,
      "openclaw.turn.id": ctx.turnId,
      "openclaw.channel.id": channelId,
    },
    input,
    output,
    traceId: ctx.traceId,
    spanId: generateId(16),
    parentSpanId: parentSpanId || ctx.rootSpanId,
  });

  const ensureEntrySpan = async (ctx: TraceContext, channelId: string, options: Record<string, any> = {}) => {
    if (ctx.rootSpanStartTime) return;

    const now = Date.now();
    ctx.rootSpanStartTime = now;

    const rootSpanData = {
      name: "enter_openclaw_system",
      type: "entry",
      startTime: now,
      attributes: {
        "gen_ai.operation.name": "enter",
        "gen_ai.user.id": options.userId || "unknown",
        "openclaw.session.id": ctx.sessionId || channelId,
        "gen_ai.session.id": ctx.sessionId || channelId,
        "openclaw.run.id": ctx.runId,
        "openclaw.turn.id": ctx.turnId,
        "openclaw.message.role": options.role || "unknown",
        "openclaw.message.from": options.from || "unknown",
        "openclaw.version": openclawVersion,
      },
      input: ctx.userInput,
      traceId: ctx.traceId,
      spanId: ctx.rootSpanId,
    };

    await exporter.startSpan(rootSpanData, ctx.rootSpanId);

    if (config.debug) {
      api.logger.info(`[Litefuse] Started root span: traceId=${ctx.traceId}, spanId=${ctx.rootSpanId}`);
    }
  };

  // -- Hook: gateway_stop -------------------------------------------------
  api.on("gateway_stop", async () => {
    await exporter.dispose();
  });

  // -- Hook: gateway_start ------------------------------------------------
  if (shouldHookEnabled("gateway_start")) {
    api.on("gateway_start", async (event: any) => {
      const now = Date.now();
      const { ctx, channelId } = getOrCreateContext("system/gateway", undefined, "gateway_start");

      const span = createSpan(ctx, channelId, "gateway_start", "gateway", now, now, {
        "gateway.port": event.port || 0,
      });

      delete (span.attributes as any)["openclaw.session.id"];
      delete (span.attributes as any)["gen_ai.session.id"];

      await exporter.export(span);
    });
  }

  // -- Hook: session_start ------------------------------------------------
  if (shouldHookEnabled("session_start")) {
    api.on("session_start", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx, event.sessionId);
      const { ctx, channelId, isNew } = getOrCreateContext(rawChannelId, undefined, "session_start");

      // Propagate sessionId to the context so later hooks (agent_end,
      // llm_input fallbacks) can use it. Only persist it on the trace when
      // we reused an existing context (typically the user's message_received
      // trace, which already has userId). If we created a brand-new context,
      // setting sessionId would produce a ghost trace — visible in the
      // sessions list but without a user_id — which inflates session
      // trace_count when users filter sessions by their userId.
      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
        if (!isNew) {
          exporter.updateTrace(ctx.traceId, { sessionId: event.sessionId });
        }
      }

      const now = Date.now();
      const span = createSpan(ctx, channelId, "session_start", "session", now, now, {
        "event.type": "session_start",
      });

      delete (span.attributes as any)["gen_ai.session.id"];

      if (event.sessionId) {
        span.attributes["openclaw.session.id"] = event.sessionId;
      }

      await exporter.export(span);
    });
  }

  // -- Hook: session_end --------------------------------------------------
  if (shouldHookEnabled("session_end")) {
    api.on("session_end", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx, event.sessionId);
      const { ctx, channelId, isNew } = getOrCreateContext(rawChannelId, undefined, "session_end");

      // Same rationale as session_start: only persist sessionId on the trace
      // when we reused an existing context, to avoid creating ghost traces
      // with session_id but no user_id.
      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
        if (!isNew) {
          exporter.updateTrace(ctx.traceId, { sessionId: event.sessionId });
        }
      }

      const now = Date.now();
      const span = createSpan(
        ctx,
        channelId,
        "session_end",
        "session",
        now,
        now,
        {
          "session.duration_ms": event.durationMs || 0,
          "session.message_count": event.messageCount || 0,
        },
        undefined,
        {
          messageCount: event.messageCount,
          durationMs: event.durationMs,
        }
      );

      delete (span.attributes as any)["gen_ai.session.id"];

      if (event.sessionId) {
        span.attributes["openclaw.session.id"] = event.sessionId;
      }

      await exporter.export(span);
      endTurn(channelId);
    });
  }

  // -- Hook: message_received ---------------------------------------------
  if (shouldHookEnabled("message_received")) {
    api.on("message_received", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx, event.from || event.metadata?.senderId);
      const { ctx, channelId, isNew } = getOrCreateContext(rawChannelId, undefined, "message_received");

      let role = event.role;
      if (!role && event.from) role = "user";

      const isUserMessage = !rawChannelId.startsWith("agent/");

      if (isUserMessage) {
        if (!role) role = "user";
        lastUserChannelId = channelId;
        lastUserTraceContext = ctx;
        ctx.userInput = event.content;

        await ensureEntrySpan(ctx, channelId, {
          userId: event.from || event.metadata?.senderId,
          role: role || "user",
          from: event.from,
        });

        // Set userId and input on trace
        const rawUserId = event.from || event.metadata?.senderId;
        const msgUserId = config.userId && rawUserId ? `${config.userId}/${rawUserId}` : config.userId || rawUserId;
        exporter.updateTrace(ctx.traceId, {
          ...(msgUserId ? { userId: String(msgUserId) } : {}),
          input: event.content,
        });
      }
    });
  }

  // -- Hook: message_sending ----------------------------------------------
  if (shouldHookEnabled("message_sending")) {
    api.on("message_sending", async (event: any, _hookCtx: any) => {
      if (lastUserTraceContext) {
        lastUserTraceContext.lastOutput = event.content;
      } else {
        const rawChannelId = resolveChannelId(_hookCtx, event.to);
        const { ctx } = getOrCreateContext(rawChannelId, undefined, "message_sending");
        ctx.lastOutput = event.content;
      }
    });
  }

  // -- Hook: message_sent -------------------------------------------------
  if (shouldHookEnabled("message_sent")) {
    api.on("message_sent", async (event: any, hookCtx: any) => {
      if (event.content && event.success) {
        if (lastUserTraceContext) {
          lastUserTraceContext.lastOutput = event.content;
        } else {
          const rawChannelId = resolveChannelId(hookCtx, event.to);
          const { ctx } = getOrCreateContext(rawChannelId, undefined, "message_sent");
          ctx.lastOutput = event.content;
        }
      }
    });
  }

  // -- Hook: llm_input ----------------------------------------------------
  if (shouldHookEnabled("llm_input")) {
    api.on("llm_input", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx } = getOrCreateContext(rawChannelId, event.runId, "llm_input");

      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
      }

      // Update trace with sessionId and userId when available
      const rawLlmUserId = hookCtx.trigger || event.from || event.metadata?.senderId || undefined;
      const userId = config.userId && rawLlmUserId ? `${config.userId}/${rawLlmUserId}` : config.userId || rawLlmUserId;
      exporter.updateTrace(ctx.traceId, {
        ...(ctx.sessionId ? { sessionId: ctx.sessionId } : {}),
        ...(userId ? { userId: String(userId) } : {}),
      });

      if (!ctx.userInput && event.prompt) {
        ctx.userInput = event.prompt;
      }

      ctx.llmStartTime = Date.now();
      ctx.llmSpanId = generateId(16);

      if (event.systemPrompt) {
        ctx.llmSystemInstructions = formatSystemInstructions(event.systemPrompt);
      }

      const historyMsgs = event.historyMessages?.length
        ? event.historyMessages.map((msg: any) => safeClone(msg))
        : [];

      ctx.llmInputMessages = formatInputMessages(historyMsgs, event.prompt);

      lastLlmSystemInstructions = ctx.llmSystemInstructions;
      lastLlmInputMessages = ctx.llmInputMessages;
      lastLlmStartTime = ctx.llmStartTime;
      lastLlmSpanId = ctx.llmSpanId;

      // Store per-runId for reliable lookup in llm_output
      if (event.runId) {
        llmTimingByRunId.set(event.runId, {
          startTime: ctx.llmStartTime,
          spanId: ctx.llmSpanId,
          systemInstructions: ctx.llmSystemInstructions,
          inputMessages: ctx.llmInputMessages,
        });
      }

      if (config.debug) {
        api.logger.info(`[Litefuse] LLM input started: ${event.provider}/${event.model}, runId=${event.runId}`);
      }
    });
  }

  // -- Hook: llm_output ---------------------------------------------------
  if (shouldHookEnabled("llm_output")) {
    api.on("llm_output", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, event.runId, "llm_output");

      if (event.sessionId) {
        ctx.sessionId = event.sessionId;
      }

      const now = Date.now();

      // Lookup LLM timing: ctx → global → per-runId map → fallback to now
      const runTiming = event.runId ? llmTimingByRunId.get(event.runId) : undefined;
      const startTime = ctx.llmStartTime || lastLlmStartTime || runTiming?.startTime || now;

      // Extract output text: prefer assistantTexts, fallback to lastAssistant.content
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

        if (lastUserTraceContext) {
          lastUserTraceContext.lastOutput = outputText;
        }
      }

      const systemInstructions = ctx.llmSystemInstructions || lastLlmSystemInstructions || runTiming?.systemInstructions;
      const inputMessages = ctx.llmInputMessages || lastLlmInputMessages || runTiming?.inputMessages;
      const llmSpanId = ctx.llmSpanId || lastLlmSpanId || runTiming?.spanId;

      // event.usage = cumulative across all LLM calls (from getUsageTotals)
      // lastAssistant.usage = per-call from API response (may be all zeros)
      // Prefer event.usage as it has actual token counts when available
      const lastAssistantUsage = (event.lastAssistant as any)?.usage;
      const inputTokens = event.usage?.input ?? lastAssistantUsage?.input ?? 0;
      const outputTokens = event.usage?.output ?? lastAssistantUsage?.output ?? 0;
      const cacheReadTokens = event.usage?.cacheRead ?? lastAssistantUsage?.cacheRead ?? 0;
      const cacheCreationTokens = event.usage?.cacheWrite ?? lastAssistantUsage?.cacheWrite ?? 0;

      const lastAssistantObj = event.lastAssistant as any;
      const stopReason =
        typeof lastAssistantObj?.stopReason === "string" ? lastAssistantObj.stopReason : undefined;

      const llmAttrs: Record<string, any> = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": event.provider,
        "gen_ai.request.model": event.model,
        "gen_ai.response.model": event.model,
        "gen_ai.usage.input_tokens": inputTokens,
        "gen_ai.usage.output_tokens": outputTokens,
        "gen_ai.usage.total_tokens": inputTokens + outputTokens,
        "gen_ai.usage.cache_read.input_tokens": cacheReadTokens,
        "gen_ai.usage.cache_creation.input_tokens": cacheCreationTokens,
      };

      if (stopReason) {
        llmAttrs["gen_ai.response.finish_reasons"] = JSON.stringify([stopReason]);
      }

      if (systemInstructions) {
        llmAttrs["gen_ai.system_instructions"] = systemInstructions;
      }

      if (inputMessages) {
        llmAttrs["gen_ai.input.messages"] = inputMessages;
      }

      if (outputTexts.length) {
        llmAttrs["gen_ai.output.messages"] = formatOutputMessages(outputTexts, stopReason || "stop");
      }

      const span = createSpan(ctx, channelId, `chat ${event.model}`, "model", startTime, now, llmAttrs);

      if (llmSpanId) span.spanId = llmSpanId;

      ctx.llmStartTime = undefined;
      ctx.llmSpanId = undefined;
      ctx.llmSystemInstructions = undefined;
      ctx.llmInputMessages = undefined;

      lastLlmSystemInstructions = undefined;
      lastLlmInputMessages = undefined;
      lastLlmStartTime = undefined;
      lastLlmSpanId = undefined;
      if (event.runId) llmTimingByRunId.delete(event.runId);

      await exporter.export(span);

      if (config.debug) {
        api.logger.info(
          `[Litefuse] Exported LLM span: ${event.provider}/${event.model}, duration=${now - startTime}ms`
        );
      }
    });
  }

  // -- Hook: before_tool_call ---------------------------------------------
  if (shouldHookEnabled("before_tool_call")) {
    api.on("before_tool_call", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "before_tool_call");

      pendingToolCall = {
        toolName: event.toolName,
        toolCallId: `call_${generateId(12)}`,
        toolSpanId: generateId(16),
        toolStartTime: Date.now(),
        toolInput: event.params,
        traceContext: ctx,
        channelId,
      };

      if (config.debug) {
        api.logger.info(`[Litefuse] Tool call started: ${event.toolName}, spanId=${pendingToolCall.toolSpanId}`);
      }
    });
  }

  // -- Hook: after_tool_call ----------------------------------------------
  if (shouldHookEnabled("after_tool_call")) {
    api.on("after_tool_call", async (event: any, _hookCtx: any) => {
      if (!pendingToolCall || pendingToolCall.toolName !== event.toolName) {
        return;
      }

      const { toolName, toolCallId, toolSpanId, toolStartTime, toolInput, traceContext, channelId } =
        pendingToolCall;

      pendingToolCall = undefined;

      const now = Date.now();

      const toolAttrs: Record<string, any> = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": toolName,
        "gen_ai.tool.call.id": toolCallId,
        "gen_ai.tool.type": "function",
        "tool.duration_ms": event.durationMs || now - toolStartTime,
      };

      if (toolInput !== undefined) {
        toolAttrs["gen_ai.tool.call.arguments"] = truncateAttr(
          typeof toolInput === "string" ? toolInput : JSON.stringify(toolInput)
        );
      }

      if (event.error) {
        toolAttrs["error.type"] = event.error;
      } else if (event.result !== undefined) {
        toolAttrs["gen_ai.tool.call.result"] = truncateAttr(
          typeof event.result === "string" ? event.result : JSON.stringify(event.result)
        );
      }

      const span = createSpan(
        traceContext,
        channelId,
        `execute_tool ${toolName}`,
        "tool",
        toolStartTime,
        now,
        toolAttrs
      );

      span.spanId = toolSpanId;

      await exporter.export(span);

      if (config.debug) {
        api.logger.info(
          `[Litefuse] Exported tool span: ${toolName}, duration=${now - toolStartTime}ms`
        );
      }
    });
  }

  // -- Hook: before_agent_start -------------------------------------------
  if (shouldHookEnabled("before_agent_start")) {
    api.on("before_agent_start", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const agentId = hookCtx.agentId || event.agentId || "openclaw";
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "before_agent_start");

      await ensureEntrySpan(ctx, channelId, {
        userId: hookCtx.trigger || "system",
        role: hookCtx.trigger || "system",
        from: agentId,
      });

      if (ctx.agentSpanId) return;

      const now = Date.now();
      ctx.agentStartTime = now;
      ctx.agentSpanId = generateId(16);

      const agentInput = ctx.userInput || lastUserTraceContext?.userInput;

      const spanData = {
        name: `invoke_agent ${agentId}`,
        type: "agent",
        startTime: now,
        attributes: {
          "gen_ai.operation.name": "invoke_agent",
          "gen_ai.provider.name": "openclaw",
          "gen_ai.agent.id": agentId,
          "gen_ai.agent.name": agentId,
          "openclaw.session.id": ctx.sessionId || channelId,
          "gen_ai.session.id": ctx.sessionId || channelId,
          "openclaw.run.id": ctx.runId,
          "openclaw.turn.id": ctx.turnId,
          "openclaw.version": openclawVersion,
        },
        input: agentInput,
        traceId: ctx.traceId,
        spanId: ctx.agentSpanId,
        parentSpanId: ctx.rootSpanId,
      };

      await exporter.startSpan(spanData, ctx.agentSpanId);

      if (config.debug) {
        api.logger.info(`[Litefuse] Started agent span: ${agentId}, spanId=${ctx.agentSpanId}`);
      }
    });
  }

  // -- Hook: agent_end ----------------------------------------------------
  if (shouldHookEnabled("agent_end")) {
    api.on("agent_end", async (event: any, hookCtx: any) => {
      const rawChannelId = resolveChannelId(hookCtx);
      const { ctx, channelId } = getOrCreateContext(rawChannelId, undefined, "agent_end");

      const now = Date.now();

      const pendingAgentSpanId = ctx.agentSpanId;
      const agentEndTime = now;

      let agentEndAttrs: Record<string, any> | undefined;

      if (pendingAgentSpanId) {
        agentEndAttrs = {
          "agent.duration_ms": event.durationMs || 0,
          "agent.message_count": Array.isArray(event.messages) ? event.messages.length : 0,
          "agent.success": event.success ?? true,
          ...(event.error ? { "agent.error": event.error } : {}),
        };

        if (ctx.sessionId) {
          agentEndAttrs["openclaw.session.id"] = ctx.sessionId || channelId;
          agentEndAttrs["gen_ai.session.id"] = ctx.sessionId || channelId;
        }

        ctx.agentSpanId = undefined;
        ctx.agentStartTime = undefined;
      }

      // Snapshot references — do NOT clear global pointers yet.
      // llm_output may fire after agent_end and still needs lastUserTraceContext
      // to link to the correct context.
      const savedLastUserTraceContext = lastUserTraceContext;
      const savedLastUserChannelId = lastUserChannelId;
      const originalChannelId = ctx.originalChannelId || savedLastUserChannelId || channelId;

      const rootCtx = savedLastUserTraceContext || ctx;

      if (rootCtx.rootSpanStartTime || pendingAgentSpanId) {
        const rootSpanId = rootCtx.rootSpanId;
        const rootSpanStartTime = rootCtx.rootSpanStartTime;
        const userInput = rootCtx.userInput;
        const traceId = rootCtx.traceId;
        const resolvedSessionId = ctx.sessionId || rootCtx.sessionId;

        setTimeout(async () => {
          const finalOutput = ctx.lastOutput || rootCtx.lastOutput;

          // Update trace input/output/sessionId
          // NOTE: sessionId must be persisted on the trace itself so it appears
          // in Doris traces.session_id, otherwise the trace is excluded from the
          // sessions list (WHERE t.session_id IS NOT NULL).
          const traceUpdates: Record<string, any> = {};
          if (userInput) traceUpdates.input = userInput;
          if (finalOutput) traceUpdates.output = finalOutput;
          if (resolvedSessionId) traceUpdates.sessionId = resolvedSessionId;
          if (Object.keys(traceUpdates).length > 0) {
            exporter.updateTrace(rootCtx.traceId, traceUpdates);
          }

          // End agent span
          if (pendingAgentSpanId && agentEndAttrs) {
            if (userInput) {
              agentEndAttrs["gen_ai.input.messages"] = truncateAttr(
                JSON.stringify([{ role: "user", parts: [{ type: "text", content: String(userInput) }] }])
              );
            }
            if (finalOutput) {
              agentEndAttrs["gen_ai.output.messages"] = formatOutputMessages([
                typeof finalOutput === "string" ? finalOutput : JSON.stringify(finalOutput),
              ]);
            }

            exporter.endSpanById(pendingAgentSpanId, agentEndTime, agentEndAttrs, finalOutput, userInput);

            if (config.debug) {
              api.logger.info(
                `[Litefuse] Ended agent span: spanId=${pendingAgentSpanId}, duration=${event.durationMs}ms`
              );
            }
          }

          // End root span
          if (rootSpanStartTime) {
            const endTime = Date.now();
            const rootEndAttrs: Record<string, any> = {
              "request.duration_ms": endTime - rootSpanStartTime,
            };

            if (resolvedSessionId) {
              rootEndAttrs["openclaw.session.id"] = resolvedSessionId;
              rootEndAttrs["gen_ai.session.id"] = resolvedSessionId;
            }

            if (userInput) {
              rootEndAttrs["gen_ai.input.messages"] = truncateAttr(
                JSON.stringify([{ role: "user", parts: [{ type: "text", content: String(userInput) }] }])
              );
            }

            if (finalOutput) {
              rootEndAttrs["gen_ai.output.messages"] = formatOutputMessages([
                typeof finalOutput === "string" ? finalOutput : JSON.stringify(finalOutput),
              ]);
            }

            exporter.endSpanById(rootSpanId, endTime, rootEndAttrs, finalOutput, userInput);

            if (config.debug) {
              api.logger.info(
                `[Litefuse] Ended root span: spanId=${rootSpanId}, duration=${endTime - rootSpanStartTime}ms, traceId=${traceId}`
              );
            }
          }

          // Clear global pointers (deferred to allow llm_output to use them)
          lastUserChannelId = undefined;
          lastUserTraceContext = undefined;

          // Clean up Map entries
          if (savedLastUserChannelId) endTurn(savedLastUserChannelId);
          if (originalChannelId && originalChannelId !== savedLastUserChannelId) {
            endTurn(originalChannelId);
          }
          if (rawChannelId !== originalChannelId && rawChannelId !== savedLastUserChannelId) {
            contextByChannelId.delete(rawChannelId);
          }

          // Wait for SDK batch queue to process end() events before flushing
          await new Promise(r => setTimeout(r, 500));
          await exporter.flush();
          exporter.endTrace();
        }, 200);
      } else {
        lastUserChannelId = undefined;
        lastUserTraceContext = undefined;
        if (savedLastUserChannelId) endTurn(savedLastUserChannelId);
        if (originalChannelId && originalChannelId !== savedLastUserChannelId) {
          endTurn(originalChannelId);
        }
        if (rawChannelId !== originalChannelId && rawChannelId !== savedLastUserChannelId) {
          contextByChannelId.delete(rawChannelId);
        }
        await exporter.flush();
        exporter.endTrace();
      }
    });
  }

  api.logger.info(
    `[Litefuse] Plugin activated (baseUrl: ${config.baseUrl || "litefuse.cloud"})`
  );
}

// ---------------------------------------------------------------------------
// Plugin definition — this is the object openclaw's plugin loader expects
// ---------------------------------------------------------------------------

const plugin = {
  id: "openclaw-litefuse-plugin",
  name: "OpenClaw Litefuse Plugin",
  version: "0.2.0",
  description: "Report OpenClaw AI agent execution traces to Litefuse",

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
