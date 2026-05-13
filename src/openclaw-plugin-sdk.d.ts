/**
 * Type declarations for openclaw/plugin-sdk.
 * This module is provided by the openclaw runtime at load time.
 */
declare module "openclaw/plugin-sdk" {
  export interface OpenClawPluginApi {
    id: string;
    name: string;
    version?: string;
    description?: string;
    source: string;
    config: Record<string, any>;
    pluginConfig?: Record<string, unknown>;
    runtime: { version?: string; [key: string]: any };
    logger: {
      info: (msg: string) => void;
      error: (msg: string) => void;
      warn: (msg: string) => void;
      debug?: (msg: string) => void;
    };
    on: (hookName: string, handler: (...args: any[]) => any, opts?: { priority?: number }) => void;
    [key: string]: any;
  }
}
