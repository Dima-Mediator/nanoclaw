/**
 * Host Runner for NanoClaw
 * Runs the agent directly on the host using the Claude Agent SDK.
 * No container isolation — full filesystem access.
 */
import { ChildProcess } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  query,
  HookCallback,
  PreCompactHookInput,
} from '@anthropic-ai/claude-agent-sdk';

import {
  AGENT_CWD,
  CONTAINER_TIMEOUT,
  DATA_DIR,
  GROUPS_DIR,
  IDLE_TIMEOUT,
} from './config.js';
import { readEnvFile } from './env.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { logger } from './logger.js';
import { validateAdditionalMounts } from './mount-security.js';
import { RegisteredGroup } from './types.js';
import { ContainerInput, ContainerOutput } from './container-runner.js';

const IPC_POLL_MS = 500;

// ---------------------------------------------------------------------------
// MessageStream — push-based async iterable for feeding messages to query()
// ---------------------------------------------------------------------------

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

// ---------------------------------------------------------------------------
// IPC helpers — same filesystem-based approach, using real host paths
// ---------------------------------------------------------------------------

function getIpcInputDir(groupFolder: string): string {
  return path.join(resolveGroupIpcPath(groupFolder), 'input');
}

function getCloseSentinelPath(groupFolder: string): string {
  return path.join(getIpcInputDir(groupFolder), '_close');
}

function shouldClose(groupFolder: string): boolean {
  const sentinel = getCloseSentinelPath(groupFolder);
  if (fs.existsSync(sentinel)) {
    try {
      fs.unlinkSync(sentinel);
    } catch {
      /* ignore */
    }
    return true;
  }
  return false;
}

function drainIpcInput(groupFolder: string): string[] {
  const inputDir = getIpcInputDir(groupFolder);
  try {
    fs.mkdirSync(inputDir, { recursive: true });
    const files = fs
      .readdirSync(inputDir)
      .filter((f) => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(inputDir, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        logger.warn(
          { file, err },
          'Host runner: failed to process IPC input file',
        );
        try {
          fs.unlinkSync(filePath);
        } catch {
          /* ignore */
        }
      }
    }
    return messages;
  } catch (err) {
    logger.warn({ err }, 'Host runner: IPC drain error');
    return [];
  }
}

function waitForIpcMessage(groupFolder: string): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose(groupFolder)) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput(groupFolder);
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

// ---------------------------------------------------------------------------
// Pre-compact hook — archive transcript before compaction
// ---------------------------------------------------------------------------

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text =
          typeof entry.message.content === 'string'
            ? entry.message.content
            : entry.message.content
                .map((c: { text?: string }) => c.text || '')
                .join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
      /* skip malformed lines */
    }
  }
  return messages;
}

function formatTranscriptMarkdown(
  messages: ParsedMessage[],
  title?: string | null,
  assistantName?: string,
): string {
  const now = new Date();
  const formatDateTime = (d: Date) =>
    d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : assistantName || 'Assistant';
    const content =
      msg.content.length > 2000
        ? msg.content.slice(0, 2000) + '...'
        : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

function getSessionSummary(
  sessionId: string,
  transcriptPath: string,
): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) return null;

  try {
    const index: SessionsIndex = JSON.parse(
      fs.readFileSync(indexPath, 'utf-8'),
    );
    const entry = index.entries.find((e) => e.sessionId === sessionId);
    if (entry?.summary) return entry.summary;
  } catch {
    /* ignore */
  }

  return null;
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

function createPreCompactHook(
  groupDir: string,
  assistantName?: string,
): HookCallback {
  return async (input) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) return {};

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);
      if (messages.length === 0) return {};

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = path.join(groupDir, 'conversations');
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(
        messages,
        summary,
        assistantName,
      );
      fs.writeFileSync(filePath, markdown);

      logger.debug({ filePath }, 'Host runner: archived conversation');
    } catch (err) {
      logger.warn({ err }, 'Host runner: failed to archive transcript');
    }

    return {};
  };
}

// ---------------------------------------------------------------------------
// Directory preparation — mirrors container-runner.ts buildVolumeMounts
// ---------------------------------------------------------------------------

function prepareDirectories(
  group: RegisteredGroup,
  isMain: boolean,
): { groupDir: string; sessionsDir: string; ipcDir: string } {
  const groupDir = resolveGroupFolderPath(group.folder);
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  // Per-group Claude sessions directory
  const sessionsDir = path.join(DATA_DIR, 'sessions', group.folder, '.claude');
  fs.mkdirSync(sessionsDir, { recursive: true });

  const settingsFile = path.join(sessionsDir, 'settings.json');
  if (!fs.existsSync(settingsFile)) {
    fs.writeFileSync(
      settingsFile,
      JSON.stringify(
        {
          env: {
            CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
            CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
            CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
          },
        },
        null,
        2,
      ) + '\n',
    );
  }

  // Sync skills
  const skillsSrc = path.join(process.cwd(), 'container', 'skills');
  const skillsDst = path.join(sessionsDir, 'skills');
  if (fs.existsSync(skillsSrc)) {
    for (const skillDir of fs.readdirSync(skillsSrc)) {
      const srcDir = path.join(skillsSrc, skillDir);
      if (!fs.statSync(srcDir).isDirectory()) continue;
      const dstDir = path.join(skillsDst, skillDir);
      fs.cpSync(srcDir, dstDir, { recursive: true });
    }
  }

  // IPC directories
  const ipcDir = resolveGroupIpcPath(group.folder);
  fs.mkdirSync(path.join(ipcDir, 'messages'), { recursive: true });
  fs.mkdirSync(path.join(ipcDir, 'tasks'), { recursive: true });
  fs.mkdirSync(path.join(ipcDir, 'input'), { recursive: true });
  fs.mkdirSync(path.join(ipcDir, 'queries'), { recursive: true });

  return { groupDir, sessionsDir, ipcDir };
}

// ---------------------------------------------------------------------------
// runHostAgent — same interface as runContainerAgent
// ---------------------------------------------------------------------------

export async function runHostAgent(
  group: RegisteredGroup,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const startTime = Date.now();
  const { groupDir, sessionsDir, ipcDir } = prepareDirectories(
    group,
    input.isMain,
  );

  const safeName = group.folder.replace(/[^a-zA-Z0-9-]/g, '-');
  const runName = `nanoclaw-host-${safeName}-${Date.now()}`;

  logger.info(
    { group: group.name, runName, isMain: input.isMain },
    'Starting host agent',
  );

  // Register a dummy process for queue compatibility
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onProcess(null as any, runName);

  // Read credentials from .env
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'GH_TOKEN',
  ]);

  // Build SDK environment
  const sdkEnv: Record<string, string | undefined> = {
    ...process.env,
    HOME: path.dirname(sessionsDir), // sessions/{folder}/ — SDK finds .claude/ here
    GH_CONFIG_DIR:
      process.env.GH_CONFIG_DIR || path.join(os.homedir(), '.config', 'gh'),
    TZ: process.env.TZ || Intl.DateTimeFormat().resolvedOptions().timeZone,
    NANOCLAW_IS_MAIN: input.isMain ? '1' : '0',
  };

  // Inject credentials into SDK env
  if (secrets.ANTHROPIC_API_KEY) {
    sdkEnv.ANTHROPIC_API_KEY = secrets.ANTHROPIC_API_KEY;
  }
  if (secrets.CLAUDE_CODE_OAUTH_TOKEN) {
    sdkEnv.CLAUDE_CODE_OAUTH_TOKEN = secrets.CLAUDE_CODE_OAUTH_TOKEN;
  }
  if (secrets.ANTHROPIC_AUTH_TOKEN) {
    sdkEnv.ANTHROPIC_AUTH_TOKEN = secrets.ANTHROPIC_AUTH_TOKEN;
  }
  if (secrets.GH_TOKEN) {
    sdkEnv.GH_TOKEN = secrets.GH_TOKEN;
  }

  // MCP server — run the TypeScript source via tsx at runtime
  const mcpServerPath = path.join(
    process.cwd(),
    'container',
    'agent-runner',
    'src',
    'ipc-mcp-stdio.ts',
  );

  // Validate AGENT_CWD is accessible before using it — macOS TCC restrictions
  // can cause ~/Documents/ to hang indefinitely for launchd services
  let effectiveCwd = groupDir;
  if (AGENT_CWD) {
    try {
      fs.readdirSync(AGENT_CWD);
      effectiveCwd = AGENT_CWD;
    } catch (err) {
      logger.warn(
        { cwd: AGENT_CWD, err },
        'AGENT_CWD inaccessible, falling back to group dir',
      );
    }
  }

  // Additional directories (from mount config)
  const additionalDirs: string[] = [];
  if (!input.isMain) {
    const globalDir = path.join(GROUPS_DIR, 'global');
    if (fs.existsSync(globalDir)) {
      additionalDirs.push(globalDir);
    }
  }

  // When AGENT_CWD overrides cwd, add the group dir as an additional directory
  // so the agent still picks up the group's CLAUDE.md and context
  if (effectiveCwd !== groupDir) {
    additionalDirs.push(groupDir);
  }

  if (group.containerConfig?.additionalMounts) {
    const validatedMounts = validateAdditionalMounts(
      group.containerConfig.additionalMounts,
      group.name,
      input.isMain,
    );
    for (const mount of validatedMounts) {
      additionalDirs.push(mount.hostPath);
    }
  }

  // Global CLAUDE.md for non-main groups
  let globalClaudeMd: string | undefined;
  if (!input.isMain) {
    const globalClaudeMdPath = path.join(GROUPS_DIR, 'global', 'CLAUDE.md');
    if (fs.existsSync(globalClaudeMdPath)) {
      globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
    }
  }

  // Clean up stale _close sentinel
  try {
    fs.unlinkSync(getCloseSentinelPath(group.folder));
  } catch {
    /* ignore */
  }

  // Build initial prompt
  let prompt = input.prompt;
  if (input.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput(group.folder);
  if (pending.length > 0) {
    prompt += '\n' + pending.join('\n');
  }

  let sessionId = input.sessionId;
  let newSessionId: string | undefined;
  let resumeAt: string | undefined;
  let outputChain = Promise.resolve();

  // Timeout handling
  let timedOut = false;
  let hadOutput = false;
  const configTimeout = group.containerConfig?.timeout || CONTAINER_TIMEOUT;
  const timeoutMs = Math.max(configTimeout, IDLE_TIMEOUT + 30_000);

  const killController = new AbortController();

  const killOnTimeout = () => {
    timedOut = true;
    logger.error({ group: group.name, runName }, 'Host agent timeout');
    // Write close sentinel to end the query loop
    try {
      const inputDir = getIpcInputDir(group.folder);
      fs.mkdirSync(inputDir, { recursive: true });
      fs.writeFileSync(path.join(inputDir, '_close'), '');
    } catch {
      /* ignore */
    }
    // Abort the SDK query — this kills the CLI subprocess and stops iteration
    killController.abort();
  };

  let timeout = setTimeout(killOnTimeout, timeoutMs);

  const resetTimeout = () => {
    clearTimeout(timeout);
    timeout = setTimeout(killOnTimeout, timeoutMs);
  };

  // Inner query function
  async function runQuery(queryPrompt: string): Promise<{
    lastAssistantUuid?: string;
    closedDuringQuery: boolean;
  }> {
    const stream = new MessageStream();
    stream.push(queryPrompt);

    let ipcPolling = true;
    let closedDuringQuery = false;

    const pollIpc = () => {
      if (!ipcPolling) return;
      if (shouldClose(group.folder)) {
        closedDuringQuery = true;
        stream.end();
        ipcPolling = false;
        return;
      }
      const messages = drainIpcInput(group.folder);
      for (const text of messages) {
        stream.push(text);
      }
      setTimeout(pollIpc, IPC_POLL_MS);
    };
    setTimeout(pollIpc, IPC_POLL_MS);

    let lastAssistantUuid: string | undefined;

    for await (const message of query({
      prompt: stream,
      options: {
        abortController: killController,
        cwd: effectiveCwd,
        additionalDirectories:
          additionalDirs.length > 0 ? additionalDirs : undefined,
        resume: sessionId,
        resumeSessionAt: resumeAt,
        systemPrompt: globalClaudeMd
          ? {
              type: 'preset' as const,
              preset: 'claude_code' as const,
              append: globalClaudeMd,
            }
          : undefined,
        allowedTools: [
          'Bash',
          'Read',
          'Write',
          'Edit',
          'Glob',
          'Grep',
          'WebSearch',
          'WebFetch',
          'Task',
          'TaskOutput',
          'TaskStop',
          'TeamCreate',
          'TeamDelete',
          'SendMessage',
          'TodoWrite',
          'ToolSearch',
          'Skill',
          'NotebookEdit',
          'mcp__nanoclaw__*',
        ],
        env: sdkEnv,
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        settingSources: ['project', 'user'],
        mcpServers: {
          nanoclaw: {
            command: 'npx',
            args: ['tsx', mcpServerPath],
            env: {
              NANOCLAW_CHAT_JID: input.chatJid,
              NANOCLAW_CHAT_NAME: group.name,
              NANOCLAW_GROUP_FOLDER: input.groupFolder,
              NANOCLAW_IS_MAIN: input.isMain ? '1' : '0',
              NANOCLAW_IPC_DIR: ipcDir,
            },
          },
        },
        hooks: {
          PreCompact: [
            {
              hooks: [createPreCompactHook(groupDir, input.assistantName)],
            },
          ],
        },
      },
    })) {
      if (message.type === 'assistant' && 'uuid' in message) {
        lastAssistantUuid = (message as { uuid: string }).uuid;
      }

      if (message.type === 'system' && message.subtype === 'init') {
        newSessionId = message.session_id;
        sessionId = newSessionId;
      }

      if (message.type === 'result') {
        const textResult =
          'result' in message ? (message as { result?: string }).result : null;

        const output: ContainerOutput = {
          status: 'success',
          result: textResult || null,
          newSessionId,
        };

        hadOutput = true;
        resetTimeout();

        if (onOutput) {
          outputChain = outputChain.then(() => onOutput(output));
        }
      }
    }

    ipcPolling = false;
    return { lastAssistantUuid, closedDuringQuery };
  }

  // Query loop
  try {
    while (!timedOut) {
      logger.debug(
        {
          group: group.name,
          sessionId: sessionId || 'new',
          resumeAt: resumeAt || 'latest',
        },
        'Host agent starting query',
      );

      const queryResult = await runQuery(prompt);
      if (queryResult.lastAssistantUuid) {
        resumeAt = queryResult.lastAssistantUuid;
      }

      if (queryResult.closedDuringQuery) {
        logger.debug(
          { group: group.name },
          'Close sentinel during query, exiting',
        );
        break;
      }

      // Emit session update
      if (onOutput) {
        const sessionUpdate: ContainerOutput = {
          status: 'success',
          result: null,
          newSessionId,
        };
        outputChain = outputChain.then(() => onOutput(sessionUpdate));
      }

      // Wait for next IPC message or close
      const nextMessage = await waitForIpcMessage(group.folder);
      if (nextMessage === null) {
        logger.debug({ group: group.name }, 'Close sentinel received, exiting');
        break;
      }

      prompt = nextMessage;
    }
  } catch (err) {
    clearTimeout(timeout);
    const errorMessage = err instanceof Error ? err.message : String(err);
    logger.error(
      { group: group.name, error: errorMessage },
      'Host agent error',
    );

    return {
      status: 'error',
      result: null,
      newSessionId,
      error: errorMessage,
    };
  }

  clearTimeout(timeout);
  const duration = Date.now() - startTime;

  if (timedOut && !hadOutput) {
    return {
      status: 'error',
      result: null,
      error: `Host agent timed out after ${configTimeout}ms`,
    };
  }

  // Wait for output chain to settle
  await outputChain;

  logger.info(
    { group: group.name, duration, newSessionId },
    'Host agent completed',
  );

  return {
    status: 'success',
    result: null,
    newSessionId,
  };
}
