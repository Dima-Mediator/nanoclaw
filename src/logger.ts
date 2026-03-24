import fs from 'fs';
import path from 'path';
import pino from 'pino';

const logDir = path.resolve(process.cwd(), 'logs');
fs.mkdirSync(logDir, { recursive: true });
const logFile = path.join(logDir, 'nanoclaw.log');

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    targets: [
      { target: 'pino-pretty', options: { colorize: true }, level: 'info' },
      {
        target: 'pino/file',
        options: { destination: logFile },
        level: 'debug',
      },
    ],
  },
});

// Route uncaught errors through pino so they get timestamps in stderr
process.on('uncaughtException', (err) => {
  logger.fatal({ err }, 'Uncaught exception');
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  logger.error({ err: reason }, 'Unhandled rejection');
});
