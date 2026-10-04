import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: { baseURL: 'http://127.0.0.1:8012', channel: 'chrome', screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: { command: 'npm exec vite preview -- --host 127.0.0.1 --port 8012 --strictPort',
               url: 'http://127.0.0.1:8012', reuseExistingServer: false, timeout: 30000 },
});
