import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:net';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

for (const width of [1440, 375]) {
  test(`server process restart resumes graph at ${width}px`, async ({ context }) => {
    test.setTimeout(60000);
    const reservation = createServer();
    await new Promise<void>(r => reservation.listen(0, '127.0.0.1', r));
    const address = reservation.address();
    if (!address || typeof address === 'string') throw Error('No port');
    const port = address.port;
    await new Promise<void>(r => reservation.close(() => r()));
    const origin = `http://127.0.0.1:${port}`;
    const root = resolve('test-results', 'recovery-' + randomUUID());
    const start = (phase: string) => spawn('D:/conda/python.exe', ['tests/fixtures/recovery_server.py', String(port), root, phase], { cwd: '..', stdio: 'ignore' });
    let child = start('initial');
    try {
      await expect.poll(async () => { try { return (await fetch(origin + '/fixture')).status; } catch { return 0; } }, { timeout: 20000 }).toBe(200);
      await context.addCookies([{ name: 'legal_rag_session', value: 'fixture', domain: '127.0.0.1', path: '/' }]);
      await context.route('**/api/**', route => route.continue({ url: origin + new URL(route.request().url()).pathname }));
      const page = await context.newPage();
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await page.getByRole('textbox', { name: '法律问题' }).fill('测试恢复的问题');
      await page.getByRole('button', { name: '发送问题' }).click();
      await expect(page.getByText('重启前的半段', { exact: true })).toBeVisible();
      const exited = once(child, 'exit');
      child.kill();
      await exited;
      child = start('resume');
      await expect.poll(async () => { try { return (await (await fetch(origin + '/fixture')).json()).phase; } catch { return ''; } }, { timeout: 20000 }).toBe('resume');
      await expect(page.getByText('重新生成的内容', { exact: true })).toBeVisible();
      await expect(page.getByText('重启前的半段', { exact: true })).toHaveCount(0);
      await expect(page.getByText('服务已恢复，正在重新生成完整回答', { exact: true })).toBeVisible();
      await page.reload();
      await expect(page.getByText('测试恢复的问题', { exact: true })).toHaveCount(1);
      await fetch(origin + '/fixture/release', { method: 'POST' });
      await expect(page.getByText('重新生成的内容，完整回答', { exact: true })).toBeVisible();
      await page.getByText('参考来源', { exact: false }).click();
      await expect(page.getByText('恢复后的来源', { exact: true })).toBeVisible();
      const data = await (await fetch(origin + '/fixture')).json();
      for (const stage of ['rewrite', 'retrieve', 'rerank']) expect(data.calls.filter((s: string) => s === stage)).toHaveLength(1);
      expect(data.calls.filter((s: string) => s === 'answer')).toHaveLength(2);
      const history = await (await fetch(origin + `/api/conversations/${data.cid}/messages`, { headers: { Cookie: 'legal_rag_session=fixture' } })).json();
      expect(history.messages).toHaveLength(2);
      await page.screenshot({ path: `test-results/recovery-${width}.png` });
    } finally { child.kill(); }
  });
}
