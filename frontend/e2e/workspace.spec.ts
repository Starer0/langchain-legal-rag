import { test, expect, type Page } from '@playwright/test';

async function fixture(page: Page, initialHistory: boolean = false) {
  let username: string | null = null;
  let sequence = 0;
  let calls = 0;
  let streamFailure = false;
  const catalogs: Record<string, { id: string; title: string; messages: { role: string; content: string }[] }[]> = {
    demo_ab: initialHistory ? [{ id: 'long-history', title: '旧聊天', messages: Array.from({ length: 40 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `第 ${i + 1} 条消息：` + '这是一段用于验证阅读滚动的法律资料。'.repeat(8) })) }] : [],
    demo_c: [],
  };
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url()).pathname;
    const method = request.method();
    const reply = (data: unknown, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(data) });
    const identity = () => ({ user: { id: username, username, level_id: 'fixture' }, csrf_token: 'fixture-csrf' });
    if (url === '/api/auth/me') return username ? reply(identity()) : reply({ detail: '请登录' }, 401);
    if (url === '/api/auth/login') { username = request.postDataJSON().username; return reply(identity()); }
    if (url === '/api/auth/logout') { username = null; return route.fulfill({ status: 204 }); }
    if (!username) return reply({ detail: '请登录' }, 401);
    if (method !== 'GET') expect(request.headers()['x-csrf-token']).toBe('fixture-csrf');
    const conversations = catalogs[username];
    if (url === '/api/conversations' && method === 'GET') return reply({ conversations: conversations.map(({ id, title }) => ({ id, title })) });
    if (url === '/api/conversations' && method === 'POST') {
      const conversation = { id: `fixture-${++sequence}`, title: '新对话', messages: [] };
      conversations.unshift(conversation);
      return reply(conversation, 201);
    }
    const [, id, suffix] = url.match(/\/api\/conversations\/([^/]+)(?:\/(messages|chat))?/) || [];
    const conversation = conversations.find(item => item.id === id);
    if (!conversation) return reply({ detail: '对话不存在' }, 404);
    if (suffix === 'messages') return reply({ messages: conversation.messages });
    if (suffix === 'chat') {
      calls++;
      const question = request.postDataJSON().question;
      const answer = '**试用期规定**\n\n> 根据当前可访问资料回答。\n\n最长六个月。';
      const sources = [{ law_name: '劳动合同法', article: '第十九条', pages: [4], content: '试用期最长不得超过六个月。' }];
      if (!streamFailure) {
        conversation.messages.push({ role: 'user', content: question }, { role: 'assistant', content: answer });
        conversation.title = question.slice(0, 80);
      }
      return route.fulfill({ status: 200, headers: { 'Content-Type': 'text/event-stream', 'X-Request-ID': 'fixture-request' }, body:
        `event: status\ndata: {"stage":"retrieve"}\n\nevent: delta\ndata: ${JSON.stringify({ text: answer })}\n\n` +
        (streamFailure ? '' : `event: done\ndata: ${JSON.stringify({ answer, sources })}\n\n`) });
    }
    if (method === 'PATCH') { conversation.title = request.postDataJSON().title; return reply(conversation); }
    if (method === 'DELETE') { conversations.splice(conversations.indexOf(conversation), 1); return route.fulfill({ status: 204 }); }
    return reply({}, 404);
  });
  return { calls: () => calls, failStream: () => { streamFailure = true; } };
}

async function login(page: Page, username = 'demo_ab') {
  await page.getByRole('textbox', { name: '用户名', exact: true }).fill(username);
  await page.getByLabel('密码', { exact: true }).fill('fixture-only');
  await page.getByRole('button', { name: '登录', exact: true }).click();
  await expect(page.getByRole('textbox', { name: '法律问题' })).toBeEnabled();
}

test('short mobile login viewport can scroll to the submit button', async ({ page }) => {
  await fixture(page);
  await page.setViewportSize({ width: 375, height: 400 });
  await page.goto('/');
  await page.mouse.move(180, 220);
  await page.mouse.wheel(0, 600);
  await expect.poll(() => page.locator('.login-page').evaluate(element => element.scrollTop)).toBeGreaterThan(0);
  await page.getByLabel('密码', { exact: true }).fill('fixture-only');
  const button = page.getByRole('button', { name: '登录', exact: true });
  await button.scrollIntoViewIfNeeded();
  const box = await button.boundingBox();
  expect(box!.y + box!.height).toBeLessThanOrEqual(400);
  await page.getByRole('textbox', { name: '用户名', exact: true }).fill('demo_ab');
  await button.click();
  await expect(page.getByRole('textbox', { name: '法律问题' })).toBeEnabled();
});

test('desktop login/chat/sources/rename/delete and account isolation', async ({ page }) => {
  const data = await fixture(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await page.screenshot({ path: 'test-results/login-desktop.png' });
  await login(page);
  await page.getByRole('button', { name: '试用期最长可以约定多久？' }).click();
  expect(data.calls()).toBe(0);
  const longTitle = '请仅依据劳动合同法第二十条说明试用期工资的规定，并解释具体适用条件。'.repeat(3);
  await page.getByRole('textbox', { name: '法律问题' }).fill(longTitle);
  await page.getByRole('button', { name: '发送问题' }).click();
  await expect(page.getByText('最长六个月。', { exact: true })).toBeVisible();
  await page.getByText('参考来源 · 1', { exact: true }).click();
  await expect(page.getByText('试用期最长不得超过六个月。', { exact: true })).toBeVisible();
  await expect(page.locator('h1')).toHaveText(longTitle.slice(0, 80));
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/chat-desktop.png' });
  await page.getByRole('button', { name: `重命名 ${longTitle.slice(0, 80)}`, exact: true }).click();
  await page.getByLabel('对话名称').fill('工资法规咨询');
  await page.getByRole('button', { name: '保存名称' }).click();
  await expect(page.locator('h1')).toHaveText('工资法规咨询');
  await page.getByRole('button', { name: '新建对话', exact: false }).click();
  await expect(page.locator('h1')).toHaveText('新对话');
  await page.getByRole('button', { name: '删除 新对话', exact: true }).click();
  await page.getByRole('button', { name: '确认删除' }).click();
  await expect(page.locator('h1')).toHaveText('工资法规咨询');
  await page.getByRole('button', { name: '退出登录', exact: true }).click();
  await expect(page.getByRole('heading', { name: '登录工作空间' })).toBeVisible();
  await login(page, 'demo_c');
  await expect(page.getByText('工资法规咨询', { exact: true })).toHaveCount(0);
  await expect(page.getByText('试用期规定', { exact: true })).toHaveCount(0);
});

test('mobile composer stays visible, drawer works and IME does not submit', async ({ page }) => {
  const data = await fixture(page);
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  await login(page);
  await page.getByRole('button', { name: '打开对话列表' }).click();
  await expect(page.getByRole('dialog', { name: '对话列表' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: '打开对话列表' })).toBeFocused();
  const input = page.getByRole('textbox', { name: '法律问题' });
  await input.fill('中文输入的问题');
  await input.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', isComposing: true });
  expect(data.calls()).toBe(0);
  await input.press('Enter');
  await expect(page.getByText('最长六个月。', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '发送问题' })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/chat-mobile.png' });
});

test('failed stream restores question and removes partial answer', async ({ page }) => {
  const data = await fixture(page);
  await page.goto('/');
  await login(page);
  data.failStream();
  await page.getByRole('textbox', { name: '法律问题' }).fill('保留这条问题');
  await page.getByRole('button', { name: '发送问题' }).click();
  await expect(page.getByRole('alert')).toContainText('fixture-request');
  await expect(page.getByRole('textbox', { name: '法律问题' })).toHaveValue('保留这条问题');
  await expect(page.getByText('最长六个月。', { exact: true })).toHaveCount(0);
});

test('long history scrolls inside pane with composer anchored', async ({ page }) => {
  await fixture(page, true);
  await page.setViewportSize({ width: 768, height: 900 });
  await page.goto('/');
  await login(page);
  const pane = page.locator('.message-scroll');
  expect(await pane.evaluate(e => e.scrollHeight > e.clientHeight)).toBe(true);
  await pane.evaluate(e => { e.scrollTop = 0; e.dispatchEvent(new Event('scroll')); });
  await expect(page.getByRole('button', { name: '发送问题' })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/chat-tablet.png' });
});
