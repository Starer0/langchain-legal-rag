const { test } = require('node:test');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const path = require('node:path');

const moduleUrl = pathToFileURL(path.join(__dirname, '../web/static/auth-client.mjs')).href;
const response = (status, data = {}) => ({ status, ok: status >= 200 && status < 300, json: async () => data, headers: { get: () => null } });

test('unmounted React client cancels pending requests without revoking server login', async () => {
  const { createAuthClient } = await import(moduleUrl);
  let signal, release;
  let signedOut = 0;
  const client = createAuthClient({ onSignedOut: () => { signedOut++; }, fetchImpl: async (url, options) => {
    if (url === '/api/auth/login') return response(200, { user: { id: 'A' }, csrf_token: 'csrf' });
    signal = options.signal;
    return new Promise(resolve => { release = resolve; });
  }});
  await client.login('demo_ab', 'fixture-password');
  const pending = client.request('/api/conversations');
  client.dispose();
  assert.equal(signal.aborted, true);
  assert.equal(signedOut, 0);
  release(response(200));
  await assert.rejects(pending, /登录状态已变化/);
});

test('login supplies CSRF for mutations and keeps credentials out of later requests', async () => {
  const { createAuthClient } = await import(moduleUrl);
  const calls = [];
  const client = createAuthClient({ fetchImpl: async (url, options) => {
    calls.push({ url, options });
    return url === '/api/auth/login' ? response(200, { user: { id: 'user-1', username: 'demo_ab' }, csrf_token: 'csrf-1' }) : response(201);
  }});
  await client.login('demo_ab', 'fixture-password');
  await client.request('/api/conversations', { method: 'POST' });
  assert.equal(calls[1].options.headers['X-CSRF-Token'], 'csrf-1');
  assert.ok(!JSON.stringify(calls[1]).includes('fixture-password'));
  assert.equal(client.user.id, 'user-1');
});

test('401 clears current identity and notifies UI to clear private content', async () => {
  const { createAuthClient } = await import(moduleUrl);
  let cleared = 0;
  const client = createAuthClient({ onSignedOut: () => { cleared++; }, fetchImpl: async url => url === '/api/auth/login' ? response(200, { user: { id: 'user-1' }, csrf_token: 'csrf-1' }) : response(401) });
  await client.login('demo_ab', 'fixture-password');
  await client.request('/api/conversations');
  assert.equal(client.user, null);
  assert.equal(cleared, 1);
});

test('logout clears identity only after server accepts revocation', async () => {
  const { createAuthClient } = await import(moduleUrl);
  let logoutStatus = 500;
  const client = createAuthClient({ fetchImpl: async url => url === '/api/auth/login' ? response(200, { user: { id: 'user-1' }, csrf_token: 'csrf-1' }) : response(logoutStatus) });
  await client.login('demo_ab', 'fixture-password');
  await assert.rejects(client.logout());
  assert.equal(client.user.id, 'user-1');
  logoutStatus = 204;
  await client.logout();
  assert.equal(client.user, null);
});
