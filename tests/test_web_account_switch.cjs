const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(fetchImpl) {
  const elements = new Map();
  const element = () => ({ hidden: false, disabled: false, value: '', textContent: '',
    classList: { toggle() {} }, replaceChildren() {}, append() {}, focus() {},
    addEventListener() {}, close() {}, select() {}, setAttribute() {} });
  let callback;
  const client = { user: { id: 'A', username: 'A' }, revision: 1, request: fetchImpl };
  const context = vm.createContext({ document: {
    querySelector(key) { if (!elements.has(key)) elements.set(key, element()); return elements.get(key); },
    createElement: element, addEventListener() {},
  }, window: { addEventListener() {}, confirm: () => true },
  createAuthClient(options) { callback = options.onSignedIn; return client; }, renderMarkdown() {} });
  const source = fs.readFileSync(path.join(__dirname, '../web/static/app.js'), 'utf8')
    .replace(/^import .*;\r?\n/gm, '').replace(/checkIdentity\(\);\s*$/, '');
  vm.runInContext(source, context);
  return { client, elements, signIn: (...args) => callback(...args),
    run: code => vm.runInContext(code, context) };
}

test('late create response from A cannot select an A conversation after switching to B', async () => {
  let release;
  const delayedBody = new Promise(resolve => { release = resolve; });
  const ui = page(async (url, options) => options?.method === 'POST'
    ? { ok: true, json: () => delayedBody }
    : { ok: true, json: async () => ({ conversations: [{ id: 'B-conv', title: 'B' }] }) });
  const creating = ui.run('createConversation()');
  await Promise.resolve();
  ui.client.user = { id: 'B' }; ui.client.revision++;
  ui.run("conversations = [{id:'B-conv',title:'B'}]; currentConversationId = 'B-conv'");
  release({ id: 'A-conv' });
  await creating;
  assert.equal(ui.run('currentConversationId'), 'B-conv');
});

test('account initialization disables sending and retries after a visible list failure', async () => {
  let attempts = 0;
  const ui = page(async url => {
    if (url.endsWith('/messages')) return { ok: true, json: async () => ({ messages: [] }) };
    attempts++;
    assert.equal(ui.elements.get('#send-button').disabled, true);
    return { ok: attempts > 1, json: async () => ({ conversations: [{ id: 'A-conv', title: 'A' }] }) };
  });
  await ui.signIn(ui.client.user, true);
  assert.equal(ui.elements.get('#workspace').hidden, false);
  assert.match(ui.elements.get('#status').textContent, /加载/);
  assert.equal(ui.elements.get('#send-button').disabled, true);
  await ui.signIn(ui.client.user, false);
  assert.equal(ui.run('currentConversationId'), 'A-conv');
  assert.equal(ui.elements.get('#send-button').disabled, false);
});
