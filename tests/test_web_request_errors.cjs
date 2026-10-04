const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Run the real sendQuestion function with controlled network and DOM boundaries.
const app = fs.readFileSync(path.join(__dirname, '../web/static/app.js'), 'utf8');
const source = app.slice(app.indexOf('async function sendQuestion('), app.indexOf('async function loadHistory('));

async function send({ ok = true, id = 'req-123', events = [], networkError = null }) {
  const statuses = [];
  const messages = [];
  const context = {
    currentConversationId: 'conversation',
    questionInput: { value: '' },
    stageLabels: {},
    appendMessage: () => {
      const message = { removed: false, remove() { this.removed = true; }, querySelector() { return {}; } };
      messages.push(message);
      return message;
    },
    renderMarkdown() {}, addSources() {},
    setStatus: (text, error) => statuses.push({ text, error }),
    authClient: { user: { id: 'test-user' }, release() {} },
    apiFetch: async () => {
      if (networkError) throw new Error(networkError);
      return { ok, headers: { get: () => id } };
    },
    receiveStream: async (_, onEvent) => { for (const event of events) onEvent(event); },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  await context.sendQuestion('问题');
  return { statuses, messages, input: context.questionInput.value };
}

test('HTTP rejection displays the response request ID and restores the question', async () => {
  const result = await send({ ok: false });
  assert.match(result.statuses.at(-1).text, /请求编号：req-123/);
  assert.equal(result.input, '问题');
  assert.equal(result.messages[0].removed, true);
});

test('SSE failure displays the event request ID and removes partial output', async () => {
  const result = await send({ events: [
    { event: 'delta', data: { text: '部分回答' } },
    { event: 'error', data: { message: '暂时无法完成回答', request_id: 'event-456' } },
  ] });
  assert.match(result.statuses.at(-1).text, /请求编号：event-456/);
  assert.ok(result.messages.every(message => message.removed));
});

test('stream EOF without done is an error, rather than a successful partial answer', async () => {
  const result = await send({ events: [{ event: 'delta', data: { text: '部分回答' } }] });
  assert.equal(result.statuses.at(-1).error, true);
  assert.match(result.statuses.at(-1).text, /请求编号：req-123/);
  assert.ok(result.messages.every(message => message.removed));
});

test('completed answer remains visible without an error', async () => {
  const result = await send({ events: [
    { event: 'delta', data: { text: '回答' } },
    { event: 'done', data: { answer: '回答', sources: [] } },
  ] });
  assert.equal(result.statuses.at(-1).text, '');
  assert.ok(result.messages.every(message => !message.removed));
  assert.equal(result.input, '');
});

test('network failure before response has no invented request ID', async () => {
  const result = await send({ networkError: '网络连接中断' });
  assert.equal(result.statuses.at(-1).text, '网络连接中断');
  assert.equal(result.input, '问题');
});
