import { renderMarkdown } from "/static/markdown.mjs";
import { createAuthClient } from "/static/auth-client.mjs";

const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messageLog = document.querySelector("#message-log");
const statusArea = document.querySelector("#status");
const sourceTemplate = document.querySelector("#source-template");
const stageLabels = {
  rewrite: "正在理解当前问题…",
  retrieve: "正在检索法规资料…",
  rerank: "正在筛选相关法条…",
  answer: "正在生成回答…",
};
let conversations = [];
let currentConversationId = null;
let isGenerating = false;
let workspaceReady = false;
let initializingRevision = null;
const conversationList = document.querySelector("#conversation-list");
const conversationTitle = document.querySelector("#current-conversation-title");
const newConversationButton = document.querySelector("#new-conversation");
const renameDialog = document.querySelector("#rename-dialog");
const renameForm = document.querySelector("#rename-form");
const renameInput = document.querySelector("#rename-input");
const renameError = document.querySelector("#rename-error");
const renameCancelButton = document.querySelector("#rename-cancel");
let renamingConversation = null;
const workspace = document.querySelector('#workspace');
const loginPanel = document.querySelector('#login-panel');
const loginForm = document.querySelector('#login-form');
const loginUsername = document.querySelector('#login-username');
const loginPassword = document.querySelector('#login-password');
const loginError = document.querySelector('#login-error');
const loginButton = document.querySelector('#login-button');
const logoutButton = document.querySelector('#logout-button');
const authLoading = document.querySelector('#auth-loading');
const currentUserLabel = document.querySelector('#current-user');
const authClient = createAuthClient({
  onSignedOut: wasSignedIn => showLogin(wasSignedIn ? '登录已失效，请重新登录。' : ''),
  onSignedIn: async (user, changed) => {
    authLoading.hidden = true;
    loginPanel.hidden = true;
    workspace.hidden = false;
    currentUserLabel.textContent = user.username;
    if (changed) clearPrivateContent();
    if (workspaceReady || initializingRevision === authClient.revision) return;
    const revision = authClient.revision;
    initializingRevision = revision;
    sendButton.disabled = true;
    newConversationButton.disabled = true;
    setStatus('正在加载你的对话…');
    try {
      if (!await loadConversations()) return;
      setStatus('');
      if (!conversations.length) await createConversation();
      else await selectConversation(conversations[0].id);
      if (revision !== authClient.revision || !currentConversationId) return;
      workspaceReady = true;
      sendButton.disabled = false;
      newConversationButton.disabled = false;
    } catch {
      if (revision === authClient.revision) {
        setStatus('对话暂时无法加载，请刷新页面重试。', true);
        newConversationButton.disabled = false;
      }
    } finally {
      if (initializingRevision === revision) initializingRevision = null;
    }
  },
});
const apiFetch = (url, options) => authClient.request(url, options);

function clearPrivateContent() {
  workspaceReady = false;
  isGenerating = false;
  sendButton.disabled = true;
  newConversationButton.disabled = true;
  logoutButton.disabled = false;
  conversations = [];
  currentConversationId = null;
  conversationList.replaceChildren();
  messageLog.replaceChildren();
  questionInput.value = '';
  conversationTitle.textContent = '新对话';
  setStatus('');
  renamingConversation = null;
  if (renameDialog.open) renameDialog.close();
}

function showLogin(message = '') {
  clearPrivateContent();
  authLoading.hidden = true;
  workspace.hidden = true;
  loginPanel.hidden = false;
  currentUserLabel.textContent = '';
  loginPassword.value = '';
  loginError.textContent = message;
  loginUsername.focus();
}

function setStatus(message, isError = false) {
  statusArea.textContent = message;
  statusArea.classList.toggle("error", isError);
}

function showWelcome() {
  const welcome = document.createElement("section");
  welcome.className = "welcome-card";
  welcome.id = "welcome-card";
  welcome.innerHTML = `
    <p class="eyebrow">开始咨询</p>
    <h2>请输入一条劳动法律法规问题</h2>
    <p>系统会结合当前对话，将追问改写为适合检索的问题，再提供相关法条来源。</p>`;
  messageLog.append(welcome);
}

function appendMessage(role, content = "") {
  document.querySelector("#welcome-card")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "法律法规助手";
  const body = document.createElement("div");
  body.className = "message-content";
  renderMarkdown(body, content);
  article.append(label, body);
  messageLog.append(article);
  article.scrollIntoView({ block: "end", behavior: "smooth" });
  return article;
}

function addSources(message, sources) {
  if (!sources?.length) return;
  const fragment = sourceTemplate.content.cloneNode(true);
  const list = fragment.querySelector("ul");
  for (const source of sources) {
    const item = document.createElement("li");
    const title = source.law_name || source.title || source.source || "法律资料";
    const locator = source.article || source.section;
    const article = locator ? `《${title}》${locator}` : title;
    const pages = source.pages?.length ? `，第 ${source.pages.join("、")} 页` : "";
    item.textContent = `${article}${pages}`;
    list.append(item);
  }
  message.append(fragment);
}

function parseFrame(frame) {
  const lines = frame.split("\n");
  const event = lines.find((line) => line.startsWith("event: "))?.slice(7);
  const serialized = lines.find((line) => line.startsWith("data: "))?.slice(6);
  return event && serialized ? { event, data: JSON.parse(serialized) } : null;
}

async function receiveStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (parsed) onEvent(parsed);
    }
    if (done) break;
  }
}

async function sendQuestion(question) {
  const owner = authClient.user?.id;
  const revision = authClient.revision;
  const pendingUser = appendMessage("user", question);
  let assistantMessage = null;
  let streamedAnswer = "";
  let requestId = null;
  let completed = false;
  let response = null;
  try {
    response = await apiFetch(`/api/conversations/${currentConversationId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    requestId = response.headers.get("X-Request-ID");
    if (!response.ok) throw new Error("请求未成功，请稍后重试。");
    await receiveStream(response, ({ event, data }) => {
      if (authClient.user?.id !== owner || authClient.revision !== revision) throw new Error('登录状态已变化。');
      if (event === "status") setStatus(stageLabels[data.stage] || "正在处理…");
      if (event === "delta") {
        assistantMessage ||= appendMessage("assistant");
        streamedAnswer += data.text;
        renderMarkdown(assistantMessage.querySelector(".message-content"), streamedAnswer);
      }
      if (event === "done") {
        completed = true;
        if (!assistantMessage) assistantMessage = appendMessage("assistant", data.answer);
        else renderMarkdown(assistantMessage.querySelector(".message-content"), data.answer);
        addSources(assistantMessage, data.sources);
        setStatus("");
      }
      if (event === "error") {
        requestId = data.request_id || requestId;
        throw new Error(data.message || "暂时无法完成回答，请稍后重试。");
      }
    });
    if (!completed) throw new Error("回答连接已中断，请重试。");
  } catch (error) {
    pendingUser.remove();
    assistantMessage?.remove();
    if (authClient.user?.id !== owner || authClient.revision !== revision) return;
    questionInput.value = question;
    const message = error.message || "网络连接中断，请重试。";
    setStatus(requestId ? `${message} 请求编号：${requestId}` : message, true);
  } finally {
    if (response) authClient.release(response);
  }
}

async function loadHistory() {
  const owner = authClient.user?.id;
  const revision = authClient.revision;
  const conversationId = currentConversationId;
  try {
    const response = await apiFetch(`/api/conversations/${currentConversationId}/messages`);
    if (!response.ok) throw new Error("无法读取历史对话");
    const { messages } = await response.json();
    if (authClient.user?.id !== owner || authClient.revision !== revision || currentConversationId !== conversationId) return;
    messageLog.replaceChildren();
    if (!messages.length) showWelcome();
    messages.forEach(({ role, content }) => appendMessage(role, content));
  } catch {
    if (authClient.user?.id !== owner || authClient.revision !== revision || currentConversationId !== conversationId) return;
    messageLog.replaceChildren();
    showWelcome();
    setStatus("历史对话暂时无法加载，仍可开始新对话。", true);
  }
}

function renderConversations() {
  conversationList.replaceChildren();
  for (const conversation of conversations) {
    const item = document.createElement("div"); item.className = `conversation-item${conversation.id === currentConversationId ? " active" : ""}`;
    const title = document.createElement("button"); title.type = "button"; title.className = "conversation-title"; title.textContent = conversation.title; title.title = "双击重命名"; title.disabled = isGenerating; title.onclick = () => selectConversation(conversation.id); title.ondblclick = (event) => { event.preventDefault(); openRenameDialog(conversation); };
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "conversation-action delete-conversation"; remove.setAttribute("aria-label", `删除聊天：${conversation.title}`); remove.title = "删除聊天"; remove.disabled = isGenerating; remove.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6m4-6v6M9 7l1-2h4l1 2m-9 0 1 13h10l1-13" /></svg>'; remove.onclick = () => deleteConversation(conversation.id);
    item.append(title, remove); conversationList.append(item);
  }
}
async function loadConversations() {
  const revision = authClient.revision;
  const response = await apiFetch("/api/conversations");
  if (!response.ok) throw new Error("无法读取对话列表");
  const data = await response.json();
  if (authClient.revision !== revision) return false;
  conversations = data.conversations;
  return true;
}
async function selectConversation(id) {
  if (isGenerating || id === currentConversationId) return;
  const current = conversations.find(item => item.id === id);
  if (!current) return;
  const revision = authClient.revision;
  currentConversationId = id;
  conversationTitle.textContent = current.title;
  messageLog.replaceChildren();
  await loadHistory();
  if (authClient.revision !== revision) return;
  renderConversations();
}
async function createConversation() {
  if (isGenerating || !authClient.user) return;
  const revision = authClient.revision;
  const response = await apiFetch("/api/conversations", {method: "POST"});
  if (authClient.revision !== revision) return;
  if (!response.ok) throw new Error("无法新建对话，请稍后重试。");
  const created = await response.json();
  if (authClient.revision !== revision || !await loadConversations()) return;
  await selectConversation(created.id);
  if (authClient.revision !== revision) return;
  workspaceReady = Boolean(currentConversationId);
  sendButton.disabled = !workspaceReady;
  newConversationButton.disabled = false;
  questionInput.focus();
}
function openRenameDialog(conversation) { if (isGenerating) return; renamingConversation = conversation; renameInput.value = conversation.title; renameError.textContent = ""; renameDialog.showModal(); renameInput.focus(); renameInput.select(); }
async function renameConversation() {
  const title = renameInput.value.trim();
  if (!renamingConversation || !title) { renameError.textContent = "请输入 1 到 80 个字符。"; return; }
  const revision = authClient.revision;
  const response = await apiFetch(`/api/conversations/${renamingConversation.id}`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify({title})});
  if (authClient.revision !== revision) return;
  if (!response.ok) { renameError.textContent = "标题需要 1 到 80 个字符。"; return; }
  renameDialog.close();
  renamingConversation = null;
  if (!await loadConversations()) return;
  const current = conversations.find(item => item.id === currentConversationId);
  conversationTitle.textContent = current?.title || "新对话";
  renderConversations();
}
async function deleteConversation(id) {
  if (isGenerating || !window.confirm("删除后无法恢复这段对话，确定删除吗？")) return;
  const revision = authClient.revision;
  const response = await apiFetch(`/api/conversations/${id}`, {method:"DELETE"});
  if (authClient.revision !== revision) return;
  if (!response.ok) return setStatus("无法删除对话，请稍后重试。", true);
  if (!await loadConversations()) return;
  currentConversationId = null;
  if (conversations.length) await selectConversation(conversations[0].id);
  else await createConversation();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !workspaceReady || isGenerating || !currentConversationId) return;
  const revision = authClient.revision;
  questionInput.value = "";
  sendButton.disabled = true;
  logoutButton.disabled = true;
  isGenerating = true;
  newConversationButton.disabled = true;
  renderConversations();
  await sendQuestion(question);
  if (authClient.revision !== revision) return;
  sendButton.disabled = false;
  logoutButton.disabled = false;
  isGenerating = false;
  newConversationButton.disabled = false;
  if (!authClient.user) return;
  if (!await loadConversations()) return;
  const current = conversations.find((item) => item.id === currentConversationId);
  conversationTitle.textContent = current?.title || "新对话";
  renderConversations();
  questionInput.focus();
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

newConversationButton.addEventListener("click", async () => {
  try { await createConversation(); }
  catch { setStatus("无法新建对话，请稍后重试。", true); }
});
renameForm.addEventListener("submit", async (event) => { event.preventDefault(); await renameConversation(); });
renameCancelButton.addEventListener("click", () => { renameDialog.close(); renamingConversation = null; });
loginForm.addEventListener('submit', async event => {
  event.preventDefault();
  loginButton.disabled = true;
  loginButton.textContent = '正在登录…';
  loginError.textContent = '';
  try { await authClient.login(loginUsername.value.trim(), loginPassword.value); }
  catch (error) { loginError.textContent = error.message || '暂时无法登录，请稍后重试。'; }
  finally { loginPassword.value = ''; loginButton.disabled = false; loginButton.textContent = '登录'; }
});
logoutButton.addEventListener('click', async () => {
  if (isGenerating) return;
  logoutButton.disabled = true;
  try { await authClient.logout(); showLogin('你已退出登录。'); }
  catch (error) { setStatus(error.message || '退出未完成，请重试。', true); }
  finally { logoutButton.disabled = false; }
});
async function checkIdentity() {
  try { await authClient.check(); }
  catch (error) {
    if (!authClient.user) showLogin(error.message || '无法检查登录状态，请刷新重试。');
    else setStatus(error.message, true);
  }
}
window.addEventListener('pageshow', checkIdentity);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') checkIdentity(); });
checkIdentity();
