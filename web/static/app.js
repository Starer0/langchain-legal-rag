import { renderMarkdown } from "/static/markdown.mjs";

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
const conversationList = document.querySelector("#conversation-list");
const conversationTitle = document.querySelector("#current-conversation-title");
const newConversationButton = document.querySelector("#new-conversation");

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
    const law = source.law_name || source.source || "法律资料";
    const article = source.article ? `《${law}》${source.article}` : law;
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
  const pendingUser = appendMessage("user", question);
  let assistantMessage = null;
  let streamedAnswer = "";
  try {
    const response = await fetch(`/api/conversations/${currentConversationId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error("请求未成功");
    await receiveStream(response, ({ event, data }) => {
      if (event === "status") setStatus(stageLabels[data.stage] || "正在处理…");
      if (event === "delta") {
        assistantMessage ||= appendMessage("assistant");
        streamedAnswer += data.text;
        renderMarkdown(assistantMessage.querySelector(".message-content"), streamedAnswer);
      }
      if (event === "done") {
        if (!assistantMessage) assistantMessage = appendMessage("assistant", data.answer);
        else renderMarkdown(assistantMessage.querySelector(".message-content"), data.answer);
        addSources(assistantMessage, data.sources);
        setStatus("");
      }
      if (event === "error") throw new Error(data.message || "暂时无法完成回答，请稍后重试。");
    });
  } catch (error) {
    pendingUser.remove();
    assistantMessage?.remove();
    questionInput.value = question;
    setStatus(error.message || "网络连接中断，请重试。", true);
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`/api/conversations/${currentConversationId}/messages`);
    if (!response.ok) throw new Error("无法读取历史对话");
    const { messages } = await response.json();
    messageLog.replaceChildren();
    if (!messages.length) showWelcome();
    messages.forEach(({ role, content }) => appendMessage(role, content));
  } catch {
    messageLog.replaceChildren();
    showWelcome();
    setStatus("历史对话暂时无法加载，仍可开始新对话。", true);
  }
}

function renderConversations() {
  conversationList.replaceChildren();
  for (const conversation of conversations) {
    const item = document.createElement("div"); item.className = `conversation-item${conversation.id === currentConversationId ? " active" : ""}`;
    const title = document.createElement("button"); title.type = "button"; title.className = "conversation-title"; title.textContent = conversation.title; title.title = conversation.title; title.disabled = isGenerating; title.onclick = () => selectConversation(conversation.id);
    const rename = document.createElement("button"); rename.type = "button"; rename.className = "conversation-action"; rename.textContent = "改名"; rename.disabled = isGenerating; rename.onclick = () => renameConversation(conversation);
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "conversation-action"; remove.textContent = "删除"; remove.disabled = isGenerating; remove.onclick = () => deleteConversation(conversation.id);
    item.append(title, rename, remove); conversationList.append(item);
  }
}
async function loadConversations() { const response = await fetch("/api/conversations"); if (!response.ok) throw new Error("无法读取对话列表"); conversations = (await response.json()).conversations; }
async function selectConversation(id) { if (isGenerating || id === currentConversationId) return; currentConversationId = id; const current = conversations.find((item) => item.id === id); conversationTitle.textContent = current.title; messageLog.replaceChildren(); await loadHistory(); renderConversations(); }
async function createConversation() { if (isGenerating) return; const response = await fetch("/api/conversations", {method:"POST"}); if (!response.ok) return setStatus("无法新建对话，请稍后重试。", true); const created = await response.json(); await loadConversations(); await selectConversation(created.id); questionInput.focus(); }
async function renameConversation(conversation) { if (isGenerating) return; const title = window.prompt("对话标题", conversation.title); if (title === null) return; const response = await fetch(`/api/conversations/${conversation.id}`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify({title})}); if (!response.ok) return setStatus("标题需要 1 到 80 个字符。", true); await loadConversations(); const current = conversations.find((item) => item.id === currentConversationId); conversationTitle.textContent = current?.title || "新对话"; renderConversations(); }
async function deleteConversation(id) { if (isGenerating || !window.confirm("删除后无法恢复这段对话，确定删除吗？")) return; const response = await fetch(`/api/conversations/${id}`, {method:"DELETE"}); if (!response.ok) return setStatus("无法删除对话，请稍后重试。", true); await loadConversations(); currentConversationId = null; if (conversations.length) await selectConversation(conversations[0].id); else await createConversation(); }

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  sendButton.disabled = true;
  isGenerating = true;
  newConversationButton.disabled = true;
  renderConversations();
  await sendQuestion(question);
  sendButton.disabled = false;
  isGenerating = false;
  newConversationButton.disabled = false;
  await loadConversations();
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

newConversationButton.addEventListener("click", createConversation);
(async () => { await loadConversations(); if (!conversations.length) await createConversation(); else await selectConversation(conversations[0].id); })();
