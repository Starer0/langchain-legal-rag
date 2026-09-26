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

function setStatus(message, isError = false) {
  statusArea.textContent = message;
  statusArea.classList.toggle("error", isError);
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
  body.textContent = content;
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
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error("请求未成功");
    await receiveStream(response, ({ event, data }) => {
      if (event === "status") setStatus(stageLabels[data.stage] || "正在处理…");
      if (event === "delta") {
        assistantMessage ||= appendMessage("assistant");
        assistantMessage.querySelector(".message-content").textContent += data.text;
      }
      if (event === "done") {
        if (!assistantMessage) assistantMessage = appendMessage("assistant", data.answer);
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
    const response = await fetch("/api/history");
    if (!response.ok) throw new Error("无法读取历史对话");
    const { messages } = await response.json();
    messages.forEach(({ role, content }) => appendMessage(role, content));
  } catch {
    setStatus("历史对话暂时无法加载，仍可开始新对话。", true);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  sendButton.disabled = true;
  await sendQuestion(question);
  sendButton.disabled = false;
  questionInput.focus();
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

loadHistory();
