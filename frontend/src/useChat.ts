import { useCallback, useEffect, useRef, useState } from "react";
import { createAuthClient } from "../../web/static/auth-client.mjs";
import { receiveStream } from "./stream";
import type {
  ChatController,
  Conversation,
  Message,
  Phase,
  User,
} from "./types";

type State = Pick<
  ChatController,
  | "user"
  | "phase"
  | "conversations"
  | "activeId"
  | "messages"
  | "draft"
  | "status"
  | "error"
  | "busy"
  | "runningConversationId"
  | "loginBusy"
  | "memoryWarning"
>;
type Client = {
  user: User | null;
  revision: number;
  check(): Promise<unknown>;
  login(username: string, password: string): Promise<unknown>;
  logout(): Promise<void>;
  request(url: string, options?: RequestInit): Promise<Response>;
  release(response: Response): void;
  dispose(): void;
};
const factory = createAuthClient as unknown as (options: {
  onSignedIn(user: User, changed: boolean): Promise<void>;
  onSignedOut(wasSignedIn: boolean): void;
}) => Client;
const initial: State = {
  user: null,
  phase: "checking",
  conversations: [],
  activeId: null,
  messages: [],
  draft: "",
  status: "",
  error: "",
  busy: false,
  runningConversationId: null,
  loginBusy: false,
  memoryWarning: '',
};
const stageLabels: Record<string, string> = {
  understand: '正在理解问题与记忆意图',
  context: "正在整理对话上下文",
  recovering: "服务已恢复，正在接续原任务",
  answer_restarted: "服务已恢复，正在重新生成完整回答",
  rewrite: "正在理解问题",
  retrieve: "正在查找相关法规",
  rerank: "正在筛选参考资料",
  answer: "正在生成回答",
  memory: "正在读取相关记忆",
  memory_command: "正在整理记忆修改",
};
let messageSequence = 0;
const messageId = () => `message-${++messageSequence}`;
type TurnView = Pick<State, "messages" | "draft" | "status" | "error" | "memoryWarning">;
type Task = { id: string; conversation_id: string; question: string; status: string;
  stage: string; answer: string; error?: string; request_id: string; memory?:{warning?:string;message?:string;status?:string} };

export function useChat(): ChatController {
  const [state, setState] = useState<State>(initial);
  const snapshot = useRef(state);
  const clientRef = useRef<Client | null>(null);
  const mounted = useRef(false);
  const selection = useRef(0);
  const actionLock = useRef(false);
  const initializing = useRef(false);
  const turnViews = useRef(new Map<string, TurnView>());
  const tasksEnabled = useRef(false);
  const observedTask = useRef<Task | null>(null);
  const polling = useRef(false);
  const submission = useRef<{ id: string; question: string; key: string } | null>(null);
  function patch(update: Partial<State>) {
    if (!mounted.current) return;
    snapshot.current = { ...snapshot.current, ...update };
    setState(snapshot.current);
  }
  function valid(client: Client, revision: number) {
    return (
      mounted.current &&
      clientRef.current === client &&
      client.revision === revision &&
      Boolean(client.user)
    );
  }
  async function checked(client: Client, url: string, options?: RequestInit) {
    const response = await client.request(url, options);
    if (!response.ok) {
      let detail: string | undefined;
      try {
        const data = await response.json();
        if (typeof data.detail === "string") detail = data.detail;
      } catch {
        /* Safe fallback. */
      }
      const id = response.headers.get("X-Request-ID");
      throw new Error(
        `${detail || "操作暂时未能完成，请稍后重试。"}${id ? ` 请求编号：${id}` : ""}`,
      );
    }
    return response;
  }
  async function list(
    client: Client,
    revision: number,
  ): Promise<Conversation[] | null> {
    const data = await (await checked(client, "/api/conversations")).json();
    if (!valid(client, revision)) return null;
    tasksEnabled.current = data.background_tasks === true;
    patch({ conversations: data.conversations });
    return data.conversations;
  }
  async function history(client: Client, revision: number, id: string) {
    const epoch = ++selection.current;
    const turn = turnViews.current.get(id);
    if (turn && (snapshot.current.runningConversationId === id || turn.error)) {
      patch({ ...turn, activeId: id, phase: "ready" });
      return;
    }
    turnViews.current.delete(id);
    patch({
      activeId: id,
      messages: [],
      phase: "history-loading",
      error: "",
      status: "",
      memoryWarning: '',
    });
    try {
      const data = await (
        await checked(client, `/api/conversations/${id}/messages`)
      ).json();
      if (!valid(client, revision) || epoch !== selection.current) return;
      const messages = data.messages.map((message: Message) => ({
          ...message,
          id: messageId(),
        }));
      patch({
        messages,
        phase: "ready",
        memoryWarning: data.task?.memory?.message || data.task?.memory?.warning || '',
      });
      if (tasksEnabled.current && data.task && data.task.status !== "completed")
        showTask(client, revision, data.task, messages);
    } catch (error) {
      if (valid(client, revision) && epoch === selection.current)
        patch({ phase: "history-error", error: text(error) });
    }
  }
  async function boot(client: Client) {
    if (initializing.current) return;
    const revision = client.revision;
    initializing.current = true;
    patch({ phase: "loading", error: "" });
    try {
      const conversations = await list(client, revision);
      if (!conversations) return;
      if (tasksEnabled.current) {
        const data = await (await checked(client, "/api/conversations/generation/active")).json();
        if (!valid(client, revision)) return;
        if (data.task) {
          observedTask.current = data.task;
          actionLock.current = true;
          patch({ busy: true, runningConversationId: data.task.conversation_id });
        }
      }
      if (conversations.length)
        await history(client, revision, conversations[0].id);
      else {
        const created = await (
          await checked(client, "/api/conversations", { method: "POST" })
        ).json();
        if (!valid(client, revision)) return;
        const refreshed = await list(client, revision);
        if (refreshed) await history(client, revision, created.id);
      }
    } catch (error) {
      if (valid(client, revision))
        patch({ phase: "load-error", error: text(error) });
    } finally {
      if (clientRef.current === client) initializing.current = false;
    }
  }
  useEffect(() => {
    mounted.current = true;
    const client = factory({
      onSignedOut(wasSignedIn) {
        if (clientRef.current !== client) return;
        selection.current++;
        actionLock.current = false;
        initializing.current = false;
        turnViews.current.clear();
        observedTask.current = null;
        submission.current = null;
        tasksEnabled.current = false;
        patch({
          ...initial,
          phase: "login",
          error: wasSignedIn ? "登录已失效，请重新登录。" : "",
        });
      },
      async onSignedIn(user, changed) {
        if (clientRef.current !== client) return;
        if (changed) {
          selection.current++;
          actionLock.current = false;
          initializing.current = false;
          turnViews.current.clear();
          observedTask.current = null;
          submission.current = null;
          patch({ ...initial, user, phase: "loading" });
        } else patch({ user });
        if (
          changed ||
          ["checking", "login", "load-error"].includes(snapshot.current.phase)
        )
          await boot(client);
      },
    });
    clientRef.current = client;
    const checkIdentity = async () => {
      try {
        await client.check();
      } catch (error) {
        if (clientRef.current === client && mounted.current)
          patch({
            phase: client.user ? snapshot.current.phase : "login",
            error: text(error),
          });
      }
    };
    const visible = () => {
      if (document.visibilityState === "visible") void checkIdentity();
    };
    window.addEventListener("pageshow", checkIdentity);
    document.addEventListener("visibilitychange", visible);
    void checkIdentity();
    const timer = window.setInterval(() => void pollTask(client), 800);
    return () => {
      mounted.current = false;
      window.removeEventListener("pageshow", checkIdentity);
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", visible);
      client.dispose();
      turnViews.current.clear();
      if (clientRef.current === client) clientRef.current = null;
    };
  }, []);

  function showTask(client: Client, revision: number, task: Task, base?: Message[]) {
    if (!valid(client, revision)) return;
    const id = task.conversation_id;
    const messages = (base || turnViews.current.get(id)?.messages ||
      (snapshot.current.activeId === id ? snapshot.current.messages : []))
      .filter(m => !m.id.startsWith("task:"));
    const running = task.status === "running";
    const view: TurnView = {
      messages: [...messages,
        { id: `task:${task.id}:user`, role: "user", content: task.question },
        { id: `task:${task.id}:assistant`, role: "assistant", content: running ? task.answer : "", pending: running }],
      draft: "", status: running ? stageLabels[task.stage] || "正在处理" : "",
      error: running ? "" : `${task.error || "回答未完成，请重新发送。"} 请求编号：${task.request_id}`,
      memoryWarning: task.memory?.message || task.memory?.warning || '',
    };
    turnViews.current.set(id, view);
    if (snapshot.current.activeId === id) patch(view);
    if (running) {
      observedTask.current = task;
      actionLock.current = true;
      patch({ busy: true, runningConversationId: id });
    }
  }

  async function pollTask(client: Client) {
    if (!tasksEnabled.current || polling.current || !client.user || initializing.current) return;
    const revision = client.revision;
    polling.current = true;
    try {
      const previous = observedTask.current;
      let task: Task | null;
      if (previous) task = await (await checked(client, `/api/conversations/${previous.conversation_id}/tasks/${previous.id}`)).json();
      else task = (await (await checked(client, "/api/conversations/generation/active")).json()).task;
      if (!valid(client, revision) || !task) return;
      if (task.status === "running") {
        let base: Message[] | undefined;
        if (!turnViews.current.has(task.conversation_id)) {
          const data = await (await checked(client, `/api/conversations/${task.conversation_id}/messages`)).json();
          if (!valid(client, revision)) return;
          // History and task snapshots may cross completion; don't render both.
          if (data.task?.id === task.id && data.task.status === "completed") task = data.task;
          base = data.messages.map((m: Message) => ({ ...m, id: messageId() }));
        }
        if (task?.status === "running") { showTask(client, revision, task, base); return; }
      }
      if (!task) return;
      const id = task.conversation_id;
      observedTask.current = null;
      actionLock.current = false;
      patch({ busy: false, runningConversationId: null });
      if (task.status === "completed") {
        turnViews.current.delete(id);
        if (snapshot.current.activeId === id) await history(client, revision, id);
        await list(client, revision);
      } else showTask(client, revision, task);
    } catch (error) {
      if (valid(client, revision) && observedTask.current && snapshot.current.activeId === observedTask.current.conversation_id)
        patch({ error: `暂时无法读取进度，任务可能仍在运行。${text(error)}` });
    } finally { polling.current = false; }
  }

  async function sendTask(client: Client, revision: number, id: string, question: string) {
    if (!submission.current || submission.current.id !== id || submission.current.question !== question)
      submission.current = { id, question, key: crypto.randomUUID() };
    actionLock.current = true;
    patch({ busy: true, error: "" });
    try {
      const task: Task = await (await checked(client, `/api/conversations/${id}/tasks`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, submission_key: submission.current.key }),
      })).json();
      if (!valid(client, revision)) return;
      submission.current = null;
      if (task.status === "running") showTask(client, revision, task);
      else {
        actionLock.current = false;
        patch({ busy: false, runningConversationId: null });
        await history(client, revision, id);
      }
    } catch (error) {
      if (valid(client, revision)) {
        actionLock.current = false;
        patch({ busy: false, error: text(error) });
        // A lost acceptance response is resolved by observation, never automatic resubmit.
        await pollTask(client);
      }
    }
  }

  async function check() {
    try {
      await clientRef.current?.check();
    } catch (error) {
      patch({ error: text(error) });
    }
  }
  async function login(username: string, password: string) {
    const client = clientRef.current;
    if (!client || snapshot.current.loginBusy) return;
    patch({ loginBusy: true, error: "" });
    try {
      await client.login(username.trim(), password);
    } catch (error) {
      if (clientRef.current === client) patch({ error: text(error) });
    } finally {
      if (clientRef.current === client) patch({ loginBusy: false });
    }
  }
  async function logout() {
    const client = clientRef.current;
    if (!client || actionLock.current || snapshot.current.busy) return;
    const revision = client.revision;
    actionLock.current = true;
    patch({ busy: true, error: "" });
    try {
      await client.logout();
    } catch (error) {
      if (valid(client, revision)) patch({ error: text(error) });
    } finally {
      if (valid(client, revision)) {
        actionLock.current = false;
        patch({ busy: false });
      }
    }
  }
  async function retry() {
    const client = clientRef.current;
    if (
      !client?.user ||
      (actionLock.current && !snapshot.current.runningConversationId)
    )
      return;
    if (snapshot.current.phase === "history-error" && snapshot.current.activeId)
      await history(client, client.revision, snapshot.current.activeId);
    else await boot(client);
  }
  async function select(id: string) {
    const client = clientRef.current;
    if (
      !client?.user ||
      (actionLock.current && !snapshot.current.runningConversationId) ||
      snapshot.current.activeId === id ||
      !snapshot.current.conversations.some((item) => item.id === id)
    )
      return;
    const previous = snapshot.current.activeId;
    const previousTurn = previous ? turnViews.current.get(previous) : undefined;
    if (previous && previousTurn?.error)
      turnViews.current.set(previous, {
        ...previousTurn,
        draft: snapshot.current.draft,
      });
    patch({ draft: "" });
    await history(client, client.revision, id);
  }
  async function mutate(
    operation: (client: Client, revision: number) => Promise<void>,
  ): Promise<boolean> {
    const client = clientRef.current;
    if (!client?.user || actionLock.current || initializing.current)
      return false;
    const revision = client.revision;
    actionLock.current = true;
    patch({ busy: true, error: "" });
    try {
      await operation(client, revision);
      return valid(client, revision);
    } catch (error) {
      if (valid(client, revision)) patch({ error: text(error) });
      return false;
    } finally {
      if (valid(client, revision)) {
        actionLock.current = false;
        patch({ busy: false });
      }
    }
  }
  async function create() {
    await mutate(async (client, revision) => {
      const created = await (
        await checked(client, "/api/conversations", { method: "POST" })
      ).json();
      if (!valid(client, revision)) return;
      if (await list(client, revision)) {
        patch({ draft: "" });
        await history(client, revision, created.id);
      }
    });
  }
  async function rename(id: string, title: string) {
    const trimmed = title.trim();
    if (!trimmed || [...trimmed].length > 80) {
      patch({ error: "聊天名称需要 1 到 80 个字符。" });
      return false;
    }
    return mutate(async (client, revision) => {
      await checked(client, `/api/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: trimmed }),
      });
      await list(client, revision);
    });
  }
  async function remove(id: string) {
    await mutate(async (client, revision) => {
      await checked(client, `/api/conversations/${id}`, { method: "DELETE" });
      if (!valid(client, revision)) return;
      const deletedActive = snapshot.current.activeId === id;
      turnViews.current.delete(id);
      patch({
        conversations: snapshot.current.conversations.filter(
          (item) => item.id !== id,
        ),
      });
      if (deletedActive) {
        selection.current++;
        patch({ activeId: null, messages: [], draft: "", phase: "load-error" });
      }
      const remaining = await list(client, revision);
      if (!remaining) return;
      if (deletedActive) {
        patch({ draft: "" });
        if (remaining.length) await history(client, revision, remaining[0].id);
        else {
          const created = await (
            await checked(client, "/api/conversations", { method: "POST" })
          ).json();
          if (await list(client, revision))
            await history(client, revision, created.id);
        }
      }
    });
  }
  async function send() {
    const client = clientRef.current;
    const current = snapshot.current;
    const question = current.draft.trim();
    if (
      !client?.user ||
      actionLock.current ||
      current.phase !== "ready" ||
      !current.activeId ||
      !question
    )
      return;
    if (question.length > 2000) {
      patch({ error: "问题请控制在 2000 个字符以内。" });
      return;
    }
    const revision = client.revision;
    const id = current.activeId;
    if (tasksEnabled.current) {
      await sendTask(client, revision, id, question);
      return;
    }
    const original = current.messages;
    const assistantId = messageId();
    const pending: Message[] = [
      ...original,
      { id: messageId(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", pending: true },
    ];
    const own = () => valid(client, revision);
    function updateTurn(update: Partial<TurnView>) {
      if (!own()) return;
      const turn = { ...turnViews.current.get(id)!, ...update };
      turnViews.current.set(id, turn);
      if (snapshot.current.activeId === id) patch(turn);
    }
    actionLock.current = true;
    turnViews.current.set(id, {
      messages: pending,
      draft: "",
      error: "",
      status: stageLabels.rewrite,
    });
    patch({
      busy: true,
      runningConversationId: id,
      draft: "",
      error: "",
      messages: pending,
      status: stageLabels.rewrite,
    });
    let response: Response | undefined;
    let requestId: string | null = null;
    let answer = "";
    let completed = false;
    try {
      response = await client.request(`/api/conversations/${id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      requestId = response.headers.get("X-Request-ID");
      if (!response.ok) {
        let detail = "暂时无法完成回答，请稍后重试。";
        try {
          const data = await response.json();
          if (typeof data.detail === "string") detail = data.detail;
        } catch {
          /* Safe fallback. */
        }
        throw new Error(detail);
      }
      await receiveStream(response, ({ event, data }) => {
        if (!own()) throw new Error("登录状态已变化。");
        if (event === "status")
          updateTurn({ status: stageLabels[data.stage] || "正在处理" });
        else if (event === "delta") {
          answer += data.text;
          updateTurn({
            messages: turnViews.current
              .get(id)!
              .messages.map((message) =>
                message.id === assistantId
                  ? { ...message, content: answer }
                  : message,
              ),
          });
        } else if (event === "done") {
          completed = true;
          updateTurn({
            status: "",
            memoryWarning: data.memory?.message || data.memory?.warning || '',
            messages: turnViews.current.get(id)!.messages.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: data.answer,
                    sources: data.sources || [],
                    pending: false,
                  }
                : message,
            ),
          });
        } else if (event === "error") {
          requestId = data.request_id || requestId;
          throw new Error(data.message || "回答未完成，请重试。");
        }
      });
      if (!completed) throw new Error("回答连接已中断，请重试。");
      try {
        await list(client, revision);
      } catch {
        if (own())
          updateTurn({ error: "回答已保存，但聊天列表暂时无法刷新。" });
      }
    } catch (error) {
      if (own()) {
        const detail = completed ? "回答已保存，连接随后中断。" : text(error);
        updateTurn({
          ...(completed ? {} : { messages: original, draft: question }),
          status: "",
          error: `${detail}${requestId ? ` 请求编号：${requestId}` : ""}`,
        });
      }
    } finally {
      if (response) client.release(response);
      if (valid(client, revision)) {
        actionLock.current = false;
        updateTurn({ status: "" });
        patch({ busy: false, runningConversationId: null });
        if (!turnViews.current.get(id)?.error) turnViews.current.delete(id);
      }
    }
  }
  const memoryRequest=useCallback(async(path:string,options?:RequestInit)=>{
    const client=clientRef.current;
    if(!client?.user)throw Error('请先登录');
    const revision=client.revision;
    const response=await checked(client,path,options);
    try {
      const data=await response.json();
      if(!valid(client,revision))throw Error('登录账号已变化，请重新打开设置');
      return data;
    } finally {client.release(response);}
  },[]);
  return {
    ...state,
    title:
      state.conversations.find((item) => item.id === state.activeId)?.title ||
      "新对话",
    setDraft: (value) => patch({ draft: value }),
    login,
    logout,
    check,
    retry,
    select,
    create,
    rename,
    remove,
    send,
    memoryRequest,
  };
}

function text(error: unknown) {
  return error instanceof Error ? error.message : "操作未能完成，请稍后重试。";
}
