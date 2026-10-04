import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  FileText,
  LoaderCircle,
  LogOut,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  Scale,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { renderMarkdown } from "../../web/static/markdown.mjs";
import type { ChatController, Conversation, Message } from "./types";

const suggestions = [
  "试用期最长可以约定多久？",
  "劳动合同解除需要满足哪些条件？",
  "公司应如何保护员工个人信息？",
];
function Brand() {
  return (
    <div className="brand">
      <span className="brand-mark">
        <Scale size={22} />
      </span>
      <div>
        <strong>法律法规问答</strong>
        <span>LEGAL KNOWLEDGE ASSISTANT</span>
      </div>
    </div>
  );
}
function Spinner() {
  return <LoaderCircle className="spinner" size={18} aria-hidden="true" />;
}
function ErrorNotice({ chat }: { chat: ChatController }) {
  return chat.error ? (
    <div className="error-notice" role="alert">
      <span>{chat.error}</span>
      {["history-error", "load-error"].includes(chat.phase) && (
        <button onClick={() => void chat.retry()} disabled={chat.busy}>
          重新加载
        </button>
      )}
    </div>
  ) : null;
}
function Modal({
  title,
  children,
  onClose,
  drawer = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  drawer?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const backdrop = ref.current?.parentElement;
    const siblings = Array.from(backdrop?.parentElement?.children ?? []).filter(
      (element) => element !== backdrop,
    ) as HTMLElement[];
    const previousInert = siblings.map((element) => element.inert);
    siblings.forEach((element) => {
      element.inert = true;
    });
    ref.current
      ?.querySelector<HTMLElement>('input,button,[tabindex="0"]')
      ?.focus();
    return () => {
      siblings.forEach((element, index) => {
        element.inert = previousInert[index];
      });
      previous?.focus();
    };
  }, []);
  return (
    <div
      className={`modal-backdrop ${drawer ? "drawer-backdrop" : ""}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        className={drawer ? "drawer" : "modal"}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            onClose();
          }
          if (event.key === "Tab") {
            const items = Array.from(
              ref.current?.querySelectorAll<HTMLElement>(
                'button:not(:disabled),input:not(:disabled),textarea:not(:disabled),[tabindex="0"]',
              ) ?? [],
            );
            const first = items[0],
              last = items.at(-1);
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first?.focus();
            }
          }
        }}
      >
        {children}
      </div>
    </div>
  );
}
function Login({ chat }: { chat: ChatController }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  return (
    <main className="login-page">
      <div className="login-brand">
        <Brand />
      </div>
      <section className="login-intro">
        <span className="eyebrow">以法规为依据 · 让决策更清晰</span>
        <h1>
          法律知识，
          <br />
          触手可及。
        </h1>
        <p>
          围绕企业日常法律问题，查阅法规依据，
          <br className="desktop-break" />
          获取清晰、有来源的解答。
        </p>
        <div className="intro-proof">
          <ShieldCheck size={19} />
          <span>企业资料权限保护</span>
          <BookOpen size={19} />
          <span>法规来源可查阅</span>
        </div>
      </section>
      <section className="login-card">
        <span className="eyebrow">欢迎回来</span>
        <h2>登录工作空间</h2>
        <p>使用您的账号继续法律法规问答</p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void chat.login(username, password);
          }}
        >
          <label htmlFor="username">用户名</label>
          <input
            id="username"
            name="username"
            maxLength={32}
            autoComplete="username"
            required
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={chat.loginBusy}
          />
          <label htmlFor="password">密码</label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={chat.loginBusy}
          />
          <ErrorNotice chat={chat} />
          <button className="primary login-submit" disabled={chat.loginBusy}>
            {chat.loginBusy ? (
              <>
                <Spinner />
                正在登录
              </>
            ) : (
              "登录"
            )}
            <ChevronRight size={18} />
          </button>
        </form>
        <div className="login-footnote">
          <ShieldCheck size={14} /> 登录后仅可访问账号授权的资料
        </div>
      </section>
      <footer className="login-footer">法律法规问答 · 企业知识工作空间</footer>
    </main>
  );
}
function Sidebar({
  chat,
  onSelect,
  onEdit,
}: {
  chat: ChatController;
  onSelect: () => void;
  onEdit: (mode: "rename" | "delete", conversation: Conversation) => void;
}) {
  return (
    <>
      <Brand />
      <button
        className="new-chat"
        disabled={chat.busy || chat.phase !== "ready"}
        onClick={() => {
          void chat.create();
          onSelect();
        }}
      >
        <Plus size={18} />
        新建对话<span>+</span>
      </button>
      <div className="sidebar-section-label">
        我的对话 <span>{chat.conversations.length}</span>
      </div>
      <nav className="conversation-list" aria-label="历史对话">
        {chat.conversations.map((c) => (
          <div
            key={c.id}
            className={`conversation-item ${c.id === chat.activeId ? "selected" : ""}`}
          >
            <button
              className="conversation-select"
              title={c.title}
              aria-current={c.id === chat.activeId ? "page" : undefined}
              disabled={chat.busy}
              onClick={() => {
                void chat.select(c.id);
                onSelect();
              }}
            >
              <MessageSquare size={16} />
              <span>{c.title}</span>
            </button>
            <div className="conversation-actions">
              <button
                className="icon-button"
                aria-label={`重命名 ${c.title}`}
                title="重命名"
                disabled={chat.busy}
                onClick={() => onEdit("rename", c)}
              >
                <Pencil size={13} />
              </button>
              <button
                className="icon-button"
                aria-label={`删除 ${c.title}`}
                title="删除"
                disabled={chat.busy}
                onClick={() => onEdit("delete", c)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
        ))}
        {!chat.conversations.length && (
          <p className="empty-sidebar">新建对话，开始查阅法律知识</p>
        )}
      </nav>
      <div className="sidebar-bottom">
        <div className="workspace-note">
          <ShieldCheck size={16} />
          <div>
            <strong>资料安全与权限保护</strong>
            <span>仅检索您有权访问的内容</span>
          </div>
        </div>
        <div className="account">
          <span className="avatar">
            {chat.user?.username.slice(0, 1).toUpperCase()}
          </span>
          <div>
            <strong title={chat.user?.username}>{chat.user?.username}</strong>
            <span>已登录</span>
          </div>
          <button
            className="icon-button"
            aria-label="退出登录"
            title="退出登录"
            disabled={chat.busy}
            onClick={() => void chat.logout()}
          >
            <LogOut size={17} />
          </button>
        </div>
      </div>
    </>
  );
}
function Markdown({ content }: { content: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) renderMarkdown(ref.current, content);
  }, [content]);
  return <div ref={ref} className="markdown" />;
}
function MessageCard({ message }: { message: Message }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-label">
        {message.role === "assistant" ? (
          <>
            <span className="answer-mark">
              <Scale size={17} />
            </span>
            法律知识助手
          </>
        ) : (
          <>您</>
        )}
        {message.pending && (
          <span className="generating">
            <span />
            生成中
          </span>
        )}
      </div>
      {message.role === "assistant" ? (
        <Markdown content={message.content} />
      ) : (
        <div className="user-bubble">{message.content}</div>
      )}
      {!!message.sources?.length && (
        <details className="sources">
          <summary>
            <BookOpen size={15} />
            参考来源 · {message.sources.length}
            <ChevronRight size={15} />
          </summary>
          <div className="source-list">
            {message.sources.map((source, index) => (
              <div className="source" key={index}>
                <span className="source-number">{index + 1}</span>
                <div>
                  <strong>
                    {source.law_name ||
                      source.title ||
                      source.source ||
                      "法规来源"}
                  </strong>
                  <span className="source-location">
                    {source.article || source.section || ""}
                    {source.pages?.length
                      ? ` · 第 ${source.pages.join("、")} 页`
                      : ""}
                  </span>
                  {source.content && <p>{source.content}</p>}
                </div>
              </div>
            ))}
          </div>
        </details>
      )}
    </article>
  );
}
function Messages({
  chat,
  onSuggestion,
}: {
  chat: ChatController;
  onSuggestion: (value: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const [showLatest, setShowLatest] = useState(false);
  const latest = () => {
    const element = ref.current;
    if (element) {
      element.scrollTo({ top: element.scrollHeight, behavior: "auto" });
      following.current = true;
      setShowLatest(false);
    }
  };
  useEffect(() => {
    following.current = true;
    setShowLatest(false);
    latest();
  }, [chat.activeId]);
  useEffect(() => {
    if (following.current) latest();
    else setShowLatest(true);
  }, [chat.messages, chat.status]);
  return (
    <div className="message-region">
      <div
        className="message-scroll"
        ref={ref}
        aria-label="对话消息"
        onScroll={() => {
          const e = ref.current;
          if (e) {
            following.current =
              e.scrollHeight - e.scrollTop - e.clientHeight < 96;
            setShowLatest(!following.current);
          }
        }}
      >
        <div className="message-content">
          {["loading", "history-loading"].includes(chat.phase) ? (
            <div className="center-state" role="status">
              <Spinner />
              <p>
                {chat.phase === "loading"
                  ? "正在加载工作空间"
                  : "正在加载对话记录"}
              </p>
            </div>
          ) : chat.phase === "history-error" || chat.phase === "load-error" ? (
            <div className="center-state">
              <FileText size={30} />
              <h2>
                暂时无法加载
                {chat.phase === "history-error" ? "对话记录" : "工作空间"}
              </h2>
              <p>请重新加载后再继续提问。</p>
            </div>
          ) : chat.messages.length ? (
            chat.messages.map((message) => (
              <MessageCard key={message.id} message={message} />
            ))
          ) : (
            <section className="welcome">
              <span className="welcome-icon">
                <Scale size={34} />
              </span>
              <span className="eyebrow">您的法律知识工作空间</span>
              <h2>今天，有什么法律问题？</h2>
              <p>描述具体情境，获取清晰的解答与法规依据。</p>
              <div className="suggestions">
                {suggestions.map((text, index) => (
                  <button
                    aria-label={text}
                    key={text}
                    onClick={() => onSuggestion(text)}
                  >
                    <span className="suggestion-category">
                      {["劳动用工", "合同管理", "合规经营"][index]}
                    </span>
                    <span>{text}</span>
                    <ChevronRight size={16} />
                  </button>
                ))}
              </div>
              <div className="welcome-note">
                <BookOpen size={14} />
                回答基于可访问资料，来源可在答复中查阅
              </div>
            </section>
          )}
        </div>
      </div>
      {showLatest && (
        <button className="latest-button" onClick={latest}>
          <ArrowDown size={15} />
          回到最新
        </button>
      )}
    </div>
  );
}
function Composer({
  chat,
  inputRef,
}: {
  chat: ChatController;
  inputRef: RefObject<HTMLTextAreaElement | null>;
}) {
  const composing = useRef(false);
  const enabled = chat.phase === "ready" && !chat.busy;
  useEffect(() => {
    const input = inputRef.current;
    if (input) {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight || 48, 160)}px`;
    }
  }, [chat.draft, inputRef]);
  return (
    <div className="composer-area">
      <div className="composer-content">
        <ErrorNotice chat={chat} />
        {chat.status && (
          <div className="stage-status" role="status">
            <Spinner />
            {chat.status}
          </div>
        )}
        <form
          className={`composer ${chat.busy ? "is-busy" : ""}`}
          onSubmit={(event) => {
            event.preventDefault();
            if (enabled && chat.draft.trim()) void chat.send();
          }}
        >
          <textarea
            ref={inputRef}
            aria-label="法律问题"
            maxLength={2000}
            placeholder="输入法律问题，尽量描述具体情况…"
            rows={1}
            value={chat.draft}
            onChange={(e) => chat.setDraft(e.target.value)}
            onCompositionStart={() => {
              composing.current = true;
            }}
            onCompositionEnd={() => {
              composing.current = false;
            }}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing &&
                !composing.current &&
                event.keyCode !== 229
              ) {
                event.preventDefault();
                if (enabled && chat.draft.trim()) void chat.send();
              }
            }}
            disabled={
              chat.busy ||
              chat.phase === "loading" ||
              chat.phase === "history-loading"
            }
          />
          <button
            type="submit"
            className="send-button"
            aria-label="发送问题"
            disabled={!enabled || !chat.draft.trim()}
          >
            {chat.busy ? <Spinner /> : <ArrowUp size={20} />}
          </button>
        </form>
        <div className="composer-caption">
          <span>回答仅供参考，请结合具体情形核实法律依据。</span>
          <span className="keyboard-hint">Enter 发送 · Shift + Enter 换行</span>
        </div>
      </div>
    </div>
  );
}
function EditDialog({
  chat,
  mode,
  conversation,
  onClose,
}: {
  chat: ChatController;
  mode: "rename" | "delete";
  conversation: Conversation;
  onClose: () => void;
}) {
  const [title, setTitle] = useState(conversation.title);
  const [working, setWorking] = useState(false);
  return (
    <Modal
      title={mode === "rename" ? "重命名对话" : "删除对话"}
      onClose={() => {
        if (!working) onClose();
      }}
    >
      <div className="modal-heading">
        <span className={mode === "delete" ? "delete-icon" : "edit-icon"}>
          {mode === "delete" ? <Trash2 size={22} /> : <Pencil size={22} />}
        </span>
        <button
          className="icon-button"
          aria-label="关闭对话框"
          disabled={working}
          onClick={onClose}
        >
          <X size={19} />
        </button>
      </div>
      <h2>{mode === "rename" ? "重命名对话" : "删除对话"}</h2>
      <form
        onSubmit={async (event) => {
          event.preventDefault();
          setWorking(true);
          try {
            if (mode === "delete") {
              await chat.remove(conversation.id);
              onClose();
            } else if (await chat.rename(conversation.id, title.trim()))
              onClose();
          } finally {
            setWorking(false);
          }
        }}
      >
        {mode === "rename" ? (
          <>
            <label htmlFor="conversation-name">对话名称</label>
            <input
              id="conversation-name"
              value={title}
              required
              maxLength={80}
              onChange={(e) => setTitle(e.target.value)}
              disabled={working}
            />
          </>
        ) : (
          <p className="delete-description">
            删除“{conversation.title}”后，对话记录将无法恢复。
          </p>
        )}
        <ErrorNotice chat={chat} />
        <div className="dialog-actions">
          <button
            type="button"
            className="secondary"
            disabled={working}
            onClick={onClose}
          >
            取消
          </button>
          <button
            className={mode === "delete" ? "danger" : "primary"}
            disabled={working || !title.trim()}
          >
            {working ? (
              <Spinner />
            ) : mode === "delete" ? (
              "确认删除"
            ) : (
              "保存名称"
            )}
          </button>
        </div>
      </form>
    </Modal>
  );
}
export function ChatView({ chat }: { chat: ChatController }) {
  const [drawer, setDrawer] = useState(false);
  const [edit, setEdit] = useState<{
    mode: "rename" | "delete";
    conversation: Conversation;
  } | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  if (chat.phase === "checking")
    return (
      <main className="checking-page">
        <Brand />
        <div role="status">
          <Spinner />
          正在检查登录状态
        </div>
      </main>
    );
  if (!chat.user) return <Login chat={chat} />;
  const sidebar = (
    <Sidebar
      chat={chat}
      onSelect={() => setDrawer(false)}
      onEdit={(mode, conversation) => setEdit({ mode, conversation })}
    />
  );
  return (
    <div className="workspace">
      <aside className="sidebar" aria-label="工作空间导航">
        {sidebar}
      </aside>
      <main className="chat-main">
        <header className="conversation-header">
          <button
            className="icon-button menu-button"
            aria-label="打开对话列表"
            onClick={() => setDrawer(true)}
          >
            <Menu size={21} />
          </button>
          <div className="header-title">
            <span>法律法规问答</span>
            <h1 title={chat.title}>{chat.title || "新对话"}</h1>
          </div>
          <span className="header-badge">
            <span />
            <Check size={13} />
            资料权限已保护
          </span>
        </header>
        <Messages
          chat={chat}
          onSuggestion={(value) => {
            chat.setDraft(value);
            inputRef.current?.focus();
          }}
        />
        <Composer chat={chat} inputRef={inputRef} />
      </main>
      {drawer && (
        <Modal title="对话列表" drawer onClose={() => setDrawer(false)}>
          <button
            className="icon-button drawer-close"
            aria-label="关闭对话列表"
            onClick={() => setDrawer(false)}
          >
            <X size={20} />
          </button>
          {sidebar}
        </Modal>
      )}
      {edit && (
        <EditDialog
          key={`${edit.mode}-${edit.conversation.id}`}
          chat={chat}
          {...edit}
          onClose={() => setEdit(null)}
        />
      )}
    </div>
  );
}
