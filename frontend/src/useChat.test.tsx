import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { useChat } from "./useChat";
import { StrictMode } from "react";

test("recovered task displays context preparation separately from answer generation", async () => {
  const original = fetch;
  const task = { id: "context-task", conversation_id: "conv-a", question: "追问", status: "running", stage: "context", answer: "", request_id: "context-request" };
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url === "/api/conversations") return json({ background_tasks: true, conversations: [{ id: "conv-a", title: "A" }] });
    if (url.endsWith("/generation/active")) return json({ task });
    if (url.endsWith("conv-a/messages")) return json({ messages: [], task });
    if (url.endsWith("/tasks/context-task")) return json(task);
    return original(url);
  }));
  const { result, unmount } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  expect(result.current.status).toBe("正在整理对话上下文");
  unmount();
});

function heldStream() {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(value) {
        controller = value;
      },
    }),
    {
      headers: {
        "Content-Type": "text/event-stream",
        "X-Request-ID": "background-id",
      },
    },
  );
  return {
    response,
    emit(event: string, data: unknown) {
      controller.enqueue(
        new TextEncoder().encode(
          `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
        ),
      );
    },
    close() {
      controller.close();
    },
  };
}

test("fresh page recovers a running server task without submitting a second question", async () => {
  const original = fetch;
  const task = { id: "server-task", conversation_id: "conv-a", question: "已发送问题", status: "running", stage: "answer", answer: "已生成部分", request_id: "task-request" };
  const submit = vi.fn();
  vi.stubGlobal("fetch", vi.fn(async (url: string, options?: RequestInit) => {
    if (url === "/api/conversations") return json({ background_tasks: true, conversations: [{ id: "conv-a", title: "A" }, { id: "conv-b", title: "B" }] });
    if (url.endsWith("/tasks") && options?.method === "POST") { submit(); return json(task); }
    if (url.endsWith("/generation/active")) return json({ task });
    if (url.endsWith("/tasks/server-task")) return json(task);
    if (url.endsWith("conv-a/messages")) return json({ messages: [], task });
    return original(url, options);
  }));
  const { result, unmount } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  expect(result.current.messages.map(m => m.content)).toEqual(["已发送问题", "已生成部分"]);
  expect(result.current.runningConversationId).toBe("conv-a");
  expect(submit).not.toHaveBeenCalled();
  unmount();
});

test("switching conversations keeps the original stream alive without contaminating the selected history", async () => {
  const stream = heldStream();
  const original = fetch;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/chat")) return stream.response;
      if (url.includes("conv-b/messages"))
        return json({ messages: [{ role: "assistant", content: "B旧回答" }] });
      return original(url, options);
    }),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  act(() => result.current.setDraft("A问题"));
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.send();
  });
  await act(() => result.current.select("conv-b"));
  expect(result.current.activeId).toBe("conv-b");
  expect(result.current.runningConversationId).toBe("conv-a");
  await act(async () => stream.emit("delta", { text: "A部分回答" }));
  expect(result.current.messages.map((m) => m.content)).toEqual(["B旧回答"]);
  await act(() => result.current.select("conv-a"));
  expect(result.current.messages.at(-1)?.content).toBe("A部分回答");
  await act(() => result.current.select("conv-b"));
  await act(async () => {
    stream.emit("done", {
      answer: "A完整回答",
      sources: [{ content: "A来源" }],
    });
    stream.close();
    await pending;
  });
  expect(result.current.activeId).toBe("conv-b");
  expect(result.current.messages.map((m) => m.content)).toEqual(["B旧回答"]);
  expect(result.current.runningConversationId).toBeNull();
  expect(result.current.busy).toBe(false);
});

test("background stream failure retains its question and request ID only in that conversation", async () => {
  const stream = heldStream();
  const original = fetch;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) =>
      url.endsWith("/chat") ? stream.response : original(url, options),
    ),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  act(() => result.current.setDraft("A失败问题"));
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.send();
  });
  await act(() => result.current.select("conv-b"));
  await act(async () => {
    stream.emit("delta", { text: "半个回答" });
    stream.close();
    await pending;
  });
  expect(result.current.activeId).toBe("conv-b");
  expect(result.current.error).toBe("");
  expect(result.current.draft).toBe("");
  await act(() => result.current.select("conv-a"));
  expect(result.current.draft).toBe("A失败问题");
  expect(result.current.error).toContain("background-id");
  expect(result.current.messages).toEqual([]);
});

const identity = { user: { id: "A", username: "demo_ab" }, csrf_token: "csrf" };
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
let messages: unknown[];
beforeEach(() => {
  messages = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (url === "/api/auth/me") return json(identity);
      if (url === "/api/conversations")
        return json({
          conversations: [
            { id: "conv-a", title: "旧标题" },
            { id: "conv-b", title: "另一对话" },
          ],
        });
      if (url.endsWith("/messages")) return json({ messages });
      if (url.endsWith("/chat")) {
        return new Response(
          'event: delta\ndata: {"text":"完整答案"}\n\nevent: done\ndata: {"answer":"完整答案","sources":[]}\n\n',
          {
            headers: {
              "Content-Type": "text/event-stream",
              "X-Request-ID": "server-id",
            },
          },
        );
      }
      if (url === "/api/auth/logout")
        return new Response(null, { status: 204 });
      return json({}, 404);
    }),
  );
});
test("loads authenticated history and saves completed stream without losing existing messages", async () => {
  messages = [
    { role: "user", content: "旧问题" },
    { role: "assistant", content: "旧回答" },
  ];
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  act(() => result.current.setDraft("新问题"));
  await act(() => result.current.send());
  expect(result.current.messages.map((m) => m.content)).toEqual([
    "旧问题",
    "旧回答",
    "新问题",
    "完整答案",
  ]);
  expect(result.current.busy).toBe(false);
});
test("stream EOF without done removes partial output, restores draft and shows request ID", async () => {
  const original = fetch;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) =>
      url.endsWith("/chat")
        ? new Response('event: delta\ndata: {"text":"半个答案"}\n\n', {
            headers: {
              "Content-Type": "text/event-stream",
              "X-Request-ID": "failed-id",
            },
          })
        : original(url, options),
    ),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  act(() => result.current.setDraft("保留问题"));
  await act(() => result.current.send());
  expect(result.current.messages).toEqual([]);
  expect(result.current.draft).toBe("保留问题");
  expect(result.current.error).toContain("failed-id");
});
test("history read failure disables ready phase and retry restores history", async () => {
  const original = fetch;
  let fail = true;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) =>
      url.endsWith("/messages") && fail
        ? json({ detail: "暂时不可用" }, 503)
        : original(url, options),
    ),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("history-error"));
  act(() => result.current.setDraft("不要发送"));
  await act(() => result.current.send());
  expect(result.current.messages).toEqual([]);
  fail = false;
  await act(() => result.current.retry());
  expect(result.current.phase).toBe("ready");
});
test("late history from another conversation cannot overwrite selected conversation", async () => {
  let resolve!: (value: Response) => void;
  const original = fetch;
  let delay = false;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) =>
      url.includes("conv-a/messages") && delay
        ? new Promise<Response>((r) => {
            resolve = r;
          })
        : url.includes("conv-b/messages")
          ? json({ messages: [{ role: "user", content: "B 的问题" }] })
          : original(url, options),
    ),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  await act(() => result.current.select("conv-b"));
  delay = true;
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.select("conv-a");
  });
  await act(() => result.current.select("conv-b"));
  await act(async () => {
    resolve(json({ messages: [{ role: "user", content: "迟到的 A 内容" }] }));
    await pending;
  });
  expect(result.current.activeId).toBe("conv-b");
  expect(result.current.messages[0].content).toBe("B 的问题");
});

test("StrictMode effect cleanup does not strand initialization", async () => {
  const { result } = renderHook(() => useChat(), { wrapper: StrictMode });
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  expect(result.current.user?.id).toBe("A");
});

test("a delayed old-account history cannot expose A after identity changes to B", async () => {
  let resolve!: (value: Response) => void;
  const original = fetch;
  let delay = false;
  let accountB = false;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (accountB && url === "/api/auth/me")
        return json({
          user: { id: "B", username: "demo_c" },
          csrf_token: "csrf-b",
        });
      if (accountB && url === "/api/conversations")
        return json({ conversations: [{ id: "private-b", title: "B 对话" }] });
      if (url.includes("private-b/messages"))
        return json({ messages: [{ role: "user", content: "B 私人问题" }] });
      if (delay && url.includes("conv-b/messages"))
        return new Promise<Response>((r) => {
          resolve = r;
        });
      return original(url, options);
    }),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  delay = true;
  let oldHistory!: Promise<void>;
  act(() => {
    oldHistory = result.current.select("conv-b");
  });
  accountB = true;
  await act(() => result.current.check());
  expect(result.current.user?.id).toBe("B");
  await act(async () => {
    resolve(
      json({ messages: [{ role: "assistant", content: "A 的私人回答" }] }),
    );
    await oldHistory;
  });
  expect(result.current.messages.map((m) => m.content)).toEqual(["B 私人问题"]);
});

test("list refresh failure after completed answer never removes the saved answer", async () => {
  const original = fetch;
  let completed = false;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/chat")) {
        completed = true;
        return original(url, options);
      }
      if (url === "/api/conversations" && completed) return json({}, 503);
      return original(url, options);
    }),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  act(() => result.current.setDraft("新问题"));
  await act(() => result.current.send());
  expect(result.current.messages.at(-1)?.content).toBe("完整答案");
  expect(result.current.error).toContain("回答已保存");
});

test("successful deletion invalidates active history even if list refresh fails", async () => {
  const original = fetch;
  let deleted = false;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (options?.method === "DELETE") {
        deleted = true;
        return new Response(null, { status: 204 });
      }
      if (url === "/api/conversations" && deleted) return json({}, 503);
      return original(url, options);
    }),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  await act(() => result.current.remove("conv-a"));
  expect(result.current.activeId).toBeNull();
  expect(result.current.phase).toBe("load-error");
  act(() => result.current.setDraft("不可发送"));
  await act(() => result.current.send());
  expect(
    vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith("/chat")),
  ).toBe(false);
});

test("transport error after done preserves the persisted answer without a retry draft", async () => {
  const original = fetch;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (!url.endsWith("/chat")) return original(url, options);
      let pulled = false;
      return new Response(
        new ReadableStream({
          pull(controller) {
            if (pulled) controller.error(new Error("connection closed"));
            else {
              pulled = true;
              controller.enqueue(
                new TextEncoder().encode(
                  'event: done\ndata: {"answer":"已保存答案","sources":[]}\n\n',
                ),
              );
            }
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      );
    }),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  act(() => result.current.setDraft("问题"));
  await act(() => result.current.send());
  expect(result.current.messages.at(-1)?.content).toBe("已保存答案");
  expect(result.current.draft).toBe("");
});

test("late logout from A cannot unlock an active B answer", async () => {
  const original = fetch;
  let accountB = false;
  let resolveLogout!: (response: Response) => void;
  let closeStream!: () => void;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options?: RequestInit) => {
      if (url === "/api/auth/logout")
        return new Promise<Response>((resolve) => {
          resolveLogout = resolve;
        });
      if (accountB && url === "/api/auth/me")
        return json({
          user: { id: "B", username: "demo_c" },
          csrf_token: "csrf-b",
        });
      if (url.endsWith("/chat"))
        return new Response(
          new ReadableStream({
            start(controller) {
              closeStream = () => {
                controller.enqueue(
                  new TextEncoder().encode(
                    'event: done\ndata: {"answer":"B答案","sources":[]}\n\n',
                  ),
                );
                controller.close();
              };
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        );
      return original(url, options);
    }),
  );
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.phase).toBe("ready"));
  let logout!: Promise<void>;
  act(() => {
    logout = result.current.logout();
  });
  accountB = true;
  await act(() => result.current.check());
  act(() => result.current.setDraft("B问题"));
  let send!: Promise<void>;
  act(() => {
    send = result.current.send();
  });
  await waitFor(() => expect(closeStream).toBeDefined());
  await act(async () => {
    resolveLogout(new Response(null, { status: 204 }));
    await logout;
  });
  expect(result.current.busy).toBe(true);
  expect(result.current.error).toBe("");
  await act(async () => {
    closeStream();
    await send;
  });
});
