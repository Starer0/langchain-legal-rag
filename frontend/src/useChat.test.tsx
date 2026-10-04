import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { useChat } from "./useChat";
import { StrictMode } from "react";

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
