// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatView } from "./components";
import type { ChatController } from "./types";
afterEach(cleanup);

function controller(overrides: Partial<ChatController> = {}): ChatController {
  return {
    user: { id: "u", username: "测试用户" },
    phase: "ready",
    conversations: [{ id: "c", title: "劳动合同问题" }],
    activeId: "c",
    messages: [],
    draft: "",
    title: "劳动合同问题",
    status: "",
    error: "",
    busy: false,
    runningConversationId: null,
    loginBusy: false,
    setDraft: vi.fn(),
    login: vi.fn(async () => {}),
    logout: vi.fn(async () => {}),
    check: vi.fn(async () => {}),
    retry: vi.fn(async () => {}),
    select: vi.fn(async () => {}),
    create: vi.fn(async () => {}),
    rename: vi.fn(async () => true),
    remove: vi.fn(async () => {}),
    send: vi.fn(async () => {}),
    ...overrides,
  };
}
describe("ChatView", () => {
  it("opens account memory independently and shows degradation outside the answer", async () => {
    const request = vi.fn(async () => ({core_text:"先说结论", extended_text:"", enabled:true, revision:1, updated_at:null}));
    render(<ChatView chat={controller({memoryRequest:request, memoryWarning:"本次未使用扩展记忆", messages:[{id:"a",role:"assistant",content:"法律答案",sources:[]}]})} />);
    expect(screen.getByText("本次未使用扩展记忆")).toHaveAttribute("role", "status");
    expect(screen.getByText("法律答案").closest("article")).not.toHaveTextContent("本次未使用扩展记忆");
    await userEvent.click(screen.getByRole("button", {name:"长期记忆"}));
    expect(await screen.findByRole("dialog", {name:"长期记忆"})).toBeVisible();
    expect(await screen.findByLabelText("核心记忆")).toHaveValue("先说结论");
    expect(request).toHaveBeenCalledWith("/api/memory");
  });
  it("allows viewing another conversation and identifies the background generating row", async () => {
    const chat = controller({
      busy: true,
      runningConversationId: "c",
      conversations: [
        { id: "c", title: "劳动合同问题" },
        { id: "other", title: "其他对话" },
      ],
    });
    render(<ChatView chat={chat} />);
    expect(
      screen.getByRole("status", { name: "劳动合同问题 正在生成回答" }),
    ).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "其他对话" }));
    expect(chat.select).toHaveBeenCalledWith("other");
  });
  it("fills a suggestion without sending it", async () => {
    const chat = controller();
    render(<ChatView chat={chat} />);
    await userEvent.click(
      screen.getByRole("button", { name: "试用期最长可以约定多久？" }),
    );
    expect(chat.setDraft).toHaveBeenCalledWith("试用期最长可以约定多久？");
    expect(chat.send).not.toHaveBeenCalled();
  });
  it("does not send while confirming Chinese composition and sends on normal Enter", () => {
    const chat = controller({ draft: "法律问题" });
    render(<ChatView chat={chat} />);
    const input = screen.getByRole("textbox", { name: "法律问题" });
    fireEvent.compositionStart(input);
    fireEvent.keyDown(input, { key: "Enter" });
    expect(chat.send).not.toHaveBeenCalled();
    fireEvent.compositionEnd(input);
    fireEvent.keyDown(input, { key: "Enter" });
    expect(chat.send).toHaveBeenCalledOnce();
  });
  it("keeps history errors visible, restores draft and disables sending until retry", async () => {
    const chat = controller({
      phase: "history-error",
      error: "加载失败 · 请求编号 abc",
      draft: "恢复的问题",
    });
    render(<ChatView chat={chat} />);
    expect(screen.getByRole("textbox", { name: "法律问题" })).toHaveValue(
      "恢复的问题",
    );
    expect(screen.getByRole("button", { name: "发送问题" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("abc");
    await userEvent.click(screen.getByRole("button", { name: "重新加载" }));
    expect(chat.retry).toHaveBeenCalledOnce();
  });
  it("renders long titles and safe Markdown with expandable sources", async () => {
    const title = "超长标题".repeat(50);
    render(
      <ChatView
        chat={controller({
          title,
          conversations: [{ id: "c", title }],
          messages: [
            {
              id: "m",
              role: "assistant",
              content: "**依据** <img src=x onerror=alert(1)>",
              sources: [
                {
                  law_name: "劳动合同法",
                  article: "第十九条",
                  content: "试用期条文",
                },
              ],
            },
          ],
        })}
      />,
    );
    expect(screen.getByRole("heading", { name: title })).toHaveAttribute(
      "title",
      title,
    );
    expect(document.querySelector(".markdown strong")).toHaveTextContent(
      "依据",
    );
    expect(document.querySelector(".markdown img")).toBeNull();
    await userEvent.click(screen.getByText("参考来源 · 1"));
    expect(screen.getByText("试用期条文")).toBeVisible();
  });
  it("requires confirmation before deleting and restores focus after cancelling", async () => {
    const chat = controller();
    render(<ChatView chat={chat} />);
    const trigger = screen.getByRole("button", { name: "删除 劳动合同问题" });
    await userEvent.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "删除对话" });
    await userEvent.click(within(dialog).getByRole("button", { name: "取消" }));
    expect(chat.remove).not.toHaveBeenCalled();
    expect(trigger).toHaveFocus();
    await userEvent.click(trigger);
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "确认删除",
      }),
    );
    expect(chat.remove).toHaveBeenCalledWith("c");
  });
  it("submits login credentials and shows checking state", async () => {
    const chat = controller({
      user: null,
      phase: "login",
      error: "账号或密码错误",
    });
    const view = render(<ChatView chat={chat} />);
    await userEvent.type(screen.getByLabelText("用户名"), "demo");
    await userEvent.type(screen.getByLabelText("密码"), "secret");
    await userEvent.click(screen.getByRole("button", { name: "登录" }));
    expect(chat.login).toHaveBeenCalledWith("demo", "secret");
    view.rerender(
      <ChatView chat={controller({ phase: "checking", user: null })} />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("正在检查登录状态");
  });
  it("opens an accessible drawer, traps focus and closes with Escape", async () => {
    render(<ChatView chat={controller()} />);
    const trigger = screen.getByRole("button", { name: "打开对话列表" });
    await userEvent.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "对话列表" });
    expect(dialog).toBeVisible();
    const last = within(dialog).getByRole("button", { name: "退出登录" });
    last.focus();
    fireEvent.keyDown(last, { key: "Tab" });
    expect(
      within(dialog).getByRole("button", { name: "关闭对话列表" }),
    ).toHaveFocus();
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(trigger).toHaveFocus();
  });
  it("renames the selected conversation using its existing title", async () => {
    const chat = controller();
    render(<ChatView chat={chat} />);
    await userEvent.click(
      screen.getByRole("button", { name: "重命名 劳动合同问题" }),
    );
    const dialog = screen.getByRole("dialog", { name: "重命名对话" });
    const input = within(dialog).getByLabelText("对话名称");
    expect(input).toHaveValue("劳动合同问题");
    await userEvent.clear(input);
    await userEvent.type(input, "新的标题");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "保存名称" }),
    );
    expect(chat.rename).toHaveBeenCalledWith("c", "新的标题");
  });
  it("keeps rename failures visible inside its dialog", async () => {
    const chat = controller({
      error: "名称保存失败",
      rename: vi.fn(async () => false),
    });
    render(<ChatView chat={chat} />);
    await userEvent.click(
      screen.getByRole("button", { name: "重命名 劳动合同问题" }),
    );
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "保存名称",
      }),
    );
    expect(
      within(screen.getByRole("dialog")).getByRole("alert"),
    ).toHaveTextContent("名称保存失败");
  });
  it("does not pull a reader away from older messages during streaming", () => {
    const chat = controller({
      messages: [{ id: "m", role: "assistant", content: "旧回答" }],
    });
    const view = render(<ChatView chat={chat} />);
    const log = screen.getByLabelText("对话消息");
    Object.defineProperties(log, {
      scrollHeight: { value: 2000, configurable: true },
      clientHeight: { value: 500, configurable: true },
      scrollTop: { value: 0, writable: true },
    });
    fireEvent.scroll(log);
    const scroll = vi.spyOn(log, "scrollTo");
    scroll.mockClear();
    view.rerender(
      <ChatView
        chat={{
          ...chat,
          messages: [
            { id: "m", role: "assistant", content: "旧回答 + 新内容" },
          ],
        }}
      />,
    );
    expect(scroll).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "回到最新" })).toBeVisible();
  });
});
