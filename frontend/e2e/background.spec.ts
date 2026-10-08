import { test, expect } from "@playwright/test";
import { createServer, type ServerResponse } from "node:http";

for (const width of [1440, 375]) {
  test(`background SSE survives conversation switches at ${width}px`, async ({
    page,
  }) => {
    let stream: ServerResponse | undefined;
    let finish!: () => void;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const histories: Record<string, unknown[]> = {
      a: [{ role: "assistant", content: "A旧记录" }],
      b: [{ role: "assistant", content: "B旧记录" }],
    };
    const server = createServer((request, response) => {
      response.setHeader(
        "Access-Control-Allow-Origin",
        "http://127.0.0.1:8012",
      );
      response.setHeader(
        "Access-Control-Allow-Headers",
        "Content-Type,X-CSRF-Token",
      );
      response.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
      if (request.method === "OPTIONS") {
        response.writeHead(204);
        response.end();
        return;
      }
      const json = (data: unknown) => {
        response.setHeader("Content-Type", "application/json");
        response.end(JSON.stringify(data));
      };
      if (request.url === "/api/auth/me")
        return json({
          user: { id: "fixture", username: "测试账号" },
          csrf_token: "fixture-csrf",
        });
      if (request.url === "/api/conversations")
        return json({
          conversations: [
            { id: "a", title: "A对话" },
            { id: "b", title: "B对话" },
          ],
        });
      const id = request.url?.includes("/a/") ? "a" : "b";
      if (request.url?.endsWith("/messages"))
        return json({ messages: histories[id] });
      if (request.url === "/api/conversations/a/chat") {
        stream = response;
        response.writeHead(200, {
          "Content-Type": "text/event-stream",
          "X-Request-ID": "background-fixture",
        });
        response.write('event: delta\ndata: {"text":"A正在生成的内容"}\n\n');
        finish = () => {
          const answer = "A最终完整回答";
          const sources = [
            { law_name: "来源快照", content: "A对应的资料原文", pages: [4] },
          ];
          histories.a.push(
            { role: "user", content: "A问题" },
            { role: "assistant", content: answer, sources },
          );
          response.end(
            `event: done\ndata: ${JSON.stringify({ answer, sources })}\n\n`,
          );
        };
        markStarted();
        return;
      }
      response.writeHead(404);
      response.end();
    });
    await new Promise<void>((resolve) =>
      server.listen(0, "127.0.0.1", resolve),
    );
    const address = server.address();
    if (!address || typeof address === "string")
      throw new Error("Missing fixture address");
    const origin = `http://127.0.0.1:${address.port}`;
    try {
      await page.route("**/api/**", (route) =>
        route.continue({
          url: origin + new URL(route.request().url()).pathname,
        }),
      );
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/");
      await expect(page.getByText("A旧记录", { exact: true })).toBeVisible();
      await page.getByRole("textbox", { name: "法律问题" }).fill("A问题");
      await page.getByRole("button", { name: "发送问题" }).click();
      await started;
      await expect(
        page.getByText("A正在生成的内容", { exact: true }),
      ).toBeVisible();
      const select = async (title: string) => {
        if (width < 640)
          await page.getByRole("button", { name: "打开对话列表" }).click();
        await page
          .locator(`button.conversation-select[title="${title}"]:visible`)
          .click();
      };
      await select("B对话");
      await expect(page.locator("h1")).toHaveText("B对话");
      await expect(page.getByText("B旧记录", { exact: true })).toBeVisible();
      await expect(
        page.getByText("A正在生成的内容", { exact: true }),
      ).toHaveCount(0);
      await expect(
        page.getByRole("button", { name: "发送问题" }),
      ).toBeDisabled();
      if (width < 640)
        await page.getByRole("button", { name: "打开对话列表" }).click();
      await expect(
        page.getByRole("status", { name: "A对话 正在生成回答" }),
      ).toBeVisible();
      await page.screenshot({ path: `test-results/background-${width}.png` });
      if (width < 640) await page.keyboard.press("Escape");
      await select("A对话");
      await expect(
        page.getByText("A正在生成的内容", { exact: true }),
      ).toBeVisible();
      await select("B对话");
      finish();
      await expect(
        page.getByText("另一个对话正在生成回答，你可以先浏览历史记录。"),
      ).toHaveCount(0);
      await expect(page.locator("h1")).toHaveText("B对话");
      await expect(page.getByText("B旧记录", { exact: true })).toBeVisible();
      await select("A对话");
      await expect(
        page.getByText("A最终完整回答", { exact: true }),
      ).toBeVisible();
      await page.getByText("参考来源", { exact: false }).click();
      await expect(
        page.getByText("A对应的资料原文", { exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("status", { name: "A对话 正在生成回答" }),
      ).toHaveCount(0);
    } finally {
      stream?.destroy();
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
}
