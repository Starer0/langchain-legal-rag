import { test, expect } from "@playwright/test";
import { spawn } from "node:child_process";
import { createServer } from "node:net";

for (const width of [1440, 375]) {
  test(`real background task survives refresh and browser close at ${width}px`, async ({ context }) => {
    const reservation = createServer();
    await new Promise<void>(r => reservation.listen(0, "127.0.0.1", r));
    const address = reservation.address();
    if (!address || typeof address === "string") throw new Error("No port");
    const port = address.port;
    await new Promise<void>(r => reservation.close(() => r()));
    const origin = `http://127.0.0.1:${port}`;
    const child = spawn("D:/conda/python.exe", ["tests/fixtures/generation_server.py", String(port)], { cwd: "..", stdio: "ignore" });
    try {
      await expect.poll(async () => {
        try { return (await fetch(origin + "/fixture")).status; } catch { return 0; }
      }, { timeout: 20000 }).toBe(200);
      const fixture = await (await fetch(origin + "/fixture")).json();
      await context.addCookies([{ name: "legal_rag_session", value: "fixture", domain: "127.0.0.1", path: "/" }]);
      await context.route("**/api/**", route => route.continue({ url: origin + new URL(route.request().url()).pathname }));
      let page = await context.newPage();
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/");
      const selectA = async () => {
        if (width < 640) await page.getByRole("button", { name: "打开对话列表" }).click();
        await page.locator('button.conversation-select[title="A对话"]:visible').click();
      };
      await expect(page.getByText("B旧记录", { exact: true })).toBeVisible();
      await selectA();
      await page.getByRole("textbox", { name: "法律问题" }).fill("那实习期呢");
      await page.getByRole("button", { name: "发送问题" }).click();
      await expect(page.getByText("后台正在生成的内容", { exact: true })).toBeVisible();
      await page.reload();
      await expect(page.getByText("那实习期呢", { exact: true })).toBeVisible();
      await expect(page.getByText("后台正在生成的内容", { exact: true })).toBeVisible();
      await expect(page.getByRole("button", { name: "发送问题" })).toBeDisabled();
      await page.close();
      page = await context.newPage();
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/");
      await expect(page.getByText("后台正在生成的内容", { exact: true })).toBeVisible();
      expect((await (await fetch(origin + "/fixture")).json()).calls).toBe(1);
      if (width < 640) await page.getByRole("button", { name: "打开对话列表" }).click();
      await page.locator('button.conversation-select[title="B对话"]:visible').click();
      await expect(page.getByText("B旧记录", { exact: true })).toBeVisible();
      await fetch(origin + "/fixture/release", { method: "POST" });
      await expect(page.getByText("另一个对话正在生成回答，你可以先浏览历史记录。")).toHaveCount(0);
      await selectA();
      await expect(page.getByText("后台正在生成的内容，最终完整回答", { exact: true })).toBeVisible();
      await page.getByText("参考来源", { exact: false }).click();
      await expect(page.getByText("后台任务来源", { exact: true })).toBeVisible();
      await page.reload();
      await expect(page.getByText("那实习期呢", { exact: true })).toHaveCount(1);
      expect((await (await fetch(origin + "/fixture")).json()).calls).toBe(1);
      const response = await fetch(origin + `/api/conversations/${fixture.a}/messages`, { headers: { Cookie: "legal_rag_session=fixture" } });
      expect((await response.json()).messages).toHaveLength(2);
      await page.screenshot({ path: `test-results/tasks-${width}.png` });
    } finally {
      child.kill();
    }
  });
}
