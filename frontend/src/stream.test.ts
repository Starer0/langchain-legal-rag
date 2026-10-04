import { expect, test } from "vitest";
import { receiveStream } from "./stream";

function response(parts: string[]) {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const part of parts) controller.enqueue(encoder.encode(part));
        controller.close();
      },
    }),
  );
}
test("SSE handles split lines, CRLF, multibyte text and final frame without newline", async () => {
  const events: unknown[] = [];
  await receiveStream(
    response([
      'event: delta\r\ndata: {"text":"试',
      '用期"}\r\n\r\n',
      'event: done\ndata: {"answer":"完成"}',
    ]),
    (item) => events.push(item),
  );
  expect(events).toEqual([
    { event: "delta", data: { text: "试用期" } },
    { event: "done", data: { answer: "完成" } },
  ]);
});
test("malformed JSON rejects the stream instead of declaring success", async () => {
  await expect(
    receiveStream(response(["event: delta\ndata: bad\n\n"]), () => {}),
  ).rejects.toThrow();
});
test("callback failure cancels the reader and propagates failure", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode("event: error\ndata: {}\n\n"),
      );
    },
    cancel() {
      cancelled = true;
    },
  });
  await expect(
    receiveStream(new Response(stream), () => {
      throw new Error("failure");
    }),
  ).rejects.toThrow("failure");
  expect(cancelled).toBe(true);
});
