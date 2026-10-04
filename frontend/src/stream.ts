export type StreamEvent = { event: string; data: Record<string, any> };

export async function receiveStream(
  response: Response,
  onEvent: (event: StreamEvent) => void,
) {
  if (!response.body) throw new Error("回答连接不可用，请重试。");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let ended = false;
  function frame(value: string) {
    let event = "message";
    const data: string[] = [];
    for (const line of value.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (data.length) onEvent({ event, data: JSON.parse(data.join("\n")) });
  }
  try {
    while (!ended) {
      const chunk = await reader.read();
      ended = chunk.done;
      buffer += decoder.decode(chunk.value, { stream: !ended });
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || "";
      parts.forEach(frame);
    }
    if (buffer.trim()) frame(buffer);
  } finally {
    if (!ended) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
