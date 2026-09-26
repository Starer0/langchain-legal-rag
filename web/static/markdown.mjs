export function parseMarkdown(markdown) {
  const blocks = [];
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: "paragraph", lines: paragraph });
    paragraph = [];
  };
  const flushList = () => {
    if (list) blocks.push(list);
    list = null;
  };

  for (const rawLine of String(markdown).replaceAll("\r\n", "\n").split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    const listMatch = line.match(/^(?:- |\* |(\d+)\. )(.+)$/);
    if (listMatch) {
      flushParagraph();
      const ordered = Boolean(listMatch[1]);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { type: "list", ordered, items: [] };
      }
      list.items.push(parseInline(listMatch[2]));
      continue;
    }

    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        content: parseInline(headingMatch[2]),
      });
      continue;
    }

    if (line.startsWith("> ")) {
      flushParagraph();
      flushList();
      blocks.push({ type: "quote", lines: [parseInline(line.slice(2))] });
      continue;
    }

    flushList();
    paragraph.push(parseInline(line));
  }

  flushParagraph();
  flushList();
  return blocks;
}

export function renderMarkdown(target, markdown) {
  target.replaceChildren();
  for (const block of parseMarkdown(markdown)) {
    if (block.type === "paragraph") {
      const element = document.createElement("p");
      appendLines(element, block.lines);
      target.append(element);
    }
    if (block.type === "heading") {
      const element = document.createElement(`h${block.level + 2}`);
      appendInline(element, block.content);
      target.append(element);
    }
    if (block.type === "list") {
      const element = document.createElement(block.ordered ? "ol" : "ul");
      for (const item of block.items) {
        const listItem = document.createElement("li");
        appendInline(listItem, item);
        element.append(listItem);
      }
      target.append(element);
    }
    if (block.type === "quote") {
      const element = document.createElement("blockquote");
      appendLines(element, block.lines);
      target.append(element);
    }
  }
}

function parseInline(text) {
  const parts = [];
  const matcher = /\*\*(.+?)\*\*/g;
  let cursor = 0;
  for (const match of text.matchAll(matcher)) {
    if (match.index > cursor) parts.push({ type: "text", value: text.slice(cursor, match.index) });
    parts.push({ type: "strong", value: match[1] });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push({ type: "text", value: text.slice(cursor) });
  return parts.length ? parts : [{ type: "text", value: text }];
}

function appendLines(target, lines) {
  lines.forEach((line, index) => {
    if (index) target.append(document.createElement("br"));
    appendInline(target, line);
  });
}

function appendInline(target, parts) {
  for (const part of parts) {
    const element = part.type === "strong" ? document.createElement("strong") : null;
    if (element) {
      element.textContent = part.value;
      target.append(element);
    } else {
      target.append(document.createTextNode(part.value));
    }
  }
}
