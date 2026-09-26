import assert from "node:assert/strict";

import { parseMarkdown } from "../web/static/markdown.mjs";

const blocks = parseMarkdown(
  "根据参考资料：\n\n**试用期最长多久：**\n- 最长不得超过六个月\n- 同一用人单位只能约定一次\n\n> 实习期不等同于试用期"
);

assert.equal(blocks.length, 4);
assert.deepEqual(blocks[0], {
  type: "paragraph",
  lines: [[{ type: "text", value: "根据参考资料：" }]],
});
assert.deepEqual(blocks[1], {
  type: "paragraph",
  lines: [[{ type: "strong", value: "试用期最长多久：" }]],
});
assert.equal(blocks[2].type, "list");
assert.equal(blocks[2].ordered, false);
assert.equal(blocks[2].items.length, 2);
assert.deepEqual(blocks[3], {
  type: "quote",
  lines: [[{ type: "text", value: "实习期不等同于试用期" }]],
});
