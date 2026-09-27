"""Structure-aware chunking for procedural guides."""

import json
import re

from langchain_core.documents import Document


CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十]+章\s+.+$")
SECTION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+.+$")
PAGE_NUMBER_RE = re.compile(r"^\d+$")


def prepare_guide(pages, guide):
    """Split a guide at its natural section boundaries and retain headings."""
    chunks = []
    chapter = ""
    current = None

    def flush():
        nonlocal current
        if current is None:
            return
        section = current["section"]
        content = _format_content(section, current["lines"])
        if content:
            metadata = {
                **guide,
                "chapter": current["chapter"],
                "section": section,
                "pages": json.dumps(current["pages"]),
                "source": guide["source_file"],
                "chunk_kind": _chunk_kind(section),
                "index_status": "active",
            }
            prefix = "\n".join((
                f"文档：{guide['title']}",
                f"章节：{current['chapter']}",
                f"小节：{section}",
            ))
            chunks.append(Document(page_content=f"{prefix}\n{content}", metadata=metadata))
        current = None

    for page in pages:
        page_number = page.metadata.get("page", 0) + 1
        if page_number < guide.get("content_start_page", 1):
            continue
        for line in page.page_content.splitlines():
            line = line.strip()
            if not line or line == "ZZPAGEMARKZZ" or PAGE_NUMBER_RE.fullmatch(line):
                continue
            if CHAPTER_RE.fullmatch(line):
                flush()
                chapter = line
                if chapter == "第二章 办理流程":
                    current = {"chapter": chapter, "section": chapter, "lines": [], "pages": [page_number]}
                continue
            if SECTION_RE.fullmatch(line):
                flush()
                current = {"chapter": chapter, "section": line, "lines": [], "pages": [page_number]}
                continue
            if current is not None:
                current["lines"].append(line)
                if page_number not in current["pages"]:
                    current["pages"].append(page_number)
    flush()
    return chunks


def _chunk_kind(section):
    if section == "第二章 办理流程":
        return "procedure"
    if section.startswith("4.1 "):
        return "table"
    if section.startswith("5."):
        return "faq"
    if section.startswith("3."):
        return "checklist"
    if section.startswith("6."):
        return "scenario"
    return "section"


def _format_content(section, lines):
    if section.startswith("5."):
        question = section.split(" ", 1)[1]
        return "\n".join([f"问：{question}", *lines])
    if section.startswith("4.1 "):
        table_rows = []
        for line in lines:
            match = re.match(
                r"^(.+?)\s+(\d+\s*(?:个工作日|日)(?:（可延长\s*\d+\s*日）)?)\s+(.+)$",
                line,
            )
            if match:
                stage, deadline, starts = match.groups()
                table_rows.append(f"环节：{stage}；时限：{deadline}；起算点：{starts}")
        return "\n".join(table_rows)
    return "\n".join(lines)
