"""Versioned PDF ingestion and explicit metadata scope for the legal corpus."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag_pipeline_articles import split_by_articles


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "laws.json"
DB_DIR = ROOT / "chroma_legal_db"
SCHEMA_VERSION = "v7-articles-1"
NUMERALS = "零一二三四五六七八九"
ARTICLE_REF = re.compile(r"第\s*([零〇一二三四五六七八九十百千0-9\s]+?)\s*条")


def article_number(value):
    text = re.sub(r"\s", "", value).removeprefix("第").removesuffix("条")
    if text.isdigit():
        return int(text)
    total = digit = 0
    for char in text.replace("〇", "零"):
        if char in NUMERALS:
            digit = NUMERALS.index(char)
        elif char in "十百千":
            total += (digit or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            digit = 0
        else:
            raise ValueError(f"不支持的条号：{value}")
    return total + digit


def article_label(number):
    if not 1 <= number <= 9999:
        raise ValueError("法条号必须在 1 到 9999 之间")
    output = ""
    for unit, label in [(1000, "千"), (100, "百"), (10, "十"), (1, "")]:
        digit, number = divmod(number, unit)
        if digit:
            output += NUMERALS[digit] + label
        elif output and number and not output.endswith("零"):
            output += "零"
    if output.startswith("一十"):
        output = output[1:]
    return f"第{output}条"


def load_catalog(catalog_path=CATALOG_PATH):
    catalog_path = Path(catalog_path)
    laws = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(laws, list) or not laws:
        raise ValueError("法律目录必须是非空列表")
    ids, files = set(), set()
    for law in laws:
        for key in ("law_id", "law_name", "source_file", "version", "effective_date", "status", "source_url"):
            if not isinstance(law.get(key), str) or not law[key].strip():
                raise ValueError(f"法律目录缺少 {key}")
        if law["law_id"] in ids or law["source_file"] in files:
            raise ValueError("法律 ID 或文件重复登记")
        if not isinstance(law.get("expected_articles"), int) or law["expected_articles"] < 1:
            raise ValueError("目录必须登记法条数量")
        if not isinstance(law.get("aliases"), list) or not all(isinstance(a, str) and a for a in law["aliases"]):
            raise ValueError("法律别名必须是字符串列表")
        source = (catalog_path.parent / law["source_file"]).resolve()
        if source.parent != catalog_path.parent.resolve() or source.suffix.lower() != ".pdf":
            raise ValueError("PDF 必须位于法律目录所在文件夹")
        if not source.is_file():
            raise ValueError(f"缺少 PDF：{source.name}")
        ids.add(law["law_id"])
        files.add(law["source_file"])
    extras = {p.name for p in catalog_path.parent.iterdir() if p.suffix.lower() == ".pdf"} - files
    if extras:
        raise ValueError(f"请先在 laws.json 登记新增 PDF：{sorted(extras)}")
    return laws


def prepare_law(pages, law):
    cleaned = []
    stop = False
    for page in pages:
        lines = []
        for line in page.page_content.splitlines():
            line = line.strip()
            if line == "文档信息":
                stop = True
                break
            if line == law["law_name"] or re.fullmatch(r"(?:第\s*)?\d+\s*(?:页)?", line):
                continue
            lines.append(line)
        cleaned.append(Document(page_content="\n".join(lines), metadata=dict(page.metadata)))
        if stop:
            break
    chunks = split_by_articles(cleaned)
    numbers = [article_number(doc.metadata["article"]) for doc in chunks]
    if numbers != list(range(1, law["expected_articles"] + 1)):
        raise ValueError(f"{law['law_name']} 法条缺失、重复或顺序异常：预期 {law['expected_articles']}，实际 {len(chunks)}")
    for doc in chunks:
        article = doc.metadata["article"]
        if not doc.page_content[len(article):].strip():
            raise ValueError(f"{law['law_name']} {article} 正文为空")
        doc.metadata.update({k: law[k] for k in (
            "law_id", "law_name", "source_file", "source_url", "version", "effective_date", "status",
        )})
        doc.metadata["pages"] = json.dumps(doc.metadata["pages"])
        doc.metadata["source"] = law["source_file"]
    return chunks


def corpus_fingerprint(laws, embedding_config, catalog_path=CATALOG_PATH):
    files = {law["source_file"]: hashlib.sha256(
        (Path(catalog_path).parent / law["source_file"]).read_bytes()
    ).hexdigest() for law in laws}
    inputs = {"schema": SCHEMA_VERSION, "embedding": embedding_config, "laws": laws, "files": files}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return digest, files


def _store(embeddings, db_dir, collection):
    return Chroma(collection_name=collection, persist_directory=str(db_dir), embedding_function=embeddings)


def ingest_corpus(embeddings, embedding_config, catalog_path=CATALOG_PATH, db_dir=DB_DIR):
    laws = load_catalog(catalog_path)
    fingerprint, files = corpus_fingerprint(laws, embedding_config, catalog_path)
    db_dir = Path(db_dir)
    manifest_path = db_dir / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] == fingerprint:
            store = _store(embeddings, db_dir, saved["collection"])
            if store._collection.count() == saved["article_count"]:
                return saved
    chunks = []
    counts = {}
    for law in laws:
        pages = PyPDFLoader(str(Path(catalog_path).parent / law["source_file"])).load()
        articles = prepare_law(pages, law)
        chunks.extend(articles)
        counts[law["law_id"]] = len(articles)
    collection = f"legal_{fingerprint[:24]}"
    store = _store(embeddings, db_dir, collection)
    ids = [f"{d.metadata['law_id']}:{d.metadata['version']}:{d.metadata['article']}" for d in chunks]
    existing = set(store.get(include=[])["ids"])
    pending = [(doc, doc_id) for doc, doc_id in zip(chunks, ids) if doc_id not in existing]
    for offset in range(0, len(pending), 32):
        batch = pending[offset:offset + 32]
        store.add_documents([item[0] for item in batch], ids=[item[1] for item in batch])
        print(f"导入法条：{min(offset + 32, len(pending))}/{len(pending)}", flush=True)
    if set(store.get(include=[])["ids"]) != set(ids):
        raise RuntimeError("索引数量或法条 ID 校验失败，未发布新索引")
    manifest = {
        "schema": SCHEMA_VERSION, "fingerprint": fingerprint, "collection": collection,
        "embedding": embedding_config, "files": files, "law_counts": counts,
        "article_count": len(chunks), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = db_dir / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def open_corpus(embeddings, embedding_config, catalog_path=CATALOG_PATH, db_dir=DB_DIR):
    laws = load_catalog(catalog_path)
    fingerprint, _ = corpus_fingerprint(laws, embedding_config, catalog_path)
    path = Path(db_dir) / "manifest.json"
    if not path.exists():
        raise ValueError("尚未导入知识库，请运行 python main.py --ingest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["fingerprint"] != fingerprint:
        raise ValueError("PDF、目录或 Embedding 配置已变化，请重新导入：python main.py --ingest")
    store = _store(embeddings, db_dir, manifest["collection"])
    if store._collection.count() != manifest["article_count"]:
        raise ValueError("知识库不完整，请重新导入：python main.py --ingest")
    return store, manifest, laws


def _named_laws(text, laws):
    aliases = {name: law["law_id"] for law in laws for name in [law["law_name"], *law["aliases"]]}
    selected = []
    for title in re.findall(r"《([^》]+)》", text):
        selected.append(aliases.get(title, "__unknown_law__"))
    remaining = re.sub(r"《[^》]+》", "", text)
    for name in sorted(aliases, key=len, reverse=True):
        pattern = re.escape(name) + r"(?!实施条例|实施细则|司法解释|解释|修正案)"
        if re.search(pattern, remaining):
            selected.append(aliases[name])
            remaining = re.sub(pattern, "", remaining)
    return list(dict.fromkeys(selected))


def resolve_filter(state, laws, enabled=True):
    conditions = [{"status": "现行有效"}]
    if not enabled:
        return conditions[0]
    original = state["question"]
    rewritten = state["retrieval_question"]
    selected = _named_laws(original, laws)
    text = original if selected else rewritten
    selected = selected or _named_laws(rewritten, laws)
    if selected:
        conditions.append({"law_id": {"$in": selected}})
        articles = {article_number(m.group(1)) for m in ARTICLE_REF.finditer(text)}
        if len(selected) == 1 and len(articles) == 1:
            conditions.append({"article": article_label(articles.pop())})
    return {"$and": conditions} if len(conditions) > 1 else conditions[0]
