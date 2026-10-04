"""Versioned PDF ingestion and explicit metadata scope for the legal corpus."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from corpus_metadata import knowledge_base_id

from guide_corpus import prepare_guide
from rag_pipeline_articles import split_by_articles


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "laws.json"
GUIDES_FILENAME = "guides.json"
DB_DIR = ROOT / "chroma_legal_db"
SCHEMA_VERSION = "v12-knowledge-bases-1"
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


def load_guides(catalog_path=CATALOG_PATH):
    catalog_path = Path(catalog_path)
    path = catalog_path.parent / GUIDES_FILENAME
    if not path.exists():
        return []
    guides = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(guides, list):
        raise ValueError("办事指南目录必须是列表")
    ids, files = set(), set()
    for guide in guides:
        knowledge_base_id(guide)
        for key in ("document_id", "document_type", "title", "source_file", "version", "status"):
            if not isinstance(guide.get(key), str) or not guide[key].strip():
                raise ValueError(f"办事指南目录缺少 {key}")
        if not isinstance(guide.get("content_start_page"), int) or guide["content_start_page"] < 1:
            raise ValueError("办事指南必须登记正文起始页")
        if not isinstance(guide.get("routing_topics"), list) or not guide["routing_topics"] or not all(
            isinstance(topic, str) and topic.strip() for topic in guide["routing_topics"]
        ):
            raise ValueError("办事指南必须登记 routing_topics")
        source = (catalog_path.parent / guide["source_file"]).resolve()
        if source.parent != catalog_path.parent.resolve() or source.suffix.lower() != ".pdf":
            raise ValueError("PDF 必须位于知识库目录所在文件夹")
        if not source.is_file():
            raise ValueError(f"缺少 PDF：{source.name}")
        if guide["document_id"] in ids or guide["source_file"] in files:
            raise ValueError("办事指南 ID 或文件重复登记")
        ids.add(guide["document_id"])
        files.add(guide["source_file"])
    return guides


def load_catalog(catalog_path=CATALOG_PATH):
    catalog_path = Path(catalog_path)
    laws = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(laws, list) or not laws:
        raise ValueError("法律目录必须是非空列表")
    ids, files = set(), set()
    for law in laws:
        knowledge_base_id(law)
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
    guide_files = {guide["source_file"] for guide in load_guides(catalog_path)}
    if files & guide_files:
        raise ValueError("法律和办事指南不能登记同一个 PDF")
    extras = {p.name for p in catalog_path.parent.iterdir() if p.suffix.lower() == ".pdf"} - files - guide_files
    if extras:
        raise ValueError(f"请先在 laws.json 登记新增 PDF：{sorted(extras)}")
    return laws


def prepare_law(pages, law):
    knowledge_base_id(law)
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
            "law_id", "law_name", "source_file", "source_url", "version", "effective_date", "status", "knowledge_base_id",
        )})
        doc.metadata["index_status"] = "active"
        doc.metadata["pages"] = json.dumps(doc.metadata["pages"])
        doc.metadata["source"] = law["source_file"]
    return chunks


def corpus_fingerprint(laws, embedding_config, catalog_path=CATALOG_PATH, guides=()):
    entries = [*laws, *guides]
    files = {entry["source_file"]: hashlib.sha256(
        (Path(catalog_path).parent / entry["source_file"]).read_bytes()
    ).hexdigest() for entry in entries}
    inputs = {"schema": SCHEMA_VERSION, "embedding": embedding_config, "laws": laws, "guides": guides, "files": files}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return digest, files


def _source_inventory(laws, guides, files):
    inventory = []
    for law in laws:
        inventory.append({
            "source_id": law["law_id"],
            "knowledge_base_id": law["knowledge_base_id"],
            "source_type": "law",
            "source_file": law["source_file"],
            "version": law["version"],
            "sha256": files[law["source_file"]],
        })
    for guide in guides:
        inventory.append({
            "source_id": guide["document_id"],
            "knowledge_base_id": guide["knowledge_base_id"],
            "source_type": "guide",
            "source_file": guide["source_file"],
            "version": guide["version"],
            "sha256": files[guide["source_file"]],
        })
    return inventory


def _status_source(source):
    return {key: source[key] for key in (
        "source_id", "source_type", "source_file", "version"
    )}


def inspect_corpus_status(embedding_config, catalog_path=CATALOG_PATH, db_dir=DB_DIR):
    """Compare current corpus inputs with the last published manifest without writing data."""
    laws = load_catalog(catalog_path)
    guides = load_guides(catalog_path)
    fingerprint, files = corpus_fingerprint(laws, embedding_config, catalog_path, guides)
    current_sources = _source_inventory(laws, guides, files)
    manifest_path = Path(db_dir) / "manifest.json"
    if not manifest_path.exists():
        return {
            "state": "not_indexed",
            "needs_ingest": True,
            "changes": {
                "added": [_status_source(source) for source in current_sources],
                "modified": [],
                "deleted": [],
                "configuration_changed": False,
            },
        }

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_files = saved.get("files", {})
    current_by_file = {source["source_file"]: source for source in current_sources}
    added = [_status_source(source) for filename, source in current_by_file.items() if filename not in saved_files]
    modified = [
        _status_source(source) for filename, source in current_by_file.items()
        if filename in saved_files and saved_files[filename] != source["sha256"]
    ]
    deleted = [{"source_file": filename} for filename in saved_files if filename not in current_by_file]
    configuration_changed = (
        saved.get("fingerprint") != fingerprint
        and not (added or modified or deleted)
    )
    needs_ingest = saved.get("fingerprint") != fingerprint
    return {
        "state": "needs_ingest" if needs_ingest else "ready",
        "needs_ingest": needs_ingest,
        "changes": {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "configuration_changed": configuration_changed,
        },
    }


def inspect_index_collections(db_dir=DB_DIR):
    """List the active, stale project-managed, and unmanaged Chroma collections."""
    db_dir = Path(db_dir)
    manifest_path = db_dir / "manifest.json"
    if not manifest_path.exists():
        return {"active": None, "stale": [], "unmanaged": []}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active = manifest["collection"]
    client = chromadb.PersistentClient(path=str(db_dir))
    names = sorted(collection.name for collection in client.list_collections())
    stale = [name for name in names if name.startswith("legal_") and name != active]
    unmanaged = [name for name in names if name != active and name not in stale]
    return {"active": active, "stale": stale, "unmanaged": unmanaged}


def prune_stale_legal_collections(db_dir=DB_DIR):
    """Delete only stale project-managed collections after a manifest has selected one active index."""
    status = inspect_index_collections(db_dir)
    if status["active"] is None:
        raise ValueError("尚未发布知识库，无法判断哪些索引可安全清理")
    client = chromadb.PersistentClient(path=str(Path(db_dir)))
    for name in status["stale"]:
        client.delete_collection(name)
    return status["stale"]


def _store(embeddings, db_dir, collection):
    return Chroma(collection_name=collection, persist_directory=str(db_dir), embedding_function=embeddings)


def _validate_index_labels(store, laws, guides):
    expected = {}
    for entry in [*laws, *guides]:
        field = 'law_id' if 'law_id' in entry else 'document_id'
        expected[(field, entry[field], entry['version'])] = entry
    stored = store.get(include=['metadatas'])
    for identifier, metadata in zip(stored['ids'], stored['metadatas']):
        metadata = metadata or {}
        field = 'law_id' if metadata.get('law_id') else 'document_id'
        source_id, version = metadata.get(field), metadata.get('version')
        entry = expected.get((field, source_id, version))
        if (entry is None or not identifier.startswith(f'{source_id}:{version}:')
                or metadata.get('source_file') != entry['source_file']
                or metadata.get('knowledge_base_id') != entry['knowledge_base_id']):
            raise ValueError('索引 knowledge_base_id 与资料登记不一致，未发布；请核对索引并重新导入')


def ingest_corpus(embeddings, embedding_config, catalog_path=CATALOG_PATH, db_dir=DB_DIR):
    laws = load_catalog(catalog_path)
    guides = load_guides(catalog_path)
    fingerprint, files = corpus_fingerprint(laws, embedding_config, catalog_path, guides)
    db_dir = Path(db_dir)
    manifest_path = db_dir / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] == fingerprint:
            store = _store(embeddings, db_dir, saved["collection"])
            if store._collection.count() == saved.get("chunk_count", saved["article_count"]):
                _validate_index_labels(store, laws, guides)
                if "sources" not in saved:
                    saved["sources"] = _source_inventory(laws, guides, files)
                    temporary = db_dir / "manifest.pending.json"
                    temporary.write_text(
                        json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(manifest_path)
                return saved
    chunks = []
    counts = {}
    guide_counts = {}
    for law in laws:
        pages = PyPDFLoader(str(Path(catalog_path).parent / law["source_file"])).load()
        articles = prepare_law(pages, law)
        chunks.extend(articles)
        counts[law["law_id"]] = len(articles)
    for guide in guides:
        pages = PyPDFLoader(str(Path(catalog_path).parent / guide["source_file"])).load()
        guide_chunks = prepare_guide(pages, guide)
        for chunk in guide_chunks:
            chunk.metadata["index_status"] = "active"
        chunks.extend(guide_chunks)
        guide_counts[guide["document_id"]] = len(guide_chunks)
    collection = f"legal_{fingerprint[:24]}"
    store = _store(embeddings, db_dir, collection)
    ids = [
        f"{d.metadata['law_id']}:{d.metadata['version']}:{d.metadata['article']}"
        if d.metadata.get("law_id")
        else f"{d.metadata['document_id']}:{d.metadata['version']}:{d.metadata['section']}"
        for d in chunks
    ]
    existing = set(store.get(include=[])["ids"])
    pending = [(doc, doc_id) for doc, doc_id in zip(chunks, ids) if doc_id not in existing]
    for offset in range(0, len(pending), 32):
        batch = pending[offset:offset + 32]
        store.add_documents([item[0] for item in batch], ids=[item[1] for item in batch])
        print(f"导入法条：{min(offset + 32, len(pending))}/{len(pending)}", flush=True)
    if set(store.get(include=[])["ids"]) != set(ids):
        raise RuntimeError("索引数量或法条 ID 校验失败，未发布新索引")
    _validate_index_labels(store, laws, guides)
    manifest = {
        "schema": SCHEMA_VERSION, "fingerprint": fingerprint, "collection": collection,
        "embedding": embedding_config, "files": files, "law_counts": counts,
        "sources": _source_inventory(laws, guides, files),
        "article_count": sum(counts.values()), "guide_counts": guide_counts,
        "chunk_count": len(chunks), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = db_dir / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def open_corpus(embeddings, embedding_config, catalog_path=CATALOG_PATH, db_dir=DB_DIR):
    laws = load_catalog(catalog_path)
    guides = load_guides(catalog_path)
    fingerprint, _ = corpus_fingerprint(laws, embedding_config, catalog_path, guides)
    path = Path(db_dir) / "manifest.json"
    if not path.exists():
        raise ValueError("尚未导入知识库，请运行 python main.py --ingest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["fingerprint"] != fingerprint:
        raise ValueError("PDF、目录或 Embedding 配置已变化，请重新导入：python main.py --ingest")
    store = _store(embeddings, db_dir, manifest["collection"])
    if store._collection.count() != manifest.get("chunk_count", manifest["article_count"]):
        raise ValueError("知识库不完整，请重新导入：python main.py --ingest")
    _validate_index_labels(store, laws, guides)
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
    conditions = [{"index_status": "active"}]
    if not enabled:
        return conditions[0]
    original = state["question"]
    rewritten = state["retrieval_question"]
    original_selected = _named_laws(original, laws)
    selected = original_selected
    text = original if selected else rewritten
    selected = selected or _named_laws(rewritten, laws)
    if "include_guide" in state and state["include_guide"] is None and selected:
        return conditions[0]
    if selected:
        articles = {article_number(m.group(1)) for m in ARTICLE_REF.finditer(text)}
        law_conditions = [{"law_id": {"$in": selected}}]
        if len(selected) == 1 and len(articles) == 1:
            law_conditions.append({"article": article_label(articles.pop())})
        law_scope = {"$and": law_conditions} if len(law_conditions) > 1 else law_conditions[0]
        include_guide = state.get("include_guide") is True or (
            original_selected and "办事指南" in original
        )
        if include_guide:
            conditions.append({"$or": [law_scope, {"document_type": "办事指南"}]})
        else:
            conditions.extend(law_conditions)
    return {"$and": conditions} if len(conditions) > 1 else conditions[0]
