import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import legal_corpus as corpus


LAW = {
    "law_id": "labor_contract_law", "law_name": "中华人民共和国劳动合同法",
    "aliases": ["劳动合同法"], "source_file": "law.pdf", "version": "2012-12-28",
    "effective_date": "2013-07-01", "status": "现行有效", "expected_articles": 2,
    "source_url": "https://example.org/law",
}
PAGES = [
    Document(page_content="1\n中华人民共和国劳动合同法\n第一章 总则\n第一条 第一段。\n第二条 跨页开头", metadata={"page": 0}),
    Document(page_content="2\n中华人民共和国劳动合同法\n跨页结尾。", metadata={"page": 1}),
    Document(page_content="3\n文档信息\n用途说明：不是法条。", metadata={"page": 2}),
]


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[1.0, float(len(t) % 13), 0.5] for t in texts]

    def embed_query(self, text):
        return [1.0, float(len(text) % 13), 0.5]


class CorpusParsingTests(unittest.TestCase):
    def test_preserves_cross_page_article_and_excludes_editorial_appendix(self):
        docs = corpus.prepare_law(PAGES, LAW)
        self.assertEqual([d.metadata["article"] for d in docs], ["第一条", "第二条"])
        self.assertEqual(docs[1].page_content, "第二条 跨页开头\n跨页结尾。")
        self.assertEqual(json.loads(docs[1].metadata["pages"]), [1, 2])
        self.assertEqual(docs[1].metadata["chapter"], "第一章 总则")
        self.assertEqual(docs[1].metadata["law_id"], "labor_contract_law")
        self.assertNotIn("文档信息", docs[-1].page_content)

    def test_chapter_boundary_is_not_part_of_previous_article(self):
        pages = [Document(page_content="第一章 总则\n第一条 甲。\n第二章 工资\n第二条 乙。", metadata={"page": 0})]
        docs = corpus.prepare_law(pages, LAW)
        self.assertEqual(docs[0].page_content, "第一条 甲。")
        self.assertEqual(docs[1].metadata["chapter"], "第二章 工资")

    def test_missing_duplicate_and_empty_articles_fail_validation(self):
        for content in ["第一条 甲。", "第一条 甲。\n第一条 乙。", "第一条\n第二条 乙。"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                corpus.prepare_law([Document(page_content=content, metadata={"page": 0})], LAW)


class CorpusIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "law.pdf").write_bytes(b"test-pdf-fingerprint")
        self.catalog = self.root / "data" / "laws.json"
        self.catalog.write_text(json.dumps([LAW], ensure_ascii=False), encoding="utf-8")
        self.db = self.root / "index"
        self.embeddings = LocalEmbeddings()

    def tearDown(self):
        # Chroma owns persistent handles on Windows; release only the temporary files it allows.
        self.temp._ignore_cleanup_errors = True
        self.temp.cleanup()

    def ingest(self):
        return corpus.ingest_corpus(self.embeddings, "test-model", self.catalog, self.db)

    def open(self):
        return corpus.open_corpus(self.embeddings, "test-model", self.catalog, self.db)

    @patch("legal_corpus.PyPDFLoader")
    def test_reimport_skips_embeddings_and_online_open_does_not_parse_pdf(self, loader):
        loader.return_value.load.return_value = PAGES
        first = self.ingest()
        calls = self.embeddings.document_calls
        second = self.ingest()
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(self.embeddings.document_calls, calls)
        loader.reset_mock()
        store, manifest, laws = self.open()
        self.assertEqual(store._collection.count(), 2)
        self.assertEqual(laws[0]["law_id"], "labor_contract_law")
        self.assertEqual(manifest["article_count"], 2)
        loader.assert_not_called()

    @patch("legal_corpus.PyPDFLoader")
    def test_failed_new_import_does_not_publish_over_previous_manifest(self, loader):
        loader.return_value.load.return_value = PAGES
        first = self.ingest()
        (self.root / "data" / "law.pdf").write_bytes(b"changed-pdf")
        with self.assertRaisesRegex(ValueError, "导入"):
            self.open()
        loader.return_value.load.return_value = PAGES
        with patch.object(self.embeddings, "embed_documents", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                self.ingest()
        saved = json.loads((self.db / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["fingerprint"], first["fingerprint"])

    def test_undeclared_pdf_and_path_escape_are_rejected(self):
        (self.root / "data" / "extra.pdf").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "登记"):
            corpus.load_catalog(self.catalog)
        self.catalog.write_text(json.dumps([{**LAW, "source_file": "../outside.pdf"}]), encoding="utf-8")
        with self.assertRaises(ValueError):
            corpus.load_catalog(self.catalog)


class MetadataScopeTests(unittest.TestCase):
    def setUp(self):
        self.laws = [LAW, {**LAW, "law_id": "labor_law", "law_name": "中华人民共和国劳动法", "aliases": ["劳动法"]}]

    def resolve(self, question, rewritten=None):
        return corpus.resolve_filter({"question": question, "retrieval_question": rewritten or question}, self.laws)

    def test_general_question_searches_all_effective_laws(self):
        self.assertEqual(self.resolve("试用期工资有什么规定？"), {"status": "现行有效"})

    def test_original_law_and_arabic_article_survive_wrong_rewrite(self):
        self.assertEqual(self.resolve("劳动合同法第20条是什么？", "劳动法第四十四条"), {
            "$and": [{"status": "现行有效"}, {"law_id": {"$in": ["labor_contract_law"]}}, {"article": "第二十条"}],
        })

    def test_comparison_keeps_both_laws_and_bare_article_does_not_choose_law(self):
        scope = self.resolve("比较《劳动法》和《劳动合同法》的第二十条")
        self.assertEqual(set(scope["$and"][1]["law_id"]["$in"]), {"labor_law", "labor_contract_law"})
        self.assertEqual(len(scope["$and"]), 2)
        self.assertEqual(self.resolve("第二十条是什么？"), {"status": "现行有效"})

    def test_unknown_book_title_does_not_match_known_law_prefix(self):
        scope = self.resolve("《劳动合同法实施条例》第十条怎么规定？")
        self.assertEqual(scope["$and"][1], {"law_id": {"$in": ["__unknown_law__"]}})

    def test_history_rewrite_can_restore_law_scope(self):
        scope = self.resolve("那第二条呢？", "劳动合同法第二条怎么规定？")
        self.assertEqual(scope["$and"][-1], {"article": "第二条"})


if __name__ == "__main__":
    unittest.main()
