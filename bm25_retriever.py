"""Local BM25 keyword retrieval for the published legal corpus."""

import math
import re
from collections import Counter

from langchain_core.documents import Document


_CJK_TEXT = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Use CJK bigrams and lower-case Latin words as small, dependency-free tokens."""
    lowered = text.lower()
    tokens = []
    for segment in _CJK_TEXT.findall(lowered):
        tokens.extend(segment[index:index + 2] for index in range(len(segment) - 1))
        if len(segment) == 1:
            tokens.append(segment)
    tokens.extend(_LATIN_WORD.findall(lowered))
    return tokens


def _matches_filter(metadata: dict, expression: dict | None) -> bool:
    if not expression:
        return True
    if "$and" in expression:
        return all(_matches_filter(metadata, item) for item in expression["$and"])
    if "$or" in expression:
        return any(_matches_filter(metadata, item) for item in expression["$or"])
    for field, expected in expression.items():
        actual = metadata.get(field)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


def _document_key(document: Document):
    metadata = document.metadata
    if metadata.get("law_id") and metadata.get("article"):
        return ("law", metadata["law_id"], metadata.get("version"), metadata["article"])
    if metadata.get("document_id") and metadata.get("section"):
        return ("guide", metadata["document_id"], metadata.get("version"), metadata["section"])
    return ("content", metadata.get("source"), metadata.get("article"), document.page_content)


def merge_retrieval_candidates(
    vector_documents: list[Document], bm25_documents: list[Document],
) -> list[Document]:
    """Keep vector candidates first, then append BM25-only documents once."""
    merged = []
    seen = set()
    for document in [*vector_documents, *bm25_documents]:
        key = _document_key(document)
        if key not in seen:
            seen.add(key)
            merged.append(document)
    return merged


class BM25Retriever:
    """A small in-memory BM25 retriever built from already indexed documents."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(_tokenize(document.page_content)) for document in documents]
        self.lengths = [sum(terms.values()) for terms in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        document_frequency = Counter()
        for terms in self.term_frequencies:
            document_frequency.update(terms.keys())
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_vectorstore(cls, vectorstore):
        stored = vectorstore.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]
        return cls(documents)

    def search(self, query: str, k: int, metadata_filter: dict | None = None) -> list[Document]:
        if k < 1:
            raise ValueError("k 必须大于等于 1")
        query_terms = set(_tokenize(query))
        scored = []
        for document, frequencies, length in zip(
            self.documents, self.term_frequencies, self.lengths
        ):
            if not _matches_filter(document.metadata, metadata_filter):
                continue
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                normalization = self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                score += self.idf[term] * frequency * (self.k1 + 1) / (frequency + normalization)
            if score > 0:
                scored.append((score, document))
        return [document for _, document in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]
