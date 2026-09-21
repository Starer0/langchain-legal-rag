import unittest
from unittest.mock import patch


class CreateChromaOnlyChainTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("rag_app_chroma_only.build_rag_chain")
    @patch("rag_app_chroma_only.open_corpus")
    @patch("rag_app_chroma_only._embedding_config")
    @patch("rag_app_chroma_only._create_embeddings")
    @patch("rag_app_chroma_only.ChatOpenAI")
    @patch("rag_app_chroma_only.load_dotenv")
    def test_passes_chroma_candidates_directly_to_the_answer_chain(
        self,
        load_dotenv,
        chat_openai,
        create_embeddings,
        embedding_config,
        open_corpus,
        build_chain,
        print_output,
    ):
        open_corpus.return_value = (
            open_corpus.return_value[0],
            {"article_count": 0},
            [],
        )
        build_chain.return_value = "shared-chain"

        from rag_app_chroma_only import create_chroma_only_chain

        self.assertEqual(create_chroma_only_chain(), "shared-chain")
        direct_documents = build_chain.call_args.args[1]
        self.assertEqual(
            direct_documents.invoke({"candidates": ["candidate-document"]}),
            ["candidate-document"],
        )


if __name__ == "__main__":
    unittest.main()
