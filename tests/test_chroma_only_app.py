import unittest
from unittest.mock import patch


class CreateChromaOnlyChainTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("rag_app_chroma_only.build_rag_chain")
    @patch("rag_app_chroma_only.Chroma")
    @patch("rag_app_chroma_only.OpenAIEmbeddings")
    @patch("rag_app_chroma_only.ChatOpenAI")
    @patch("rag_app_chroma_only.PyPDFLoader")
    @patch("rag_app_chroma_only.load_dotenv")
    def test_passes_chroma_candidates_directly_to_the_answer_chain(
        self,
        load_dotenv,
        pdf_loader,
        chat_openai,
        openai_embeddings,
        chroma,
        build_chain,
        print_output,
    ):
        chroma.return_value.as_retriever.return_value = "retriever"
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
