import unittest
from unittest.mock import patch


class CreateRagChainTests(unittest.TestCase):
    @patch("rag_app.build_rag_chain")
    @patch("rag_app.Chroma")
    @patch("rag_app.OpenAIEmbeddings")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.load_dotenv")
    def test_creates_shared_chain(
        self,
        load_dotenv,
        chat_openai,
        openai_embeddings,
        chroma,
        build_chain,
    ):
        chroma.return_value.as_retriever.return_value = "retriever"
        build_chain.return_value = "shared-chain"

        import rag_app

        self.assertEqual(rag_app.create_rag_chain(), "shared-chain")
        build_chain.assert_called_once()


if __name__ == "__main__":
    unittest.main()
