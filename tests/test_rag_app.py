import unittest
from unittest.mock import patch


class CreateRagChainTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("rag_app.build_rag_chain")
    @patch("rag_app.Chroma")
    @patch("rag_app.OpenAIEmbeddings")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.PyPDFLoader")
    @patch("rag_app.load_dotenv")
    def test_creates_shared_chain(
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

        import rag_app

        self.assertEqual(rag_app.create_rag_chain(), "shared-chain")
        build_chain.assert_called_once()


    @patch.dict("os.environ", {"HISTORY_TURNS": "3"}, clear=False)
    @patch("rag_app.RetrievalQuestionRewriter")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.create_rag_chain")
    def test_creates_conversation_service_with_configured_turn_limit(
        self,
        create_rag_chain,
        chat_openai,
        rewriter,
    ):
        import rag_app

        create_rag_chain.return_value = "shared-chain"

        service = rag_app.create_conversation_service()

        self.assertEqual(service.max_turns, 3)
        self.assertEqual(service.rag_chain, "shared-chain")
        create_rag_chain.assert_called_once()
        rewriter.assert_called_once_with(chat_openai.return_value)


    @patch("builtins.print")
    @patch("rag_app.build_rag_chain")
    @patch("rag_app.SiliconFlowReranker")
    @patch("rag_app.Chroma")
    @patch("rag_app.OpenAIEmbeddings")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.PyPDFLoader")
    @patch("rag_app.load_dotenv")
    def test_reranker_uses_retrieval_question_and_defaults_to_top_four(
        self,
        load_dotenv,
        pdf_loader,
        chat_openai,
        openai_embeddings,
        chroma,
        reranker_client,
        build_chain,
        print_output,
    ):
        chroma.return_value.as_retriever.return_value = "retriever"
        build_chain.return_value = "shared-chain"

        import rag_app

        rag_app.create_rag_chain()

        self.assertEqual(reranker_client.call_args.kwargs["top_n"], 4)
        reranker = build_chain.call_args.args[1]
        reranker.invoke({
            "question": "那工资呢？",
            "retrieval_question": "试用期内劳动者工资有什么规定？",
            "candidates": ["candidate-document"],
        })
        reranker_client.return_value.rerank.assert_called_once_with(
            "试用期内劳动者工资有什么规定？",
            ["candidate-document"],
        )


if __name__ == "__main__":
    unittest.main()
