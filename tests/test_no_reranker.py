import unittest
from unittest.mock import patch


class NoRerankerTests(unittest.TestCase):
    @patch.dict("os.environ", {"USE_RERANKER": "false"}, clear=False)
    @patch("builtins.print")
    @patch("rag_app.build_rag_chain")
    @patch("rag_app.SiliconFlowReranker")
    @patch("rag_app.open_corpus")
    @patch("rag_app.OpenAIEmbeddings")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.load_dotenv")
    def test_disables_reranker_when_the_flag_is_false(
        self,
        load_dotenv,
        chat_openai,
        openai_embeddings,
        open_corpus,
        reranker_client,
        build_chain,
        print_output,
    ):
        open_corpus.return_value = (
            open_corpus.return_value[0],
            {"article_count": 0},
            [],
        )
        build_chain.return_value = "shared-chain"

        import rag_app

        self.assertEqual(rag_app.create_rag_chain(), "shared-chain")
        reranker_client.assert_not_called()
        bypass_reranker = build_chain.call_args.args[1]
        self.assertEqual(
            bypass_reranker.invoke({"candidates": ["candidate-document"]}),
            ["candidate-document"],
        )


if __name__ == "__main__":
    unittest.main()
