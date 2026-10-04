"""Frontend delivery contracts, using local build fixtures and no model clients."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app
from web_storage import SQLiteConversationStore


class FrontendDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.store = SQLiteConversationStore(self.directory / "test.sqlite3")
        self.build = self.directory / "frontend" / "dist"
        (self.build / "assets").mkdir(parents=True)
        (self.build / "index.html").write_text('<html>React build fixture</html>', encoding="utf8")
        (self.build / "assets" / "app-test.js").write_text('console.log("built asset");', encoding="utf8")
        (self.build / "private.txt").write_text("private build sibling", encoding="utf8")
        (self.directory / ".env").write_text("TEST_SECRET=do-not-serve", encoding="utf8")
        (self.directory / "frontend" / "src").mkdir()
        (self.directory / "frontend" / "src" / "main.tsx").write_text("private source", encoding="utf8")

    def client(self, **options):
        return TestClient(web_app.create_app(self.store, object(), **options))

    def test_explicit_build_is_root_with_no_store_and_legacy_session_cookie(self):
        client = self.client(frontend_directory=self.build)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, '<html>React build fixture</html>')
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertIn("legal_rag_session", client.cookies)

    def test_selected_build_assets_are_available(self):
        response = self.client(frontend_directory=self.build).get("/assets/app-test.js")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, 'console.log("built asset");')
        self.assertIn("javascript", response.headers["content-type"])

    def test_only_asset_directory_is_exposed_without_spa_fallback(self):
        client = self.client(frontend_directory=self.build)
        for path in ("/.env", "/src/main.tsx", "/frontend/src/main.tsx",
                     "/private.txt", "/assets/%2e%2e/private.txt", "/arbitrary-route",
                     "/assets/missing.js"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 404)

    def test_missing_or_incomplete_build_fails_at_construction_with_build_instruction(self):
        for component in ("directory", "index.html", "assets"):
            with self.subTest(component=component):
                build = self.directory / component.replace(".", "-")
                if component != "directory":
                    build.mkdir()
                    if component != "index.html":
                        (build / "index.html").write_text("page", encoding="utf8")
                    if component != "assets":
                        (build / "assets").mkdir()
                with self.assertRaisesRegex(RuntimeError, "npm run build"):
                    web_app.create_app(self.store, object(), frontend_directory=build)

    def test_default_create_app_still_serves_legacy(self):
        client = self.client()
        response = client.get("/")
        legacy = Path(web_app.__file__).parent / "web" / "static" / "index.html"
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, legacy.read_text(encoding="utf8"))
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertEqual(client.get("/assets/app-test.js").status_code, 404)

    def test_legacy_debug_static_files_remain_available_with_react_build(self):
        response = self.client(frontend_directory=self.build).get("/static/auth-client.mjs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("createAuthClient", response.text)

    def factory(self, settings):
        legacy = self.directory / "web" / "static"
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "index.html").write_text("legacy fixture", encoding="utf8")
        with patch.object(web_app, "__file__", str(self.directory / "web_app.py")), \
             patch.object(web_app, "read_settings", return_value=settings), \
             patch.object(web_app, "_create_store", return_value=object()), \
             patch.object(web_app, "create_web_rag_turn", side_effect=AssertionError("No model clients")):
            return web_app.create_default_app()

    def test_product_factory_selects_react_build_without_loading_model_clients(self):
        response = TestClient(self.factory({"WEB_FRONTEND": "react"})).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, '<html>React build fixture</html>')
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("set-cookie", response.headers)

    def test_product_factory_defaults_to_legacy_and_accepts_explicit_legacy(self):
        for settings in ({}, {"WEB_FRONTEND": "legacy"}):
            with self.subTest(settings=settings):
                self.assertEqual(TestClient(self.factory(settings)).get("/").text, "legacy fixture")

    def test_product_factory_rejects_missing_react_build(self):
        (self.build / "index.html").unlink()
        with self.assertRaisesRegex(RuntimeError, "npm run build"):
            self.factory({"WEB_FRONTEND": "react"})

    def test_product_factory_rejects_invalid_frontend_configuration(self):
        with self.assertRaisesRegex(ValueError, "WEB_FRONTEND.*legacy.*react"):
            self.factory({"WEB_FRONTEND": "unknown"})


if __name__ == "__main__":
    unittest.main()
