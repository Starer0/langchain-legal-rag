import json
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from request_logging import RequestLoggingMiddleware
from web_auth import AuthenticationMiddleware


class CaptureLogger:
    def __init__(self):
        self.events = []

    def info(self, message):
        self.events.append(json.loads(message))


class OriginDiagnosticsTests(unittest.TestCase):
    def test_rejection_explains_origin_and_fetch_metadata_without_credentials(self):
        logger = CaptureLogger()
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware, authentication=object())
        app.add_middleware(RequestLoggingMiddleware, logger=logger)
        client = TestClient(app, base_url='http://127.0.0.1:8001')
        result = client.post('/api/auth/login',
                             headers={'Origin': 'http://127.0.0.1:8001', 'Sec-Fetch-Site': 'cross-site'},
                             json={'username': 'demo_ab', 'password': 'secret-not-for-logs'})
        self.assertEqual(result.status_code, 403)
        detail = json.loads(logger.events[-1]['error'])
        self.assertEqual(detail['reason'], 'untrusted_request_origin')
        self.assertEqual(detail['origin'], 'http://127.0.0.1:8001')
        self.assertEqual(detail['expected_origin'], 'http://127.0.0.1:8001')
        self.assertEqual(detail['fetch_site'], 'cross-site')
        self.assertNotIn('secret-not-for-logs', json.dumps(logger.events))

    def test_malformed_origin_does_not_write_embedded_credentials_or_paths(self):
        logger = CaptureLogger()
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware, authentication=object())
        app.add_middleware(RequestLoggingMiddleware, logger=logger)
        client = TestClient(app)
        result = client.post('/api/auth/login', headers={
            'Origin': 'http://user:embedded-password@example.com/private-token',
            'Sec-Fetch-Site': 'untrusted-token-value',
        })
        self.assertEqual(result.status_code, 403)
        self.assertIn('error', logger.events[-1])
        serialized = json.dumps(logger.events)
        for secret in ('embedded-password', 'private-token', 'untrusted-token-value'):
            self.assertNotIn(secret, serialized)


if __name__ == '__main__':
    unittest.main()
