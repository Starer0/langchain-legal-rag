"""Pure ASGI authentication guard, Cookie renewal and CSRF checks."""

import hmac
import json
from collections import deque
from functools import partial
from threading import Lock
from time import monotonic
from urllib.parse import urlsplit

from anyio import to_thread
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


AUTH_COOKIE = 'legal_rag_auth'


def _origin_label(value):
    """Log only an origin, excluding embedded credentials, paths and queries."""
    if value is None or value == 'null':
        return value
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ('http', 'https', 'chrome-extension', 'moz-extension')
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment):
            return 'invalid'
        host = parsed.hostname
        if ':' in host:
            host = f'[{host}]'
        port = f':{parsed.port}' if parsed.port is not None else ''
        return f'{parsed.scheme}://{host}{port}'
    except ValueError:
        return 'invalid'


def cookie_headers(session=None, *, secure=False):
    response = Response()
    if session is None:
        response.delete_cookie(AUTH_COOKIE, httponly=True, secure=secure, samesite='lax')
    else:
        response.set_cookie(AUTH_COOKIE, session.token, max_age=259200, expires=session.expires_at,
                            httponly=True, secure=secure, samesite='lax')
    return [(key, value) for key, value in response.raw_headers if key == b'set-cookie']


def session_data(session):
    return {'user': {'id': session.user.id, 'username': session.user.username, 'level_id': session.user.level_id},
            'csrf_token': session.csrf_token, 'expires_at': session.expires_at.isoformat()}


class LoginLimiter:
    def __init__(self):
        self.attempts = {}
        self.lock = Lock()

    def allow(self, address):
        now = monotonic()
        with self.lock:
            self.attempts = {key: attempts for key, attempts in self.attempts.items() if attempts and attempts[-1] > now - 300}
            if address not in self.attempts and len(self.attempts) >= 1024:
                return False
            attempts = self.attempts.setdefault(address, deque())
            while attempts and attempts[0] <= now - 300:
                attempts.popleft()
            if len(attempts) >= 20:
                return False
            attempts.append(now)
            return True


class AuthenticationMiddleware:
    def __init__(self, app, authentication, secure_cookies=False):
        self.app = app
        self.authentication = authentication
        self.secure = secure_cookies
        self.limiter = LoginLimiter()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request = Request(scope)
        path, method = scope['path'], scope['method']
        protected = path.startswith('/api/conversations') or path == '/api/memory' or path.startswith('/api/memory/') or path in ('/api/auth/me', '/api/auth/logout')
        auth_api = path.startswith('/api/auth/')
        if not protected and not auth_api:
            return await self.app(scope, receive, send)

        async def reject(status, detail, *, clear=False):
            response = JSONResponse({'detail': detail}, status_code=status, headers={'Cache-Control': 'no-store'})
            if clear:
                response.raw_headers.extend(cookie_headers(secure=self.secure))
            await response(scope, receive, send)

        if method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            expected_origin = f'{request.url.scheme}://{request.url.netloc}'
            if request.headers.get('sec-fetch-site') == 'cross-site' or (origin is not None and origin != expected_origin):
                fetch_site = request.headers.get('sec-fetch-site')
                scope.setdefault('state', {})['request_error'] = json.dumps({
                    'reason': 'untrusted_request_origin',
                    'origin': _origin_label(origin),
                    'expected_origin': _origin_label(expected_origin),
                    'fetch_site': fetch_site if fetch_site in ('same-origin', 'same-site', 'cross-site', 'none', None) else 'invalid',
                })
                return await reject(403, '请求来源不受信任')
        if path == '/api/auth/login' and not self.limiter.allow(request.client.host if request.client else 'unknown'):
            return await reject(429, '登录尝试过多，请五分钟后重试')
        token = request.cookies.get(AUTH_COOKIE)
        session = None
        if protected:
            try:
                session = await to_thread.run_sync(partial(self.authentication.lookup, token))
            except Exception as error:
                scope.setdefault('state', {})['request_error'] = f'Authentication backend error: {type(error).__name__}'
                return await reject(503, '登录服务暂时不可用，请稍后重试')
            if session is None:
                return await reject(401, '请先登录', clear=True)
            if method not in ('GET', 'HEAD', 'OPTIONS') and not hmac.compare_digest(session.csrf_token.encode('utf8'), request.headers.get('x-csrf-token', '').encode('utf8')):
                return await reject(403, '请求校验失败，请刷新页面后重试')
            scope.setdefault('state', {})['login_session'] = session

        async def authenticated_send(message):
            if message['type'] == 'http.response.start':
                headers = [(key, value) for key, value in message.get('headers', []) if key.lower() != b'cache-control']
                headers.append((b'cache-control', b'no-store'))
                if session is not None and 200 <= message['status'] < 300 and path not in ('/api/auth/logout', '/api/auth/me'):
                    try:
                        renewed = await to_thread.run_sync(partial(self.authentication.lookup, token, renew=True))
                    except Exception as error:
                        scope.setdefault('state', {})['request_error'] = f'Session renewal error: {type(error).__name__}'
                        renewed = None
                    if renewed is not None:
                        headers.extend(cookie_headers(renewed, secure=self.secure))
                message = {**message, 'headers': headers}
            await send(message)

        await self.app(scope, receive, authenticated_send)
