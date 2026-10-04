"""Shared local PostgreSQL configuration, without changing process environment."""

import os
from pathlib import Path

from dotenv import dotenv_values


def read_settings():
    return {**dotenv_values(Path(__file__).parent / '.env'), **os.environ}


def postgres_settings(settings):
    return dict(
        host=settings.get('WEB_POSTGRES_HOST', '127.0.0.1'),
        port=int(settings.get('WEB_POSTGRES_PORT', '5433')),
        dbname=settings.get('WEB_POSTGRES_DATABASE', 'legal_rag'),
        user=settings.get('WEB_POSTGRES_USER', 'legal_rag'),
        password=Path(settings.get('WEB_POSTGRES_PASSWORD_FILE', '.postgres-password')).read_text(encoding='utf8').strip(),
        schema=settings.get('WEB_POSTGRES_SCHEMA', 'public'),
    )
