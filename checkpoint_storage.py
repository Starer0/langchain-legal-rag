"""Official PostgreSQL checkpointer resources and single-server ownership."""
import hashlib
import re
from threading import Lock
from contextlib import contextmanager
from contextvars import ContextVar
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def graph_schema(schema):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,55}', schema): raise ValueError('Invalid checkpoint schema')
    return schema + '_graph'


def serializer():
    return JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=[
        ('langchain_core.documents.base', 'Document'),
        ('langchain_core.messages.human', 'HumanMessage'),
        ('langchain_core.messages.ai', 'AIMessage')])


_claim = ContextVar('checkpoint_execution_claim', default=None)


class FencedPostgresSaver(PostgresSaver):
    """Keep official serialization/SQL, guard writes in the same transaction."""
    def __init__(self, conn, lease):
        super().__init__(conn, serde=serializer())
        self.lease = lease

    @contextmanager
    def _cursor(self, *, pipeline=False):
        with self.lock, self.conn.connection() as connection:
            with connection.transaction(), connection.cursor(binary=True, row_factory=dict_row) as c:
                claim = _claim.get()
                if claim is not None:
                    self.lease.check()
                    tid, epoch = claim
                    # A shared task row lock blocks a new epoch claim until this write commits.
                    c.execute(f'SELECT id FROM "{self.lease.schema}".generation_tasks WHERE id=%s FOR SHARE', (tid,))
                    exists = c.fetchone()
                    if exists is not None:
                        c.execute(f'SELECT epoch FROM "{self.lease.schema}".generation_recovery WHERE task_id=%s', (tid,))
                        row = c.fetchone()
                        if row is None or row['epoch'] != epoch: raise RuntimeError('Stale checkpoint execution')
                    elif epoch is not None:
                        raise RuntimeError('Stale checkpoint task deleted')
                yield c

    def put(self, config, checkpoint, metadata, new_versions):
        token = _claim.set((config['configurable']['thread_id'], config['configurable'].get('execution_epoch')))
        try: return super().put(config, checkpoint, metadata, new_versions)
        finally: _claim.reset(token)

    def put_writes(self, config, writes, task_id, task_path=''):
        token = _claim.set((config['configurable']['thread_id'], config['configurable'].get('execution_epoch')))
        try: return super().put_writes(config, writes, task_id, task_path)
        finally: _claim.reset(token)


class CheckpointStorage:
    def __init__(self, settings, *, schema='public'):
        self.settings = dict(settings)
        self.settings.pop('schema', None)
        self.settings.setdefault('connect_timeout', 5)
        self.schema, self.graph_schema = schema, graph_schema(schema)
        self.key = int.from_bytes(hashlib.sha256(('legal-rag-server:' + schema).encode()).digest()[:8], 'big', signed=True)
        self.connection = self.pool = self.saver = None
        self.guard = Lock()

    def __enter__(self):
        from task_recovery import MIGRATION
        try:
            self.connection = psycopg.connect(**self.settings, autocommit=True)
            if not self.connection.execute('SELECT pg_try_advisory_lock(%s)', (self.key,)).fetchone()[0]:
                raise RuntimeError('Generation server already running')
            if not self.connection.execute(f'SELECT 1 FROM "{self.schema}".schema_migrations WHERE name=%s', (MIGRATION,)).fetchone():
                raise RuntimeError('Run prepare_web_recovery.py --apply first')
            self.pool = ConnectionPool(kwargs={**self.settings, 'autocommit': True, 'row_factory': dict_row,
                'connect_timeout': 5, 'options': '-c search_path=' + self.graph_schema}, min_size=1, max_size=5, open=True)
            self.pool.wait(timeout=10)
            self.saver = FencedPostgresSaver(self.pool, self)
            with self.pool.connection() as c:
                if c.execute('SELECT max(v) AS version FROM checkpoint_migrations').fetchone()['version'] != len(self.saver.MIGRATIONS) - 1:
                    raise RuntimeError('Checkpoint migration version mismatch')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        with self.guard:
            if self.connection is None or self.connection.closed: raise RuntimeError('Server ownership lost')
            self.connection.execute('SELECT 1')

    def __exit__(self, *_):
        if self.pool is not None: self.pool.close()
        if self.connection is not None: self.connection.close()
        self.connection = None
