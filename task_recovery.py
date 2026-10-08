"""Recovery metadata and fenced claims; no checkpoint bytes exposed to clients."""
import json
from generation_tasks import FIELDS
from web_storage import _utc_now

MIGRATION = 'v12_web_recovery_v1'


def schema_sql(table, tasks, cleanup):
    return [f'''CREATE TABLE {table} (
        task_id TEXT PRIMARY KEY REFERENCES {tasks}(id) ON DELETE CASCADE,
        version TEXT NOT NULL, scope TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0, epoch INTEGER NOT NULL DEFAULT 0)''',
        f'CREATE TABLE {cleanup} (task_id TEXT PRIMARY KEY)']


class RecoveryStore:
    def __init__(self, tasks):
        self.tasks = tasks
        tasks.has_recovery = True
        with tasks.cursor() as c:
            if tasks.pg:
                tasks.execute(c, f'SELECT 1 FROM {tasks.table("schema_migrations")} WHERE name=?', (MIGRATION,))
                if c.fetchone() is None: raise RuntimeError('Run prepare_web_recovery.py --apply first')
            else:
                c.execute("SELECT name FROM sqlite_master WHERE name='generation_recovery'")
                if c.fetchone() is None:
                    for statement in schema_sql(tasks.table('generation_recovery'), tasks.table('generation_tasks'), tasks.table('checkpoint_cleanup')):
                        c.execute(statement)

    def pending(self):
        t = self.tasks
        with t.cursor() as c:
            fields = ','.join('t.' + f for f in FIELDS.split(','))
            t.execute(c, f'SELECT {fields} FROM {t.table("generation_tasks")} t JOIN {t.table("generation_recovery")} r ON r.task_id=t.id WHERE t.status=? ORDER BY t.created_at,t.id', ('running',))
            return [t.decode(row) for row in c.fetchall()]

    def claim(self, tid, *, recovering):
        t = self.tasks
        with t.cursor() as c:
            t.execute(c, f'SELECT status FROM {t.table("generation_tasks")} WHERE id=?' + (' FOR UPDATE' if t.pg else ''), (tid,))
            row = c.fetchone()
            if row is None or row[0] != 'running': return None
            t.execute(c, f'SELECT version,scope,attempts,epoch FROM {t.table("generation_recovery")} WHERE task_id=?', (tid,))
            row = c.fetchone()
            if row is None: return None
            version, scope, attempts, epoch = row
            if recovering and attempts >= 3:
                t.execute(c, f'UPDATE {t.table("generation_tasks")} SET status=?,error=?,updated_at=? WHERE id=?',
                          ('interrupted', '自动恢复次数已达上限，请重新发送。', _utc_now(), tid))
                return None
            attempts += int(recovering)
            epoch += 1
            t.execute(c, f'UPDATE {t.table("generation_recovery")} SET attempts=?,epoch=? WHERE task_id=?', (attempts, epoch, tid))
            return dict(version=version, scope=json.loads(scope), attempts=attempts, epoch=epoch)

    def interrupt_legacy(self):
        t = self.tasks
        with t.cursor() as c:
            t.execute(c, f'UPDATE {t.table("generation_tasks")} SET status=?,error=?,updated_at=? WHERE status=? AND id NOT IN (SELECT task_id FROM {t.table("generation_recovery")})',
                      ('interrupted', '服务已重启，旧版任务无法恢复，请重新发送。', _utc_now(), 'running'))

    def cleanup_ids(self):
        t = self.tasks
        with t.cursor() as c:
            t.execute(c, f'SELECT task_id FROM {t.table("checkpoint_cleanup")}')
            return [r[0] for r in c.fetchall()]

    def cleaned(self, tid):
        t = self.tasks
        with t.cursor() as c:
            t.execute(c, f'DELETE FROM {t.table("checkpoint_cleanup")} WHERE task_id=?', (tid,))
