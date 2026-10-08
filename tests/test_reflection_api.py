import os
import unittest
from fastapi.testclient import TestClient
import test_reflection_storage as storage
from web_app import create_app


class ReflectionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS')!='1':raise unittest.SkipTest('Real temporary PG required')
    def setUp(self):
        self.f=storage.ReflectionStorageTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.app=create_app(self.f.f.store,object(),authentication=self.f.f.auth,memory_store=self.f.memory,memory_service=self.f.service,reflection_store=self.f.store)

    def client(self,username='demo_ab'):
        c=TestClient(self.app);self.addCleanup(c.close);self.f.f.login_client(c,username);return c

    def test_opt_in_csrf_no_store_and_stale_policy(self):
        c=self.client();doc=c.get('/api/memory')
        self.assertFalse(doc.json()['auto_accumulate'])
        self.assertEqual(doc.headers['cache-control'],'no-store')
        response=c.put('/api/memory/automation',json={'enabled':True,'expected_epoch':0})
        self.assertEqual(response.status_code,200)
        self.assertEqual(c.put('/api/memory/automation',json={'enabled':False,'expected_epoch':0}).status_code,409)
        c.headers.pop('x-csrf-token')
        self.assertEqual(c.put('/api/memory/automation',json={'enabled':False,'expected_epoch':1}).status_code,403)

    def make_conflict(self):
        old=self.f.memory.read(self.f.owner)
        self.f.memory.save(self.f.owner,self.f.service.prepare(old,'我喜欢详细解释','',True),0)
        self.f.enable();self.f.enqueue('我喜欢简短回答');job=self.f.store.claim(1120)
        p=storage.candidate('我喜欢简短回答','conflict','我喜欢详细解释')
        self.f.store.apply(job['id'],job['epoch'],{'additions':[],'suggestions':[p],'skipped':[]},None)
        return self.f.store.suggestions(self.f.owner)[0]

    def test_conflict_accept_is_owned_versioned_and_once(self):
        suggestion=self.make_conflict();c=self.client();other=self.client('demo_c')
        url=f"/api/memory/suggestions/{suggestion['id']}/accept"
        payload={'expected_revision':1,'expected_epoch':suggestion['policy_epoch']}
        self.assertEqual(other.post(url,json=payload).status_code,404)
        self.assertEqual(c.post(url,json=payload).status_code,200)
        self.assertEqual(self.f.memory.read(self.f.owner)['core_text'],'我喜欢简短回答')
        self.assertEqual(c.post(url,json=payload).status_code,409)
        self.assertEqual(self.f.memory.read(self.f.owner)['revision'],2)

    def test_stale_suggestion_cannot_restore_deleted_text(self):
        suggestion=self.make_conflict();c=self.client()
        old=self.f.memory.read(self.f.owner)
        self.f.memory.save(self.f.owner,self.f.service.prepare(old,'','',True),old['revision'])
        payload={'expected_revision':1,'expected_epoch':suggestion['policy_epoch']}
        self.assertEqual(c.post(f"/api/memory/suggestions/{suggestion['id']}/accept",json=payload).status_code,409)
        self.assertEqual(self.f.memory.read(self.f.owner)['core_text'],'')

    def test_get_suggestions_and_ignore_do_not_change_memory(self):
        suggestion=self.make_conflict();c=self.client()
        self.assertEqual(len(c.get('/api/memory/suggestions').json()['suggestions']),1)
        self.assertEqual(c.post(f"/api/memory/suggestions/{suggestion['id']}/ignore").status_code,200)
        self.assertEqual(c.get('/api/memory/suggestions').json()['suggestions'],[])
        self.assertEqual(self.f.memory.read(self.f.owner)['revision'],1)

    def test_foreign_retry_denied_and_safe_status(self):
        self.f.enable();job=self.f.enqueue();other=self.client('demo_c');c=self.client()
        self.assertEqual(other.post(f"/api/memory/reflection/{job['id']}/retry").status_code,404)
        status=c.get('/api/memory').json()['reflection_status']
        self.assertNotIn('input',str(status));self.assertNotIn('我喜欢简短回答',str(status))

    def test_post_commit_status_failure_does_not_report_save_failed(self):
        from unittest.mock import patch
        c=self.client()
        with patch.object(self.f.store,'jobs',side_effect=RuntimeError('status unavailable')):
            response=c.put('/api/memory',json={'core_text':'我喜欢简短回答','extended_text':'','enabled':True,'expected_revision':0})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['revision'],1)
        self.assertEqual(self.f.memory.read(self.f.owner)['revision'],1)


if __name__=='__main__':unittest.main()
