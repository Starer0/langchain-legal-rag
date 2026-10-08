"""Server-owned one-worker reflection; browser connections do not own jobs."""
import logging
import time
from threading import Event,Thread
from conversation_context import SourceMessage
from memory_service import MemoryValidation
from request_logging import write_event


class SystemClock:
    def now(self):return time.time()


class ReflectionRunner:
    def __init__(self,store,service,*,clock=None,active_generation=None,lease=None):
        self.store,self.service,self.clock,self.lease=store,service,clock or SystemClock(),lease
        self.active_generation=active_generation
        self.stop=Event();self.signal=Event();self.thread=None
        self.logger=logging.getLogger('legal_rag.requests')

    def alive(self):
        if self.stop.is_set():raise RuntimeError('Reflection shutting down')
        if self.lease is not None:self.lease.check()

    def pump(self):
        self.alive()
        if self.active_generation is not None and self.active_generation():return False
        job=self.store.claim(self.clock.now())
        if job is None:return False
        started=time.monotonic()
        try:
            existing=self.store.memory.read(job['user_id'])
            messages=[SourceMessage(**m) for m in job['input']['messages']]
            context=[SourceMessage(**m) for m in job['input'].get('context',[])]
            if getattr(self.store.memory,'tools',None) is not None:
                from memory_tool_contract import ToolContext,MemoryToolCall
                existing=self.store.memory.snapshot(job['user_id'])
                def refs(items,new):
                    return [dict(kind='message',reference=f"{job['conversation_id']}:{m.ordinal}",role=m.role,content=m.content,new=new) for m in items]
                sources=refs(messages,True)+refs(context,False)
                existing['review_sources']=sources
                existing['suggestions']=self.store.suggestions(job['user_id'])
                call=self.service.review(messages,existing,context) or MemoryToolCall('',())
                self.alive()
                tool_context=ToolContext(job['user_id'],'background',job['id'],job['memory_revision'],job['policy_epoch'],
                    dict(document=existing,conversation_id=job['conversation_id'],sources=sources,question='',memory_request=False))
                update=self.store.memory.tools.prepare(call,tool_context)
                result=self.store.apply_tool(job['id'],job['epoch'],update)
                checked={}
            else:
                result=self._legacy_apply(job,messages,existing,context)
                checked={}
        except Exception as error:
            if not self.stop.is_set():
                self.store.finish(job['id'],job['epoch'],{'error':'暂时无法整理记忆，可查看设置后主动重试。','error_type':type(error).__name__})
            write_event(self.logger,'memory_reflection_failed',job_id=job['id'],user_id=job['user_id'],error_type=type(error).__name__)
            return True
        # Diagnostics never turn a committed result into a failed task.
        try:
            write_event(self.logger,'memory_reflection_completed',job_id=job['id'],user_id=job['user_id'],
                conversation_id=job['conversation_id'],source_start=job['start_ordinal'],source_end=job['end_ordinal'],
                reason=job['reason'],duration_ms=round((time.monotonic()-started)*1000,3),model_calls=1,**result)
        except Exception:pass
        return True

    def _legacy_apply(self,job,messages,existing,context):
            checked=self.service.extract(messages,existing,context)
            self.alive()
            try:prepared=self.service.prepare(existing,checked)
            except MemoryValidation:
                checked['suggestions'] += [{**c,'relation':'capacity','target':''} for c in checked['additions']]
                checked['additions']=[];prepared=None
            self.alive()
            result=self.store.apply(job['id'],job['epoch'],checked,prepared)
            return result

    def start(self):
        if self.thread is not None:return
        self.alive();self.store.recover(self.clock.now())
        self.thread=Thread(target=self._run,daemon=True);self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                if self.pump():continue
            except Exception as error:
                write_event(self.logger,'memory_reflection_service_error',error_type=type(error).__name__)
                if self.lease is not None:
                    try:self.lease.check()
                    except Exception:return
            self.signal.wait(.5);self.signal.clear()

    def wake(self):self.signal.set()
    def shutdown(self):
        self.stop.set();self.signal.set()
        if self.thread is not None:self.thread.join()
