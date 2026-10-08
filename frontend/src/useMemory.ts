import { useEffect, useRef, useState } from 'react';
export type MemorySource = {kind:string;reference?:string|null;quote:string;candidate_content?:string;recorded_at?:string};
export type MemoryFact = {id:string;layer:string;category?:string;content:string;basis:string;protected:boolean;sources:MemorySource[]};
export type MemorySuggestion = {id:string;memory_revision:number;policy_epoch:number;proposal:{text?:string;target?:string;relation:string;layer:string;source_quotes?:string[];
  action?:string;content?:string;basis?:string;sources?:MemorySource[];target_ids?:string[]}};
export type MemoryDocument = { core_text:string; extended_text:string; enabled:boolean; revision:number; updated_at:string|null;
  auto_accumulate?:boolean;policy_epoch?:number;
  facts?:MemoryFact[];
  reflection_status?:{id:string;status:string;result?:{added?:number;revision?:number;error?:string}}[];suggestions?:MemorySuggestion[];
  limits?: { core_chars:number; core_tokens:number; extended_chars:number } };
export type MemoryRequest = (path:string, options?:RequestInit)=>Promise<unknown>;
const empty:MemoryDocument={ core_text:'', extended_text:'', enabled:true, revision:0, updated_at:null };
export function useMemory(owner:string, request:MemoryRequest) {
  const [document,setDocument]=useState<MemoryDocument>(empty);
  const [draft,setDraft]=useState<MemoryDocument>(empty);
  const [loading,setLoading]=useState(true), [saving,setSaving]=useState(false);
  const [error,setError]=useState(''), [success,setSuccess]=useState('');
  const [ready,setReady]=useState(false);
  const epoch=useRef(0), busy=useRef(false), requestRef=useRef(request);
  const polling=useRef(false);
  const current=useRef({document,draft,ready});current.current={document,draft,ready};
  requestRef.current=request;
  async function load() {
    if(busy.current)return;
    const ticket=++epoch.current;
    setLoading(true); setReady(false); setError(''); setSuccess('');
    try {
      const data=await requestRef.current('/api/memory') as MemoryDocument;
      if(ticket!==epoch.current)return;
      setDocument(data); setDraft(data); setReady(true);
    } catch(e) { if(ticket===epoch.current)setError(e instanceof Error?e.message:'暂时无法读取记忆'); }
    finally {if(ticket===epoch.current)setLoading(false);}
  }
  useEffect(()=>{
    setDocument(empty);setDraft(empty);setSaving(false);busy.current=false;
    void load(); return ()=>{epoch.current++;};
  },[owner]);
  const dirty=document.core_text!==draft.core_text || document.extended_text!==draft.extended_text || document.enabled!==draft.enabled;
  useEffect(()=>{
    if(document.auto_accumulate===undefined)return;
    const timer=window.setInterval(async()=>{
      const state=current.current;
      if(window.document.visibilityState==='hidden'||polling.current||busy.current||!state.ready||state.document.core_text!==state.draft.core_text||state.document.extended_text!==state.draft.extended_text||state.document.enabled!==state.draft.enabled)return;
      const ticket=epoch.current;polling.current=true;
      try {
        const data=await requestRef.current('/api/memory') as MemoryDocument;
        // Edits made while the poll was in flight remain intact.
        const latest=current.current;
        if(ticket===epoch.current&&latest.document.core_text===latest.draft.core_text&&latest.document.extended_text===latest.draft.extended_text&&latest.document.enabled===latest.draft.enabled){setDocument(data);setDraft(data);}
      } catch { /* Manual reload exposes failures without interrupting edits. */ }
      finally {polling.current=false;}
    },5000);
    return ()=>window.clearInterval(timer);
  },[owner,document.auto_accumulate]);
  function edit(patch:Partial<MemoryDocument>) {setDraft(old=>({...old,...patch}));setSuccess('');}
  async function save() {
    if(busy.current || loading || !ready)return;
    busy.current=true;setSaving(true);setError('');setSuccess('');
    const ticket=++epoch.current;
    try {
      const data=await requestRef.current('/api/memory',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        core_text:draft.core_text,extended_text:draft.extended_text,enabled:draft.enabled,expected_revision:document.revision})}) as MemoryDocument;
      if(ticket!==epoch.current)return;
      setDocument(data);setDraft(data);setSuccess('记忆已保存');
    } catch(e) {if(ticket===epoch.current)setError(e instanceof Error?e.message:'保存失败，编辑内容已保留');}
    finally {if(ticket===epoch.current){busy.current=false;setSaving(false);}}
  }
  async function action(path:string,body?:unknown,refresh=true) {
    if(busy.current||loading||!ready||dirty)return;
    const ticket=++epoch.current;busy.current=true;setSaving(true);setError('');setSuccess('');
    try {
      const result=await requestRef.current(path,{method:path.endsWith('/automation')?'PUT':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body??{})}) as Partial<MemoryDocument>;
      if(ticket!==epoch.current)return;
      if(refresh){const data=await requestRef.current('/api/memory') as MemoryDocument;if(ticket!==epoch.current)return;setDocument(data);setDraft(data);}
      else {const policy=result as Partial<MemoryDocument>&{epoch?:number};const patch={auto_accumulate:policy.auto_accumulate,policy_epoch:policy.epoch};setDocument(old=>({...old,...patch}));setDraft(old=>({...old,...patch}));}
    } catch(e){if(ticket===epoch.current)setError(e instanceof Error?e.message:'操作失败，请重新加载后重试');}
    finally {if(ticket===epoch.current){busy.current=false;setSaving(false);}}
  }
  function automate(enabled:boolean){return action('/api/memory/automation',{enabled,expected_epoch:document.policy_epoch??0},false);}
  function suggestion(id:string,accept:boolean){return action(`/api/memory/suggestions/${encodeURIComponent(id)}/${accept?'accept':'ignore'}`,accept?{expected_revision:document.revision,expected_epoch:document.policy_epoch??0}:undefined);}
  function retry(id:string){return action(`/api/memory/reflection/${encodeURIComponent(id)}/retry`);}
  return {document,draft,loading,saving,ready,error,success,dirty,edit,save,load,automate,suggestion,retry};
}
