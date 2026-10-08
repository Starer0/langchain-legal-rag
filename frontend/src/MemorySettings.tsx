import { useEffect, useRef, useState } from 'react';
import { X, Brain, Save } from 'lucide-react';
import { useMemory, type MemoryRequest } from './useMemory';

export function MemorySettings({owner,request,onClose}:{owner:string;request:MemoryRequest;onClose:()=>void}) {
  const memory=useMemory(owner,request);
  const [confirm,setConfirm]=useState<'core'|'extended'|'all'|'close'|'reload'|null>(null);
  const [reviewId,setReviewId]=useState<string|null>(null);
  const reviewCancel=useRef<HTMLButtonElement>(null);
  const panel=useRef<HTMLDivElement>(null);
  const confirmation=useRef<HTMLDivElement>(null);
  const cancelConfirmation=useRef<HTMLButtonElement>(null);
  const state=useRef({memory,onClose});state.current={memory,onClose};
  function close() { if(memory.saving)return; if(memory.dirty)setConfirm('close');else onClose(); }
  useEffect(()=>{
    const old=document.activeElement as HTMLElement|null;
    panel.current?.querySelector<HTMLButtonElement>('button')?.focus();
    function keyboard(e:KeyboardEvent) {
      if(e.key==='Escape') {
        e.preventDefault();
        const {memory,onClose}=state.current;
        if(!memory.saving){ if(memory.dirty)setConfirm('close');else onClose(); }
      }
      if(e.key==='Tab') {
        const nodes=panel.current?.querySelectorAll<HTMLElement>('button:not(:disabled), textarea:not(:disabled), input:not(:disabled), summary');
        if(!nodes?.length)return;
        const first=nodes[0],last=nodes[nodes.length-1];
        if(e.shiftKey && (document.activeElement===first || !panel.current?.contains(document.activeElement))){e.preventDefault();last.focus();}
        else if(!e.shiftKey && document.activeElement===last){e.preventDefault();first.focus();}
      }
    }
    document.addEventListener('keydown',keyboard);
    return ()=>{document.removeEventListener('keydown',keyboard);old?.focus();};
  },[]);
  useEffect(()=>{setConfirm(null);},[owner]);
  useEffect(()=>{if(reviewId){reviewCancel.current?.scrollIntoView?.({block:'nearest'});reviewCancel.current?.focus();}},[reviewId]);
  useEffect(()=>{
    if(confirm){
      confirmation.current?.scrollIntoView?.({block:'nearest'});
      cancelConfirmation.current?.focus();
    }
  },[confirm]);
  function confirmed() {
    if(memory.saving)return;
    if(confirm==='close')onClose();
    else if(confirm==='reload')void memory.load();
    else if(confirm==='core')memory.edit({core_text:''});
    else if(confirm==='extended')memory.edit({extended_text:''});
    else if(confirm==='all')memory.edit({core_text:'',extended_text:''});
    setConfirm(null);
  }
  return <div className='memory-backdrop'><div ref={panel} className='memory-panel' role='dialog' aria-modal='true' aria-labelledby='memory-title'>
    <header className='memory-header'><div><Brain size={20}/><h2 id='memory-title'>长期记忆</h2></div>
      <button className='icon-button' aria-label='关闭记忆设置' onClick={close} disabled={memory.saving}><X size={20}/></button></header>
    <p className='memory-intro'>保存你的回答偏好与背景，新对话也能使用。仅当前账号可查看和修改。删除对话后已保存的账号记忆仍保留，请在这里单独清空并保存。</p>
    {memory.loading ? <p role='status'>正在读取记忆…</p> : <>
      <label className='memory-toggle'><input type='checkbox' checked={memory.draft.enabled} disabled={memory.saving||!memory.ready}
        onChange={e=>memory.edit({enabled:e.target.checked})}/>在回答中使用记忆</label>
      <p className='memory-help'>关闭后保留内容，回答时暂停使用，同时暂停后台积累。</p>
      {memory.document.auto_accumulate!==undefined&&<section className='memory-field'>
        <label className='memory-toggle'><input type='checkbox' checked={memory.document.auto_accumulate} disabled={memory.saving||memory.dirty||!memory.ready}
          onChange={e=>void memory.automate(e.target.checked)}/>自动积累长期记忆（后台处理）</label>
        <p className='memory-help'>开启后，服务器在对话空闲约两分钟或整理上下文后回顾稳定偏好。关闭浏览器也能继续；案件条件不会作为个人记忆保存。</p>
        {memory.dirty&&<p className='memory-help'>内容尚未保存。自动积累开关仍保持原设置，请先保存或放弃编辑再调整。</p>}
        {memory.document.auto_accumulate&&!memory.document.enabled&&<p role='status'>后台积累已暂停，请开启并保存“在回答中使用记忆”。</p>}
        {memory.document.reflection_status?.filter((job,index)=>index===0||job.status==='failed').map(job=><div key={job.id}>
          {job.status==='pending'&&<p role='status'>等待后台整理</p>}
          {job.status==='running'&&<p role='status'>正在后台整理记忆…</p>}
          {job.status==='completed'&&<p role='status'>{(job.result?.added??0)>0&&job.result?.revision===memory.document.revision?'已更新长期记忆':'后台整理完成'}</p>}
          {job.status==='failed'&&<><p role='status'>后台整理暂未完成，现有记忆已保留。</p><button disabled={memory.saving||memory.dirty} onClick={()=>void memory.retry(job.id)}>重试整理</button></>}
        </div>)}
      </section>}
      {memory.document.suggestions?.map(item=><section className='memory-field' key={item.id} aria-label='记忆更新建议'>
        <div className='memory-field-title'><strong>{item.proposal.relation==='capacity'?'记忆容量不足':'需要确认的记忆建议'}</strong></div>
        {item.proposal.target&&<><p className='memory-help'>当前内容</p><p>{item.proposal.target}</p></>}
        <p className='memory-help'>建议内容（来自你的发言）</p><p>{item.proposal.text}</p>
        {item.proposal.action&&<>
          <p>{item.proposal.content || (item.proposal.action==='request_clear'?'请求清空'+(item.proposal.layer==='core'?'核心':'扩展')+'记忆':'请求删除指定条目')}</p>
          {item.proposal.basis==='inferred'&&<p className='memory-help'>推测，待你确认</p>}
          {item.proposal.target_ids?.map(id=><p key={id}>当前内容：{memory.document.facts?.find(f=>f.id===id)?.content || '目标已变化，请重新加载'}</p>)}
          <details><summary>查看建议来源</summary>{item.proposal.sources?.map((source,index)=><blockquote key={index}>{source.quote}</blockquote>)}</details>
        </>}
        <div className='memory-field-footer'>
          {item.proposal.relation==='capacity'?<span>请手动精简记忆后添加。</span>:<button disabled={memory.saving||memory.dirty} onClick={()=>
            item.proposal.action?.startsWith('request_')||item.proposal.relation==='conflict'?setReviewId(item.id):void memory.suggestion(item.id,true)}>接受更新</button>}
          <button disabled={memory.saving||memory.dirty} onClick={()=>void memory.suggestion(item.id,false)}>忽略建议</button>
        </div>
        {reviewId===item.id&&<div className='memory-confirm' role='alert'><p>确认按上述建议修改记忆？请核对当前内容与建议内容。</p>
          <button ref={reviewCancel} disabled={memory.saving} onClick={()=>setReviewId(null)}>取消</button>
          <button disabled={memory.saving||memory.dirty} onClick={()=>{setReviewId(null);void memory.suggestion(item.id,true);}}>确认更新</button></div>}
      </section>)}
      {!!memory.document.facts?.length&&<details className='memory-field'><summary>查看记忆来源</summary>
        <p className='memory-help'>归纳文字与原话可分别核对。删除对话后保留摘录，原消息位置可能不可用。</p>
        {memory.document.facts.map(fact=><section key={fact.id}>
          <strong>{fact.content}</strong><p className='memory-help'>{fact.basis==='inferred'?'推测':'用户声明'}{fact.protected?' · 手动或旧文本保护':''}</p>
          {fact.sources.map((source,index)=><div key={index}><blockquote>{source.quote}</blockquote>
            {source.candidate_content&&source.candidate_content!==fact.content&&<p>候选归纳：{source.candidate_content}</p>}
            <p className='memory-help'>{source.kind==='manual_document'?'手动编辑':source.kind==='legacy_document'?'旧版文本':'用户发言'}
              {source.recorded_at?' · '+new Date(source.recorded_at).toLocaleString():''}</p></div>)}
        </section>)}
      </details>}
      <section className='memory-field'><div className='memory-field-title'><label htmlFor='memory-core'>核心记忆</label><span>每次回答使用</span></div>
        <p id='core-help'>语言、详略、解释方式等通用偏好。请保持简短；约500 token为估算预算。</p>
        <textarea id='memory-core' aria-describedby='core-help' value={memory.draft.core_text} disabled={memory.saving||!memory.ready}
          rows={4} onChange={e=>memory.edit({core_text:e.target.value})} placeholder='例如：先给结论，再分点解释，保留法条来源。'/>
        <div className='memory-field-footer'><span>{memory.draft.core_text.length} / {memory.document.limits?.core_chars??1500} 字符</span>
          <button type='button' disabled={memory.saving} onClick={()=>setConfirm('core')}>清空核心记忆</button></div></section>
      <section className='memory-field'><div className='memory-field-title'><label htmlFor='memory-extended'>扩展记忆</label><span>相关时使用</span></div>
        <p id='extended-help'>特定主题的稳定背景，建议用空行分段。内容会通过现有模型服务生成检索向量。</p>
        <textarea id='memory-extended' aria-describedby='extended-help' value={memory.draft.extended_text} disabled={memory.saving||!memory.ready}
          rows={5} onChange={e=>memory.edit({extended_text:e.target.value})} placeholder='例如：我是法律知识初学者，解释专业概念时请举例。'/>
        <div className='memory-field-footer'><span>{memory.draft.extended_text.length} / {memory.document.limits?.extended_chars??12000} 字符</span>
          <button type='button' disabled={memory.saving} onClick={()=>setConfirm('extended')}>清空扩展记忆</button></div></section>
      {memory.dirty&&<p className='memory-help' role='status'>内容尚未保存，请点击底部“保存记忆”。清空文本不会关闭自动积累。</p>}
      <p className='memory-help'>案件日期、金额和合同条件请留在当前对话。修改后从下一次提问生效。</p>
    </>}
    {memory.error&&<p className='memory-error' role='alert'>{memory.error}</p>}
    {memory.success&&<p className='memory-success' role='status'>{memory.success}</p>}
    {confirm&&<div ref={confirmation} className='memory-confirm' role='alert'>
      <p>{confirm==='close'?'还有未保存的修改，要放弃吗？':confirm==='reload'?'重新加载会放弃当前编辑内容。': '清空后需点击保存才会生效。历史回答不会被删除。'}</p>
      <button ref={cancelConfirmation} type='button' disabled={memory.saving} onClick={()=>setConfirm(null)}>取消</button>
      <button type='button' disabled={memory.saving} onClick={confirmed}>{confirm==='close'?'放弃修改':confirm==='reload'?'确认重新加载':'确认清空'}</button>
    </div>}
    <footer className='memory-actions'><button type='button' disabled={memory.saving||memory.loading} onClick={()=>setConfirm('all')}>清空全部</button>
      <button type='button' disabled={memory.saving} onClick={()=>memory.dirty?setConfirm('reload'):void memory.load()}>重新加载</button>
      <button className='memory-save' type='button' disabled={memory.saving||memory.loading||!memory.ready||!memory.dirty} onClick={()=>void memory.save()}>
        <Save size={16}/>{memory.saving?'正在保存…':'保存记忆'}</button></footer>
  </div></div>;
}
