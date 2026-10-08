"""Offline by default. Explicit --collect makes paid provider calls with synthetic inputs only."""
import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def collect(cases):
    from database_settings import read_settings
    from conversation_context import create_context_service,SourceMessage
    from web_understanding import TurnUnderstanding
    from memory_reflection import ReflectionService
    settings=read_settings();model=create_context_service(settings,request_timeout=20).model
    rows=[]
    class ObservedModel:
        def __init__(self,delegate):self.delegate=delegate;self.usage=None;self.native=[]
        def bind_tools(self,schemas):return Bound(self,self.delegate.bind_tools(schemas))
    class Bound:
        def __init__(self,parent,delegate):self.parent,self.delegate=parent,delegate
        def invoke(self,prompt):
            response=self.delegate.invoke(prompt)
            self.parent.usage=response.usage_metadata
            self.parent.native=[call['name'] for call in response.tool_calls]
            return response
    for case in cases:
        observed=ObservedModel(model);start=time.perf_counter()
        row={'id':case['id'],'model':settings.get('MODEL_NAME','deepseek-chat'),'temperature':0,
             'recorded_at':datetime.now(timezone.utc).isoformat(),'calls':1}
        doc=dict(core_text='\n\n'.join(case.get('existing',[])),extended_text='',enabled=True,
                 auto_accumulate=True,revision=0,facts=[dict(id=f'f{i}',content=text,layer='core',category='preference',basis='declared',protected=True) for i,text in enumerate(case.get('existing',[]))])
        try:
            if case.get('scope')=='background':
                messages=[SourceMessage(i*2+1,'user',text) for i,text in enumerate(case['messages'])]
                doc['review_sources']=[dict(kind='message',reference=f'fixture:{m.ordinal}',role='user',content=m.content,new=True) for m in messages]
                call=ReflectionService(observed,None).review(messages,doc,[])
                row.update(route=None,memory_request=False,operations=asdict(call)['operations'] if call else [])
            else:
                context=dict(recent_user_messages=[dict(ordinal=i*2+1,role='user',content=text) for i,text in enumerate(case.get('history',[]))],
                    sources=[dict(kind='current_input',reference='fixture-turn',role='user',content=case['question'],new=True)])
                decision=TurnUnderstanding(observed).decide(case['question'],context,doc)
                row.update(route=decision.answer.route,memory_request=decision.answer.memory_request,
                    operations=asdict(decision.memory_call)['operations'] if decision.memory_call else [],decision=asdict(decision))
        except Exception as error:row.update(error_type=type(error).__name__,operations=[],route=None,memory_request=None)
        row.update(duration_ms=round((time.perf_counter()-start)*1000,3),usage=observed.usage,native_tools=observed.native)
        rows.append(row)
    return rows


def evaluate(cases,rows=None):
    observed={row['id']:row for row in rows or []};results=[]
    for case in cases:
        row=observed.get(case['id'])
        metrics=None
        if row is not None:
            operations=row.get('operations',[]);texts=[o.get('delta') if o.get('relation')=='complement' else o.get('content','') for o in operations]
            text='\n'.join(filter(None,texts));actions=[o['action'] for o in operations]
            metrics={'route_error':int(case.get('route') is not None and row.get('route')!=case['route']),
                'scope_split_error':int('out_of_scope_request' in case and row.get('decision',{}).get('answer',{}).get('out_of_scope_request')!=case['out_of_scope_request']),
                'explicit_save_error':int('memory_request' in case and row.get('memory_request')!=case['memory_request']),
                'tool_omission':int(bool(case['actions']) and not actions),
                'unexpected_tool':int(not case['actions'] and bool(actions)),
                'invalid_action':sum(action not in case['actions'] for action in actions),
                'duplicate_facts':len(texts)-len(set(texts)),
                'lost_required_fragments':sum(fragment not in text for fragment in case['required']),
                'unsupported_fragments':sum(fragment in text for fragment in case['forbidden']),
                'basis_error':int('basis' in case and any(o['basis']!=case['basis'] for o in operations))}
        results.append({'id':case['id'],'metrics':metrics,'observation':row})
    totals={key:sum(item['metrics'][key] for item in results if item['metrics']) for key in next((item['metrics'] for item in results if item['metrics']),{})}
    return {'fixture_count':len(cases),'measured_count':len(observed),'mode':'provider_observations' if rows else 'offline_inventory',
        'metrics':totals or None,'manual_semantic_review':None,'manual_overwrite':None,'cost':None,
        'note':'Fragment checks are aids, not semantic accuracy. Mechanism tests are separate. Unmeasured values remain null.',
        'cases':results}


def legacy_routes(cases):
    """Actual previous web entry rules; no provider call or invented tool result."""
    from web_intent import is_style_preference,has_legal_request
    from memory_service import command_candidate
    results=[]
    for case in cases:
        if case.get('scope')=='background':continue
        q=case['question']
        route='preference' if is_style_preference(q) or (command_candidate(q) and not has_legal_request(q)) else 'rag'
        results.append(dict(id=case['id'],route=route,route_error=int(route!=case['route'])))
    return dict(mode='old_entry_rules_only',cases=results,route_errors=sum(r['route_error'] for r in results),
        semantic_memory_metrics=None,cost=None,note='Entry routing comparison only; not an old-provider semantic benchmark.')


def collect_legacy(cases):
    """Same provider/settings and fixtures, previous routing and memory parser.

    No legal answering, database writes or real embeddings. The deterministic
    embedding stand-in only allows the old prepare boundary to be inspected.
    """
    from database_settings import read_settings
    from conversation_context import create_context_service,SourceMessage
    from memory_service import MemoryService,command_candidate
    from memory_reflection import ReflectionService
    settings=read_settings();model=create_context_service(settings,request_timeout=20).model
    routes={r['id']:r['route'] for r in legacy_routes(cases)['cases']};rows=[]
    class Vectors:
        def embed_documents(self,texts):return [[1.,0.] for text in texts]
    class Observer:
        def __init__(self):self.calls=0;self.usage=None
        def invoke(self,prompt):
            self.calls+=1;response=model.invoke(prompt);self.usage=response.usage_metadata;return response
    for case in cases:
        observer=Observer();start=time.perf_counter()
        doc=dict(core_text='\n\n'.join(case.get('existing',[])),extended_text='',enabled=True,revision=0,entries=[])
        row=dict(id=case['id'],model=settings.get('MODEL_NAME','deepseek-chat'),temperature=0,
            recorded_at=datetime.now(timezone.utc).isoformat(),operations=[],native_tools=[],
            route=routes.get(case['id']),memory_request=False)
        try:
            service=MemoryService(Vectors(),command_model=observer,fingerprint='isolated-eval')
            if case.get('scope')=='background':
                source=[SourceMessage(i*2+1,'user',q) for i,q in enumerate(case['messages'])]
                result=ReflectionService(observer,service).extract(source,doc,[])
                row['operations']=[dict(action='remember' if p['relation']=='new' else 'propose_change',content=p['text'],basis='declared') for p in result['additions']+result['suggestions']]
            elif row['route']=='preference' and command_candidate(case['question']):
                row['memory_request']=True
                result=service.propose(case['question'],doc)
                if 'operation' in result:
                    op=result['operation'];row['operations']=[dict(action={'add':'remember','replace':'propose_change','delete':'request_delete','clear':'request_clear'}[op['action']],content=op.get('content',''),basis='declared')]
                    row['prepared_document']={k:result['prepared'][k] for k in ('core_text','extended_text')}
        except Exception as error:row['error_type']=type(error).__name__
        row.update(calls=observer.calls,usage=observer.usage,duration_ms=round((time.perf_counter()-start)*1000,3))
        rows.append(row)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',default='tests/fixtures/model_memory_cases.json');parser.add_argument('--output',required=True)
    parser.add_argument('--model-results');parser.add_argument('--collect',action='store_true');parser.add_argument('--case',action='append')
    parser.add_argument('--legacy-routing-baseline',action='store_true')
    parser.add_argument('--collect-legacy',action='store_true')
    args=parser.parse_args()
    if sum((args.collect,args.collect_legacy,bool(args.model_results)))>1:parser.error('Choose one result source')
    cases=json.loads(Path(args.fixtures).read_text(encoding='utf-8'))
    if args.case:cases=[c for c in cases if c['id'] in args.case]
    rows=collect(cases) if args.collect else collect_legacy(cases) if args.collect_legacy else json.loads(Path(args.model_results).read_text(encoding='utf-8')) if args.model_results else None
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report=evaluate(cases,rows)
    if args.collect_legacy:report['mode']='legacy_memory_provider_observations';report['note']+=' Old memory parser only; no legal generation; embeddings are isolated stand-ins.'
    if args.legacy_routing_baseline:report['legacy_routing_baseline']=legacy_routes(cases)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.collect or args.collect_legacy:output.with_suffix('.observations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'cases':len(cases),'measured':len(rows or [])}))
