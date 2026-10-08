"""Fixed synthetic acceptance batch. Default: no network or paid calls."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conversation_context import ContextService,SourceMessage,Summary,ContextLimits,create_context_service
from memory_reflection import ReflectionService

def proposal(text,relation='new',target=''):
    return dict(layer='core',text=text,source_ordinals=[1],source_quotes=[text],relation=relation,target=target)

def run(live=False):
    inputs=[('stable','我喜欢简短回答','',1,0),('temporary','这次简短一点','',0,0),
            ('case','我的合同期限两年','',0,0),('thirdparty','同事喜欢简短回答','',0,0),
            ('duplicate','我喜欢简短回答','我喜欢简短回答',0,0),
            ('conflict','我喜欢简短回答','我喜欢详细解释',0,1)]
    calls=0;rows=[]
    class FakeModel:
        def invoke(self,prompt):
            nonlocal calls
            calls+=1;values=json.loads(prompt.rsplit('\n',1)[-1]);text=values['user_messages'][0]['content']
            return SimpleNamespace(content=json.dumps({'candidates':[proposal(text)]}))
    if live:
        from database_settings import read_settings
        model=create_context_service(read_settings()).model
    else:model=FakeModel()
    reflection=ReflectionService(model,None)
    for name,text,old,added,suggested in inputs:
        started=time.perf_counter()
        result=reflection.extract([SourceMessage(1,'user',text)],{'enabled':True,'core_text':old,'extended_text':''},[])
        rows.append(dict(name=name,passed=len(result['additions'])==added and len(result['suggestions'])==suggested,
                         added=len(result['additions']),suggestions=len(result['suggestions']),duration_ms=round((time.perf_counter()-started)*1000,3),usage=result.get('usage')))
    context_calls=0
    class SummaryModel:
        def invoke(self,prompt):
            nonlocal context_calls
            context_calls+=1
            return SimpleNamespace(content=json.dumps({'items':[dict(kind='condition',text='合同期限两年',source_ordinals=[1],source_quotes=['合同期限两年'],supersedes=[])]}))
    context=ContextService(model if live else SummaryModel())
    messages=[SourceMessage(1,'user','合同期限两年'),SourceMessage(3,'user','学习背景'*1500)]+[SourceMessage(5+2*n,'user',f'近期问题{n}') for n in range(4)]
    result=context.build(Summary(),messages)
    rows.append(dict(name='early_condition',passed='合同期限两年' in '\n'.join(result['history']),old_four_turns_contains_condition=False,
                     estimated_tokens=result['estimated_tokens'],usage=result['usage'],duration_ms=result['duration_ms']))
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    return dict(mode='live' if live else 'deterministic_fake',timestamp=datetime.now(timezone.utc).isoformat(),revision=revision,working_tree='uncommitted',
                batch_sha256=hashlib.sha256(json.dumps(inputs,ensure_ascii=False).encode()).hexdigest(),budgets=vars(ContextLimits()),
                extraction_attempts=len(inputs),fake_extraction_calls=None if live else calls,fake_summary_calls=None if live else context_calls,
                rows=rows,passed=sum(row['passed'] for row in rows),total=len(rows),limitation='Synthetic acceptance only; not production accuracy, latency or cost savings.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');parser.add_argument('--allow-paid-calls',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.live and not args.allow_paid_calls:parser.error('Live mode requires explicit --allow-paid-calls authorization.')
    report=run(args.live);encoded=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded,encoding='utf-8')
    print(json.dumps({'mode':report['mode'],'passed':report['passed'],'total':report['total'],'network_calls':args.live},ensure_ascii=False))
    raise SystemExit(0 if report['passed']==report['total'] else 1)
