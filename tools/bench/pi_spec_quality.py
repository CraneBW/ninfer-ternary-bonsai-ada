"""Run the installed local Pi coding harness against native Ninfer profiles.
Uses isolated benchmark configuration, standard Pi system prompt/tools and fresh task directories.
"""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from tools.bench.creative_spec_quality import CASES,PROFILES
from tools.bench.sm89_profiles import LocalServer,dump,ROOT
from tools.bench.prefix_checkpoint import PROFILE

def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def pi_completion(path, artifact):
    events = read_events(path)
    messages = [e['message'] for e in events if e.get('type') == 'message_end'
                and e.get('message', {}).get('role') == 'assistant']
    errors = [m.get('errorMessage', m.get('stopReason')) for m in messages
              if m.get('stopReason') in ('error', 'aborted') or m.get('errorMessage')]
    stop = messages[-1].get('stopReason') if messages else None
    finished = any(e.get('type') == 'agent_end' for e in events)
    return dict(stop_reason=stop, errors=errors, agent_end=finished,
                artifact_present=artifact.is_file() and artifact.stat().st_size > 0,
                complete=finished and stop == 'stop' and not errors
                         and artifact.is_file() and artifact.stat().st_size > 0)

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profiles',nargs='+',choices=PROFILES,default=['mtp4','mtp7','ar'])
    ap.add_argument('--thinking',choices=['off','xhigh'],default='xhigh')
    ap.add_argument('--context',type=int,default=262144)
    ap.add_argument('--max-tokens',type=int,default=131072)
    ap.add_argument('--timeout',type=int,default=7200)
    ap.add_argument('--cases',nargs='+',choices=['pelican','emperor'],default=['pelican','emperor'])
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    pi=shutil.which('pi');assert pi,'local pi missing; pass the shell PATH into systemd-run'
    assert float(subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).strip())==200
    cfg=out/'pi-agent';cfg.mkdir()
    dump(cfg/'models.json',{'providers':{'ninfer-local':{'baseUrl':'http://127.0.0.1:18164/v1','api':'openai-completions','apiKey':'local-benchmark','models':[{'id':'qualification','name':'Qwen3.8 GSQ native','reasoning':args.thinking!='off','thinkingLevelMap':{'off':None,'minimal':None,'low':'low','medium':'medium','high':None,'xhigh':'xhigh','max':None},'input':['text'],'contextWindow':args.context,'maxTokens':args.max_tokens,'cost':{'input':0,'output':0,'cacheRead':0,'cacheWrite':0},'compat':{'supportsReasoningEffort':True,'supportsDeveloperRole':False,'maxTokensField':'max_tokens','supportsStore':False}}]}}})
    dump(cfg/'settings.json',{'compaction':{'enabled':False},'retry':{'enabled':False},'defaultThinkingLevel':args.thinking})
    env=dict(os.environ,**PROFILE)
    for key in list(env):
        if key.startswith('NINFER_') and key not in PROFILE:del env[key]
    env.update(NINFER_R29_BUNDLE='0',NINFER_GGUF_LARGE_PREFILL='1',NINFER_GGUF_SILU_MMQ='1',
               NINFER_SM89_SHARED_VECTOR='0',NINFER_SM89_ATTN_SPLIT_VECTOR='0',NINFER_SM89_GDN_PRENORM='0',
               PI_CODING_AGENT_DIR=str(cfg),PI_OFFLINE='1',PI_TELEMETRY='0')
    os.environ.update({k:v for k,v in env.items() if k.startswith('NINFER_')})
    result=dict(state='RUNNING',pi=pi,pi_version=subprocess.check_output([pi,'--version'],env=env,text=True).strip(),
                power_w=200,context=args.context,max_tokens=args.max_tokens,thinking=args.thinking,kv_working_set=32768,physical_kv=36864,tools=['read','write','edit','ls'],profiles=[],cases=CASES,
                harness='Installed Pi; standard system prompt; external extensions/skills/context disabled equally; isolated config; no global settings changed')
    work=out/'workspace';work.mkdir()
    result['working_directory']=str(work)
    dump(out/'results.json',result)
    for name in args.profiles:
        backend,width=PROFILES[name];log=out/f'{name}.jsonl'
        cmd=[str(ROOT/'build/apps/ninfer-serve'),'/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer','--model-id','qualification','--host','127.0.0.1','--port','18164','--max-context',str(args.context),'--kv-device-tokens','36864','--kvmem','--kvmem-budget','32768','--kvmem-gen-reserve','512','--kvmem-transfer-mode','sync','--host-kv-mib','128','--prefill-chunk','1024','--max-concurrency','1','--kv-dtype','int8','--greedy','--preserve-thinking','--default-max-tokens',str(args.max_tokens),'--seed','42','--presence-penalty','0','--request-log-jsonl',str(log)]
        if args.thinking=='off':cmd+=['--no-thinking']
        if name=='dflash2':
            result['profiles'].append(dict(name=name,state='CAPACITY_REJECTED_STATIC',reason='32K working set / 36K physical KV does not meet the DFlash2 256MiB reserve gate; short-mode receipt is separate'))
            dump(out/'results.json',result);continue
        if backend!='none':cmd+=['--spec',backend,'--draft-tokens',str(width),'--lm-head-draft']
        row=dict(name=name,command=cmd,tasks=[]);result['profiles'].append(row)
        dump(out/'results.json',result)
        with LocalServer(cmd,'127.0.0.1',18164,log) as server:
            start=server.wait_until_ready();row['startup']=start
            dump(out/'results.json',result)
            if start['memory']['available_after_startup_bytes']<256*1024**2:row['state']='CAPACITY_REJECTED'
            else:
                for c in CASES:
                    if c['id'] not in args.cases:continue
                    wd=out/name/c['id'];wd.mkdir(parents=True)
                    assert not list(work.iterdir()), 'benchmark workspace must start empty'
                    # Same task framing for all profiles, original source sentence unchanged.
                    prompt=c['prompt']+'\nSave the complete result in this working directory as '+('result.svg' if c['id']=='pelican' else 'index.html')+'. Use self-contained SVG/HTML with no external dependencies.'
                    command=[pi,'--offline','--no-extensions','--no-skills','--no-context-files','--no-prompt-templates','--provider','ninfer-local','--model','qualification','--thinking',args.thinking,'--tools','read,write,edit,ls','--mode','json','--no-session','-p',prompt]
                    task=dict(case=c['id'],command=command,cwd=str(work),state='RUNNING');row['tasks'].append(task);dump(out/'results.json',result)
                    begin=time.monotonic()
                    event_start=len(read_events(log))
                    with (wd/'pi.jsonl').open('w') as fout,(wd/'pi.stderr').open('w') as ferr:
                        proc=subprocess.Popen(command,cwd=work,env=env,stdout=fout,stderr=ferr)
                        try:code=proc.wait(timeout=args.timeout)
                        except subprocess.TimeoutExpired:
                            proc.terminate();code=proc.wait(timeout=40);task['timeout']=True
                    for produced in list(work.iterdir()):
                        shutil.move(str(produced),str(wd/produced.name))
                    task.update(exit_code=code,wall_s=time.monotonic()-begin,files=[str(p.relative_to(wd)) for p in wd.rglob('*') if p.is_file() and p.name not in ('pi.jsonl','pi.stderr')])
                    task['pi_completion']=pi_completion(wd/'pi.jsonl',wd/('result.svg' if c['id']=='pelican' else 'index.html'))
                    events=read_events(log)[event_start:]
                    task['request_events']=[e for e in events if e.get('event') in ('request_done','request_error')]
                    errors=[e for e in events if e.get('event')=='request_error']
                    task['state']='COMPLETED_NOT_QUALITY_VERDICT' if code==0 and not errors and task['pi_completion']['complete'] and not task.get('timeout') else 'FAILED'
                    for e in events:
                        if e.get('event')!='request_done':continue
                        request=e['request']
                        assert request['enable_thinking']==(args.thinking!='off'), 'thinking mode mismatch'
                        if args.thinking=='xhigh':
                            assert request['requested_reasoning_effort']=='xhigh' and request['resolved_reasoning_effort']=='xhigh', 'xhigh was not transmitted/resolved'
                        assert request['sampling']['temperature']==0, 'greedy comparison changed'
                    dump(out/'results.json',result)
                    print(name,c['id'],code,task['files'],round(task['wall_s'],2),flush=True)
                    if task['state']=='FAILED':
                        row['state']=result['state']='FAILED'
                        dump(out/'results.json',result)
                        raise RuntimeError(f"Pi task failed despite possible zero exit: {task['pi_completion']}; server errors={errors}")
                row['state']='COLLECTED'
        row['server_exit']=server.process.returncode;dump(out/'results.json',result)
    result['state']='COLLECTED_NOT_QUALITY_VERDICT';dump(out/'results.json',result)

if __name__=='__main__':main()
