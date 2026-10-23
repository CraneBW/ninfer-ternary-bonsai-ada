"""Small real-model directed lifecycle proof, using the owned Serve harness."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from tools.bench.prefix_checkpoint import ROOT, PROFILE
from tools.bench.sm89_profiles import LocalServer, dump


def post(port,payload,cancel=False,cancel_after=1):
    payload=dict(payload,stream=True,stream_options={'include_usage':True})
    request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    result=dict(text='',done=False,usage=None,ids=[],finish=[],cancelled=False,frames=[])
    try:
        with urllib.request.urlopen(request,timeout=120) as response:
            result['http']=response.status
            for line in response:
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if data==b'[DONE]':result['done']=True;continue
                frame=json.loads(data);result['frames'].append(frame)
                if frame.get('id'):result['ids'].append(frame['id'])
                if frame.get('usage'):result['usage']=frame['usage']
                for choice in frame.get('choices',[]):
                    result['text']+=choice.get('delta',{}).get('content') or ''
                    if choice.get('finish_reason'):result['finish'].append(choice['finish_reason'])
                if cancel and len(result['text'])>=cancel_after:
                    result['cancelled']=True
                    break
    except urllib.error.HTTPError as error:
        result.update(http=error.code,error=error.read().decode())
    result['complete']=(result.get('http')==200 and result['done'] and result['usage'] is not None
                        and len(set(result['ids']))==1 and len(result['finish'])==1)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['original','concurrency'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in list(os.environ):
        if name.startswith(('NINFER_TEST_','NINFER_BENCH_')):del os.environ[name]
    os.environ.update(PROFILE,NINFER_SM89_SHARED_VECTOR='1',
        NINFER_KVMEM_ORIGINAL_POSITIONS='1' if a.mode=='original' else '0',
        NINFER_TEST_ORIGINAL_PAIR_TRACE='1' if a.mode=='original' else '0')
    command=[str(ROOT/'build/apps/ninfer-serve'),
        '/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer',
        '--model-id','qualification','--host','127.0.0.1','--port','18176',
        '--kv-dtype','int8','--prefill-chunk','128','--greedy','--no-thinking',
        '--request-log-jsonl',str(out/'requests.jsonl')]
    if a.mode=='original':
        command+=['--max-context','12288','--kv-device-tokens','8192','--host-kv-mib','256',
            '--max-concurrency','1','--kvmem','--kvmem-budget','4096',
            '--kvmem-gen-reserve','256','--kvmem-transfer-mode','sync',
            '--spec','mtp','--draft-tokens','3','--lm-head-draft']
        fixture=json.loads((ROOT/'tmp-nvcc/qualification-20261020/long-resident/32768-0.json').read_text())
        messages=fixture['messages']
        messages[1]['content']='\n'.join(messages[1]['content'].splitlines()[:550])
        base=dict(model='qualification',messages=messages,max_tokens=8,temperature=0)
    else:
        command+=['--max-context','8192','--kv-capacity','16384','--no-prefix-reuse',
                  '--max-concurrency','4','--max-pending-requests','4']
        base=dict(model='qualification',max_tokens=8,temperature=0,
            messages=[dict(role='system',content='Follow the last instruction exactly.')])
    def make(answer,length=0,tokens=8):
        payload=json.loads(json.dumps(base));payload['max_tokens']=tokens
        if a.mode=='original':payload['messages'][-1]['content']=f'Reply exactly {answer}.'
        else:payload['messages'].append(dict(role='user',content=('Background. '*length)+f'\nReply exactly {answer}.'))
        return payload
    report=dict(mode=a.mode,command=command,power_w=200)
    with LocalServer(command,'127.0.0.1',18176,out/'requests.jsonl') as server:
        report['startup']=server.wait_until_ready()
        if a.mode=='concurrency':
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(post,18176,make(word,100+150*i))
                         for i,word in enumerate(['ALPHA','BETA','GAMMA'])]
                cancel=make('COUNT',500,512)
                cancel['messages'][-1]['content']='Count from 1 through 1000, one number per line, without stopping.'
                future_cancel=pool.submit(post,18176,cancel,True)
                report['completed']=[future.result() for future in futures]
                report['cancel']=future_cancel.result()
            rejected=make('X',12000)
            report['capacity_reject']=post(18176,rejected)
        else:
            report['completed']=[post(18176,make('ACK'))]
            cancel=make('X',tokens=512)
            cancel['messages'][-1]['content']='Count from 1 through 1000, one number per line, without stopping.'
            report['cancel']=post(18176,cancel,True)
        # Follow cancellation with different fresh data in the same process.
        report['after_cancel']=post(18176,make('DELTA'))
        for _ in range(50):
            with urllib.request.urlopen('http://127.0.0.1:18176/metrics',timeout=3) as response:
                metrics=response.read().decode()
            if re.search(r'^ninfer:requests_active 0$',metrics,re.M):break
            time.sleep(.1)
        (out/'metrics.txt').write_text(metrics)
        report['drained']=all(re.search('^'+re.escape(name)+r' 0$',metrics,re.M)
            for name in ['ninfer:requests_active','ninfer:kvmem_inflight_pages',
                         'ninfer:kvmem_inflight_bytes'])
    report['server_exit']=server.process.returncode
    log=(out/'requests.console.log').read_text()
    report['commit_markers']=re.findall(r'STRATA_ORIGINAL_COMMIT[^\n]*',log)
    expected=['ACK'] if a.mode=='original' else ['ALPHA','BETA','GAMMA']
    report['pass']=(all(r['complete'] and r['text']==word for r,word in zip(report['completed'],expected))
        and report['cancel']['cancelled'] and report['after_cancel']['complete']
        and report['after_cancel']['text']=='DELTA' and report['server_exit']==0 and report['drained'])
    if a.mode=='original':
        report['pass']=report['pass'] and any('terminal=1' in m and 'repair=1' in m for m in report['commit_markers'])
    else:
        report['pass']=report['pass'] and 400<=report['capacity_reject']['http']<500
    dump(out/'report.json',report)
    print(json.dumps(dict(mode=a.mode,passed=report['pass'],exit=report['server_exit'],
                         commits=report['commit_markers'][-8:])),flush=True)
    return 0 if report['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
