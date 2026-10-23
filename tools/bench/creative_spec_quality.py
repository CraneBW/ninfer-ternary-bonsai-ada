"""Full creative outputs under identical GSQ/greedy settings; no quality from acceptance.
Reuses the local bounded Serve/SSE lifecycle and donor fixed-K comparison method.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from tools.bench.sm89_profiles import LocalServer, stream_request, dump, ROOT
from tools.bench.prefix_checkpoint import PROFILE

CASES = [
    dict(id='pelican', source='https://simonwillison.net/2024/Oct/25/pelicans-on-a-bicycle/',
         prompt='Generate an SVG of a pelican riding a bicycle'),
    dict(id='emperor', source='https://linux.do/t/topic/2758684',
         prompt='创建一个html，使用SVG绘制秦始皇骑北极熊的2D动画'),
]
# Emperor uses the original core line, not the later extra three design directives.
PROFILES = {'ar': ('none',0), 'mtp4': ('mtp',4), 'mtp7': ('mtp',7),
            'mtp15': ('mtp',15), 'dflash2': ('dflash2',3)}

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES))
    ap.add_argument('--max-tokens',type=int,default=7168)
    ap.add_argument('--port',type=int,default=18163)
    ap.add_argument('--direct',action='store_true',help='Controlled system instruction; preserve original user prompt')
    ap.add_argument('--cases',nargs='+',choices=['pelican','emperor'],default=['pelican','emperor'])
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    power=float(subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).strip())
    assert power==200, power
    artifact=Path('/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer')
    env=dict(PROFILE,NINFER_R29_BUNDLE='0',NINFER_GGUF_LARGE_PREFILL='1',
             NINFER_GGUF_SILU_MMQ='1',NINFER_KVMEM_LAYER_READY='0',
             NINFER_SM89_SHARED_VECTOR='0',NINFER_SM89_ATTN_REDUCE_VECTOR='0',
             NINFER_SM89_GDN_PRENORM='0')
    for key in list(os.environ):
        if key.startswith('NINFER_') and key not in env: del os.environ[key]
    os.environ.update(env)
    result=dict(state='RUNNING',power_w=power,model=str(artifact),model_bytes=artifact.stat().st_size,
                cases=[c for c in CASES if c['id'] in args.cases],direct=args.direct,environment=env,profiles=[],scope='greedy no-thinking C1; existing mixed GSQ only; creative sample, not a broad quality benchmark')
    binary=ROOT/'build/apps/ninfer-serve'
    result['binary']={'path':str(binary),'size':binary.stat().st_size,'mtime_ns':binary.stat().st_mtime_ns}
    for name in args.profiles:
        backend,width=PROFILES[name];log=out/f'{name}.jsonl'
        cmd=[str(ROOT/'build/apps/ninfer-serve'),str(artifact),'--model-id','qualification',
             '--host','127.0.0.1','--port',str(args.port),'--max-context','8192',
             '--kv-capacity','8192','--prefill-chunk','1024','--max-concurrency','1',
             '--kv-dtype','int8','--greedy','--no-thinking','--presence-penalty','0',
             '--no-prefix-reuse','--request-log-jsonl',str(log),'--log-stats-interval-ms','1000']
        if backend!='none':cmd+=['--spec',backend,'--draft-tokens',str(width),'--lm-head-draft']
        row=dict(name=name,command=cmd,requests=[]);result['profiles'].append(row)
        try:
            with LocalServer(cmd,'127.0.0.1',args.port,log) as server:
                start=server.wait_until_ready();row['startup']=start
                free=start['memory']['available_after_startup_bytes'];row['free_bytes']=free
                if free<256*1024**2:
                    row['state']='CAPACITY_REJECTED'
                else:
                    for case in result['cases']:
                        messages=[dict(role='user',content=case['prompt'])]
                        if args.direct:
                            messages.insert(0,dict(role='system',content='You have no tools, files, or project workspace. Answer directly with complete self-contained code in one code block. Do not call tools or describe future steps.'))
                        payload=dict(model='qualification',messages=messages,
                                     temperature=0,seed=42,max_tokens=args.max_tokens)
                        client=stream_request(args.port,payload)
                        event=server.wait_for_request_done(start['server_instance_id'])
                        choice=client['response']['choices'][0];text=choice['message']['content']
                        (out/f"{name}-{case['id']}.txt").write_text(text)
                        blocks=re.findall(r'```(?:html|svg|xml)?\s*\n(.*?)```',text,re.S|re.I)
                        code=max(blocks,key=len) if blocks else text
                        ext='svg' if case['id']=='pelican' else 'html'
                        if ext=='svg':
                            match=re.search(r'<svg\b.*?</svg>',code,re.S)
                            code=match[0] if match else code
                        (out/f"{name}-{case['id']}.{ext}").write_text(code)
                        parse=None
                        if ext=='svg':
                            try:ET.fromstring(code);parse=True
                            except ET.ParseError:parse=False
                        req=dict(case=case['id'],response=client['response'],event=event,
                                 client_ttft_s=client['client_ttft_s'],client_wall_s=client['client_wall_s'],
                                 sha256=hashlib.sha256(text.encode()).hexdigest(),svg_xml_valid=parse,
                                 complete=choice['finish_reason']=='stop',visual_quality='NOT_REVIEWED',payload=payload)
                        row['requests'].append(req);dump(out/'results.json',result)
                        print(name,case['id'],choice['finish_reason'],client['response']['usage'],flush=True)
                    row['state']='GENERATED'
            row['server_exit']=server.process.returncode
        except Exception as e:
            row['state']='FAILED';row['error']=repr(e)
            dump(out/'results.json',result)
            raise
        dump(out/'results.json',result)
    result['state']='COLLECTED_NOT_QUALITY_VERDICT';dump(out/'results.json',result)

if __name__=='__main__':main()
