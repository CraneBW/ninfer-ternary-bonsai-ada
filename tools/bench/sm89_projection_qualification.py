"""200W qualification of reused R27 projections and shared prefill activation.

Uses the existing independent full-row NumPy/FP64 oracle for the frozen GSQ.
Only the newly eligible shapes are selected. Matrix tests compare both complete
destinations (one strided, guarded) with the original product in the same process.
Decode compares complete random outputs between the old and new dispatch.
No model loading, creative quality, kernel timing or performance claims here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path
from tools.bench.prefix_checkpoint import PROFILE, ROOT, WORK
from tools.bench.sm89_profiles import dump
from tools.bench.cuda_process import stop_owned_cuda_process

FORMATS=('gguf_iq3_s','gguf_iq3_xxs','gguf_iq4_xs')
ROWS=(1024,2048,6144,10240,16384,34816)

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--oracle',type=Path,default=ROOT/'tmp-nvcc/opsx-cold-prefill-goal-20261023/all-linear-oracle.json')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert float(subprocess.check_output(['nvidia-smi','--query-gpu=power.limit',
        '--format=csv,noheader,nounits'],text=True).strip())==200
    frozen=json.loads((WORK/'benchmarks/frozen-long-context/213k/manifest.json').read_text())
    model=Path(frozen['model']['path']);stat=model.stat()
    assert {'path':str(model),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}==frozen['model']
    source=json.loads(args.oracle.read_text());derived={**source,'formats':{}}
    for name,value in source['formats'].items():
        if value['format'] in FORMATS and value['k']==5120:
            derived['formats'][name]=value
    for fmt in FORMATS:
        parent=next(v for v in derived['formats'].values() if v['format']==fmt and v['rows']==16384)
        derived['formats'][fmt+'_slice2048_k5120']={**parent,'rows':2048,'row_sums':parent['row_sums'][:2048]}
    oracle=out/'selected-oracle.json';dump(oracle,derived)
    env={k:v for k,v in os.environ.items() if not k.startswith('NINFER_')}
    env.update(PROFILE,NINFER_R29_BUNDLE='0',NINFER_R28_BUNDLE='0',
        NINFER_GGUF_LARGE_PREFILL='1',NINFER_GGUF_SILU_MMQ='1',
        NINFER_SM89_R27_PROJECTIONS_TRACE='1',NINFER_SM89_SHARED_MATRIX_TRACE='1')
    binary=ROOT/'build/tests/ninfer_gguf_linear_test'
    result=dict(state='RUNNING',power_w=200,oracle_source=str(args.oracle),
        oracle_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        scope=__doc__,processes=[],cases=[],compared_bytes=0)
    def save():dump(out/'qualification.json',result)
    def run(name,entry,tokens,pattern,flag,shared=False,capture=None):
        command=[str(binary),str(model),str(oracle),'--entry',entry,'--tokens',str(tokens),'--pattern',pattern]
        if shared:command+=['--shared-matrix']
        if capture:command+=['--dump-output',str(capture)]
        process_env=env|dict(NINFER_SM89_R27_PROJECTIONS=str(flag),NINFER_SM89_SHARED_MATRIX=str(int(shared)))
        row=dict(name=name,command=command,state='RUNNING');result['processes'].append(row);save()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(command,env=process_env,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=120)
            except BaseException:
                stop_owned_cuda_process(proc);raise
        row.update(state='PASS' if code==0 else 'FAILED',exit_code=code);save()
        assert code==0, name
        text=(out/(name+'.log')).read_text()
        if shared:assert 'GGUF_SHARED_MATRIX' in text and 'full_bytes=1 guards=1 PASS' in text
        elif flag:assert 'SM89_R27_PROJECTION' in text
        if pattern=='constant':assert 'real GSQ ' in text and f'entry={entry} ' in text and 'PASS' in text
    save()
    try:
        for entry,value in derived['formats'].items():
            if value['rows'] not in ROWS:continue
            for pattern in ('constant','random'):
                captures=[]
                for flag in (0,1):
                    name=f'decode-{entry}-{pattern}-{flag}';capture=out/(name+'.bf16')
                    run(name,entry,8,pattern,flag,capture=capture);captures.append(capture)
                a,b=(p.read_bytes() for p in captures)
                assert len(a)==value['rows']*8*2 and a==b,entry
                result['compared_bytes']+=len(a)
                result['cases'].append(dict(kind='decode',entry=entry,pattern=pattern,bytes=len(a),exact=True));save()
        for fmt in FORMATS:
            for n in (1024,2048,6144,17408):
                entry=next(k for k,v in derived['formats'].items() if v['format']==fmt and v['rows']==n)
                for tokens in (512,1024):
                    for pattern in ('constant','random'):
                        name=f'prefill-{entry}-{tokens}-{pattern}'
                        run(name,entry,tokens,pattern,1,shared=True)
                        result['cases'].append(dict(kind='prefill',entry=entry,tokens=tokens,pattern=pattern,
                            full_destinations_equal=True,strided_guards=True));save()
        result['state']='PASS';save()
        print(json.dumps(dict(state='PASS',processes=len(result['processes']),cases=len(result['cases']),
                              decode_compared_bytes=result['compared_bytes'])),flush=True)
    except BaseException as error:
        result.update(state='FAILED',error=str(error));save();raise

if __name__=='__main__':main()
