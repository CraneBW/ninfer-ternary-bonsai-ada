#!/usr/bin/env python3
"""Qualify the existing BF16-rounded SiLU/down fusion at T256, T512 or T1024.

Reuses R29 represented-weight FP64 row sums and full quant/projected/residual
captures. Supply the independent oracle for the fixed artifact explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from tools.bench.cuda_process import stop_owned_cuda_process,CudaProcessStillRunning
MODEL=Path('/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer')
CASES=(('iq3_s',21),('iq3_xxs',18),('iq4_xs',23),('iq2_xs',17),('iq2_s',22),('q2_k',10),('q4_k',12))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def sig(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=ROOT/'build/tests/ninfer_gguf_silu_down_test')
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--tokens',type=int,choices=(256,512,1024),default=1024)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=float,default=120)
    a=p.parse_args();a.binary=a.binary.resolve();a.oracle=a.oracle.resolve();a.output_dir=a.output_dir.resolve()
    OUT_BYTES=5120*a.tokens*2
    power=subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(power)!=1 or float(power[0])!=200:raise ValueError('one 200W GPU required')
    if a.timeout_seconds<=0:raise ValueError('positive timeout required')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    watched=(a.binary,a.oracle,MODEL);frozen={str(x):sig(x) for x in watched}
    result=dict(status='running',power_w=200,tokens=a.tokens,binary=dict(path=str(a.binary),sha256=sha(a.binary)),
        oracle=dict(path=str(a.oracle),sha256=sha(a.oracle)),input_signatures=frozen,
        fixed_flags={'NINFER_R28_BUNDLE':'0','NINFER_R27_BUNDLE':'1','NINFER_R26_BUNDLE':'0','NINFER_GGUF_LARGE_PREFILL':'1','NINFER_R29_BUNDLE':'0','NINFER_GGUF_SILU_MMQ_T256':'1'},
        cases=[],processes=[],performance_runs=0,engine_calls=0,random_independent_oracle=False)
    def save():(a.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for suffix,type_id in CASES:
            for pattern in ('constant','random'):
                captures=[]
                for flag in ('0','1'):
                    if any(sig(x)!=frozen[str(x)] for x in watched):raise RuntimeError('frozen fusion input changed')
                    label=f'{suffix}-{pattern}-bundle{flag}'
                    capture=a.output_dir/(label+'.bf16');logpath=a.output_dir/(label+'.log')
                    command=[str(a.binary),str(MODEL),str(a.oracle),'--entry',f'gguf_{suffix}_n5120_k17408',
                             '--tokens',str(a.tokens),'--pattern',pattern,'--dump-output',str(capture)]
                    env=dict(os.environ,NINFER_R29_BUNDLE='0',NINFER_R29_BUNDLE_TRACE='0',
                        NINFER_GGUF_SILU_MMQ=flag,NINFER_GGUF_SILU_MMQ_T256='1',NINFER_GGUF_SILU_MMQ_TRACE='1',NINFER_GGUF_LARGE_PREFILL='1',
                        NINFER_R28_BUNDLE='0',NINFER_R28_BUNDLE_TRACE='0',NINFER_R27_BUNDLE='1',
                        NINFER_R27_BUNDLE_TRACE='0',NINFER_R26_BUNDLE='0',NINFER_R26_BUNDLE_TRACE='0',
                        NINFER_GGUF_MMQ_BF16='1',NINFER_GGUF_IQ2_S_MMQ_BF16='1',
                        NINFER_GGUF_IQ3_S_MMQ_I64='1',NINFER_GGUF_IQ3_XXS_MMQ_I64='1',NINFER_GGUF_MMQ_TRACE='0')
                    row=dict(label=label,bundle=flag,pattern=pattern,command=command,state='starting')
                    result['processes'].append(row);save();started=time.monotonic()
                    with logpath.open('x') as log:
                        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        row.update(pid=proc.pid,state='live');save()
                        try:code=proc.wait(timeout=a.timeout_seconds)
                        except BaseException as error:
                            row.update(state='graceful_stop_requested',error=repr(error));save()
                            try:code=stop_owned_cuda_process(proc,grace_seconds=120)
                            except CudaProcessStillRunning as live:
                                row.update(state='cuda_child_still_live',error=str(live),gpu_available_for_reuse=False);save();raise
                            row.update(state='stopped',returncode=code);save();raise
                    row.update(state='pass' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-started);save()
                    if code:raise RuntimeError('fusion actual product failed: '+label)
                    text=logpath.read_text()
                    if f'R29_FUSION_BYTES policy={flag} type={type_id} N=5120 K=17408 T={a.tokens} ' not in text:raise RuntimeError('missing full-byte fixture receipt')
                    marker=f'[r29-silu-down-quant] type={type_id} K=17408 T={a.tokens} '
                    if flag=='1' and marker not in text:raise RuntimeError('missing actual fused quant checked launch')
                    if flag=='0' and '[r29-silu-down-quant]' in text:raise RuntimeError('baseline selected fused kernel')
                    if pattern=='constant' and f'entry=gguf_{suffix}_n5120_k17408 ' not in text:raise RuntimeError('independent CPU oracle missing')
                    if pattern=='constant' and ' PASS\n' not in text:raise RuntimeError('constant oracle failed')
                    captures.append([capture,Path(str(capture)+'.q8'),Path(str(capture)+'.residual.bf16')])
                detail=dict(format=suffix,pattern=pattern,comparisons=[])
                for index,name in enumerate(('projected_bf16','complete_q8_codes_scales_sums_with_guards','residual_bf16')):
                    files=[paths[index] for paths in captures];count=files[0].stat().st_size
                    if files[1].stat().st_size!=count or (index!=1 and count!=OUT_BYTES) or (index==1 and count<=17408*a.tokens):raise RuntimeError('incomplete full capture')
                    with files[0].open('rb') as f0,files[1].open('rb') as f1:
                        while b:=f0.read(1<<20):
                            if b!=f1.read(len(b)):raise RuntimeError(f'{suffix} {pattern} {name} bytes differ')
                    detail['comparisons'].append(dict(name=name,bytes=count,equal=True,baseline_sha256=sha(files[0]),candidate_sha256=sha(files[1])))
                result['cases'].append(detail);save()
        if any(sig(x)!=frozen[str(x)] for x in watched):raise RuntimeError('fusion input changed during proof')
        result.update(status='passed_independent_constant_quant_fullbytes_projected_residual_routes_not_performance',
            independent_constant_actual_product_cases=14,random_fullbytes_actual_product_cases=14,
            complete_compared_bytes=sum(v['bytes'] for c in result['cases'] for v in c['comparisons']),
            public_fixture_processes=28)
        save();print(json.dumps(dict(status=result['status'],summary=str(a.output_dir/'summary.json'))),flush=True)
    except BaseException as error:
        result.update(status=('cuda_child_still_running_no_gpu_reuse' if isinstance(error,CudaProcessStillRunning) else 'failed'),error=str(error));save();raise

if __name__=='__main__':main()
