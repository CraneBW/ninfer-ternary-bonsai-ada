"""Bounded native qualification using the existing Serve wire client.

Known-fact depths follow Companion tools/longctx_recall_probe.py. Unlike its
chars/token estimate, inputs here are sized with the real native tokenizer.
Each manifest is frozen before use; run stops on the first failed gate.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest
from tools.bench.kvmem_async import signature, write_json
from tools.bench.cuda_process import CudaProcessStillRunning, stop_owned_cuda_process


def payload(messages, limit=24):
    return {"model": "qualification", "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": limit,
            "temperature": 0}


def request_file(folder, name, body, expected, matcher="exact"):
    write_json(folder / name, body)
    return {"file": name, "sha256": hashlib.sha256((folder/name).read_bytes()).hexdigest(),
            "expected": expected, "matcher": matcher}


def command(binary, model, context, physical, host=256):
    return [str(binary.resolve()), str(model.resolve()), "--model-id", "qualification",
            "--host", "127.0.0.1", "--port", "18118", "--managed-weights",
            "--max-context", str(context), "--kv-device-tokens", str(physical),
            "--prefill-chunk", "512", "--host-kv-mib", str(host),
            "--max-concurrency", "1", "--kv-dtype", "int8", "--greedy",
            "--no-thinking", "--no-cuda-graph", "--log-level", "info"]


def audit(executable, body):
    result = subprocess.run([str(executable)], input=json.dumps(body), text=True,
                            capture_output=True, check=True)
    return json.loads(result.stdout)


def freeze_long(folder, binary, model, tokenizer, working=32768):
    if working < 256 or working > 32768 or working % 64:
        raise ValueError("working window must be a 64-token multiple in [256,32768]")
    folder.mkdir(parents=True, exist_ok=False)
    facts = [("Topaz Ledger", "silver-maple-41"), ("Indigo Archive", "amber-willow-92"),
             ("Saffron Register", "copper-lotus-35")]
    stages = []
    for target, host in [(32768,2048),(65536,3072),(212992,8192),(262144,8192)]:
        # Reserve the suffix for all fixed question/answer turns.
        desired = target - 768
        count = int(desired / 14)
        for _ in range(4):
            lines = [f"Note {i}: the record is unchanged and archived." for i in range(count)]
            for depth,(subject,label) in zip((.033,.367,.700), facts):
                lines[int(count*depth)] = f"The {subject} display label is {label}."
            lines[int(count*.20)] = "The Quartz Parcel display label is cobalt-orchid-73."
            lines[int(count*.967)] = "Update: The Quartz Parcel display label is copper-lilac-28. This replaces the earlier label."
            messages = [{"role":"system","content":"Follow the user instruction. Be concise."},
                        {"role":"user","content":"Read these notes.\n"+"\n".join(lines)},
                        {"role":"assistant","content":None,"tool_calls":[{"id":"ledger_1","type":"function","function":{"name":"read_ledger","arguments":"{}"}}]},
                        {"role":"tool","tool_call_id":"ledger_1","content":"The Jade Cabinet display label is teal-jasmine-61."},
                        {"role":"user","content":"Reply exactly ACK."}]
            measured = audit(tokenizer, payload(messages))["token_count"]
            if abs(measured-desired) < 64: break
            count += round((desired-measured) / (measured/count))
        if abs(measured-desired) >= 128: raise RuntimeError("native token sizing did not converge")
        entries=[];audits=[]
        entries.append(request_file(folder,f"{target}-0.json",payload(messages),"ACK"))
        messages.append({"role":"assistant","content":"ACK"})
        questions=[(f"What is the exact {s} display label? Reply only with the label.",v) for s,v in facts]
        questions += [("What is the current display label of the Quartz Parcel? Reply only with the label.","copper-lilac-28"),
                      ("What is the display label of the Jade Cabinet in the tool result? Reply only with the label.","teal-jasmine-61")]
        for i,(question,expected) in enumerate(questions,1):
            messages.append({"role":"user","content":question});body=payload(messages)
            view=audit(tokenizer,{**body,"needle":expected});audits.append(view)
            if view["token_count"]+24 > target: raise RuntimeError("frozen context ceiling exceeded")
            entries.append(request_file(folder,f"{target}-{i}.json",body,expected))
            messages.append({"role":"assistant","content":expected})
        write_json(folder/f"{target}-token-audits.json",audits)
        arms=["full","sync","async"] if target==32768 else ["sync","async"]
        for mode in arms:
            args=command(binary,model,target,target if mode=="full" else min(target,working+4096),256 if mode=="full" else host)
            args.remove("--managed-weights") # 10.9GiB GSQ text fits; avoid managed runtime cap
            if mode!="full": args += ["--kvmem","--kvmem-budget",str(working),"--kvmem-gen-reserve","512","--kvmem-action-lrfu","--kvmem-lrfu-lambda","0.5","--kvmem-lrfu-top-frac","0.3","--kvmem-transfer-mode",mode]
            stages.append({"id":f"{target}-{mode}","args":args,"requests":entries,"context":target,
                           "actual_first_tokens":measured,"mode":mode})
    # Force the existing Cold tier separately, after the ordinary length ladder.
    source=next(s for s in stages if s['id']=='65536-sync')
    for mode in ['sync','async']:
        args=list(source['args']);args[args.index('--host-kv-mib')+1]='2048';args[args.index('--kvmem-transfer-mode')+1]=mode
        stages.append({**source,'id':f'cold-65536-{mode}','args':args,'mode':mode,'require_cold':True,'env':{'NINFER_KVMEM_HOST_PAGES':'16','NINFER_KVMEM_COLD_DIR':str(folder.resolve())}})
    write_json(folder/'manifest.json',{'binary':signature(binary),'model':signature(model),'stages':stages,
               'request_timeout_s':1200,'startup_timeout_s':120,'max_calls':66,
               'gate':'every expected fact, complete terminal, no in-flight pages; earlier stage failure blocks dependents',
               'no_retry':True,'no_full_kv_equivalence_above_32k':True})
    print('FROZEN long:',len(stages),'stages, cap 66 calls',flush=True)


def metric_values(text):
    found={}
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():continue
        fields=line.split()
        if len(fields)==2:
            try:found[fields[0]]=float(fields[1])
            except ValueError:pass
    return found


def correct(text, entry):
    if entry.get('matcher')=='colors':
        return re.findall(r'[a-z]+',text.lower())==entry['expected']
    return text.strip()==entry['expected']


def run(folder):
    manifest=json.loads((folder/'manifest.json').read_text());ledger=folder/'results.json'
    if ledger.exists():raise RuntimeError('Existing ledger: no automatic rerun')
    for key in ['model','binary']:
        if key in manifest and signature(Path(manifest[key]['path']))!=manifest[key]:raise RuntimeError(f'{key} changed')
    results=[];sent=0
    for stage in manifest['stages']:
        for field in ['model_signature','binary_signature']:
            if field in stage and signature(Path(stage[field]['path'])) != stage[field]:raise RuntimeError(f'{field} changed')
        name=stage['id'];record={'id':name,'status':'STARTING','requests':[]};results.append(record)
        write_json(ledger,{'sent':sent,'stages':results});args=stage['args'];write_json(folder/f'{name}-command.json',args)
        exchanges=[];print('START',name,flush=True)
        with (folder/f'{name}.log').open('w') as log:
            env=os.environ.copy();env.pop('NINFER_KVMEM_TRACE',None)
            if manifest.get('trace'):env['NINFER_KVMEM_TRACE']='1'
            env.update(stage.get('env',{}))
            proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                client=NInferServeClient('http://127.0.0.1:18118',manifest.get('request_timeout_s',120))
                end=time.monotonic()+manifest.get('startup_timeout_s',90)
                while True:
                    if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}')
                    try:client.discover_model();break
                    except Exception:
                        if time.monotonic()>end:raise
                        time.sleep(.25)
                for entry in stage['requests']:
                    raw=(folder/entry['file']).read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise RuntimeError('frozen request changed')
                    if sent>=manifest['max_calls']:raise RuntimeError('frozen call cap exceeded')
                    sent+=1;write_json(ledger,{'sent':sent,'stages':results})
                    x=client.prepare(ProtocolRequest('openai_chat','/v1/chat/completions',json.loads(raw))).execute();exchanges.append(x)
                    text=''.join(e.output for e in x.events if e.kind=='model_output')
                    terminal=any(e.kind=='terminal' for e in x.events)
                    usage=next((e.payload['usage'] for e in reversed(x.events) if e.payload and e.payload.get('usage')),None)
                    ok=correct(text,entry)
                    record['requests'].append({'file':entry['file'],'text':text,'correct':ok,'terminal':terminal,'usage':usage,'status':x.http.status})
                    print(name,entry['file'],repr(text),ok,'terminal',terminal,flush=True)
                    write_json(ledger,{'sent':sent,'stages':results})
                    if x.http.status!=200 or x.http.error or x.protocol_error or x.error_code or not terminal or not usage:raise RuntimeError('request has no valid complete terminal')
                    if not ok:raise RuntimeError('frozen quality gate failed; stop dependent stages')
                with urllib.request.urlopen('http://127.0.0.1:18118/metrics',timeout=5) as response:metrics=response.read().decode()
                (folder/f'{name}.metrics.txt').write_text(metrics);record['metrics']=metric_values(metrics)
                for key,value in record['metrics'].items():
                    if ('kvmem_inflight_pages' in key or 'kvmem_inflight_bytes' in key) and value!=0:raise RuntimeError('terminal transfer still in flight')
                if stage.get('require_cold') and not any('kvmem_host_to_cold_spills_total' in k and v>0 for k,v in record['metrics'].items()):raise RuntimeError('cold stage did not actually spill')
                record['status']='PASS'
            except BaseException as error:
                record.update(status='FAIL',error=repr(error));raise
            finally:
                shutdown_error = None
                try:
                    stop_owned_cuda_process(proc)
                except CudaProcessStillRunning as error:
                    record.update(status='FAIL', shutdown_error=str(error))
                    shutdown_error = error
                record['server_exit']=proc.returncode
                for i,x in enumerate(exchanges):
                    (folder/f'{name}-{i}.sse').write_bytes(x.http.body);receipt=dataclasses.asdict(x);receipt['http'].pop('body');write_json(folder/f'{name}-{i}.receipt.json',receipt)
                write_json(ledger,{'sent':sent,'stages':results})
                print('END',name,record['status'],'server exit',proc.returncode,flush=True)
                if shutdown_error is not None:raise shutdown_error
    print('ALL_PASS',sent,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['freeze-long','run']);p.add_argument('folder',type=Path);p.add_argument('--model',type=Path);p.add_argument('--binary',type=Path);p.add_argument('--tokenizer-audit',type=Path);p.add_argument('--working-tokens',type=int,default=32768);a=p.parse_args()
    if a.action=='run':run(a.folder)
    else:
        if not all([a.model,a.binary,a.tokenizer_audit]):p.error('freeze-long requires explicit model/binary/native tokenizer audit')
        freeze_long(a.folder,a.binary,a.model,a.tokenizer_audit,a.working_tokens)

if __name__=='__main__':main()
