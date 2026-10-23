"""Locate first numerical state divergence using the unchanged R21 input.

Each arm stops just after its first real compaction and next prefill chunk.
This is a diagnostic, never a full 213K quality result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import numpy as np
from tools.bench.prefix_checkpoint import ROOT, WORK, PROFILE
from tools.bench.sm89_profiles import LocalServer, dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    # First known wrong R21 answer is silver-maple-41, not the opening ACK.
    source=json.loads((WORK/'benchmarks/frozen-long-context/213k/manifest.json').read_text())['requests'][1]
    raw=Path(source['file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source['sha256']
    prior=ROOT/'tmp-nvcc/opsx-original-causal/no_probe/no_probe-command.json'
    report=dict(input=source,scope='first compaction only; not quality acceptance',arms=[])
    for original in (0,1):
        arm=out/str(original);arm.mkdir();(arm/'backing').mkdir()
        for name in list(os.environ):
            if name.startswith(('NINFER_TEST_','NINFER_BENCH_')): del os.environ[name]
        os.environ.update(PROFILE,NINFER_KVMEM_ORIGINAL_POSITIONS=str(original),
            NINFER_R29_BUNDLE='0',NINFER_SM89_SHARED_VECTOR='0',
            NINFER_KVMEM_COLD_DIR=str(arm/'backing'),
            NINFER_TEST_FIRST_COMPACTION_DIAG=str(arm/'state'))
        command=json.loads(prior.read_text());command[-1]=str(arm/'requests.jsonl')
        with LocalServer(command,'127.0.0.1',18118,arm/'requests.jsonl') as server:
            server.wait_until_ready()
            request=urllib.request.Request('http://127.0.0.1:18118/v1/chat/completions',
                data=raw,headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(request,timeout=120) as response:
                    status=response.status;body=response.read()
            except urllib.error.HTTPError as error:
                status=error.code;body=error.read()
            (arm/'response.txt').write_bytes(body)
        text=(arm/'requests.console.log').read_text()
        assert 'KVMEM_FIRST_COMPACTION phase=after_chunk ' in text, text[-2000:]
        assert server.process.returncode==0
        report['arms'].append(dict(original=original,http=status,exit=server.process.returncode))
    phases={}
    for phase in ('before','after_compact','after_chunk'):
        files=[out/str(i)/f'state-{phase}.bin' for i in (0,1)]
        metadata=[json.loads((out/str(i)/f'state-{phase}.json').read_text()) for i in (0,1)]
        l,r=[np.memmap(path,dtype=np.uint8,mode='r') for path in files]
        rows=[];m=metadata[0]
        for region,dt,offset,stride in (
            ('conv','<u2',m['conv_offset'],m['conv_layer_bytes']),
            ('recurrent','<f4',m['recurrent_offset'],m['recurrent_layer_bytes'])):
            for layer in range(m['linear_layers']):
                left=np.frombuffer(l,dtype=dt,count=stride//np.dtype(dt).itemsize,offset=offset+layer*stride)
                right=np.frombuffer(r,dtype=dt,count=left.size,offset=offset+layer*stride)
                different=np.flatnonzero(left!=right)
                if len(different):
                    i=int(different[0]);lv=left[i].item();rv=right[i].item()
                    if region=='conv':
                        lv=np.array([int(lv)<<16],dtype=np.uint32).view(np.float32)[0].item()
                        rv=np.array([int(rv)<<16],dtype=np.uint32).view(np.float32)[0].item()
                    rows.append(dict(region=region,gdn_ordinal=layer,elements_different=len(different),
                                     first_index=i,canonical=lv,original=rv))
        phases[phase]=dict(identical=not np.any(l!=r),
            selected_equal=metadata[0]['selected']==metadata[1]['selected'],metadata=metadata,
            differing_layers=rows)
    report.update(phases=phases,state='DIAGNOSTIC_COMPLETE')
    dump(out/'report.json',report)
    print(json.dumps({k:dict(identical=v['identical'],selected_equal=v['selected_equal'],
              first_different=v['differing_layers'][:2]) for k,v in phases.items()}),flush=True)

if __name__=='__main__': main()
