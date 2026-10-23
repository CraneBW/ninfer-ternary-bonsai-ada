"""Replay frozen Pi tool histories through the native GPU Engine at 200W.

Use the installed Pi provider's message conversion. Inject only the two observed,
byte-identical BPE alternatives, prefill the saved history, then compare ordinary
re-encoding, completed-output replay, and explicit original-token continuation.
This does not rerun or grade the original long-thinking creative works.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
from tools.bench.prefix_checkpoint import PROFILE
from tools.bench.sm89_profiles import ROOT, dump
from tools.tp2.shard_v3_qwen38 import read_directory

CONVERT = r"""
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [modulePath, source, destination, needle, left, mismatchIndex] = process.argv.slice(1);
const {convertMessages} = await import(pathToFileURL(modulePath).href);
const messages = fs.readFileSync(source,'utf8').trim().split('\n').map(JSON.parse)
    .filter(e=>e.type==='message_end').map(e=>e.message).slice(0,4);
if(messages.at(-1).role!=='toolResult')throw Error('missing first tool round');
const model={id:'qualification',provider:'ninfer-local',api:'openai-completions',reasoning:true,
             input:['text'],contextWindow:262144,maxTokens:131072};
const converted=convertMessages(model,{messages},{supportsDeveloperRole:false,
    supportsStrictMode:false,supportsMidConvoSystemMessages:false});
// Match Serve's render_tool_definition order and explicit strict:false exactly.
const tools=messages[0].toolsAdded.map(t=>({type:'function',function:{name:t.name,
    parameters:t.parameters,strict:false,...(t.description?{description:t.description}:{})}}));
fs.writeFileSync(destination,JSON.stringify({messages:converted,tools,needle,left,
    mismatch_index:Number(mismatchIndex)},null,2));
"""

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=ROOT/'tmp-nvcc/opsx-pi-model-matrix-20261023')
    ap.add_argument('--profiles',nargs='+',choices=['mtp4','ar'],default=['mtp4','ar'])
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert float(subprocess.check_output(['nvidia-smi','--query-gpu=power.limit',
        '--format=csv,noheader,nounits'],text=True).strip())==200
    pi_path=shutil.which('pi');assert pi_path, 'installed Pi must be on PATH'
    pi=Path(pi_path).resolve()
    module=pi.parents[2]/'node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js'
    assert module.is_file(),module
    env={k:v for k,v in os.environ.items() if not k.startswith('NINFER_')}
    model='/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer'
    resource_dir=out/'frontend';resource_dir.mkdir()
    directory,_,payload_start,_=read_directory(model)
    objects={o['id']:o for o in directory['objects']}
    with open(model,'rb') as artifact:
        for name in ['tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja',
                     'preprocessor_config.json','video_preprocessor_config.json']:
            component='vision' if 'preprocessor' in name else 'text'
            obj=objects[directory['components'][component]['resources'][name]]
            artifact.seek(payload_start+obj['offset']);data=artifact.read(obj['bytes'])
            assert len(data)==obj['bytes'], 'incomplete frontend resource'
            (resource_dir/name).write_bytes(data)
    env.update(PROFILE,NINFER_OFFICIAL_TOKENIZER_DIR=str(resource_dir),
        NINFER_KVMEM_HOST_PAGES='128',NINFER_PREFIX_TRACE='1',NINFER_R29_BUNDLE='0',
        NINFER_GGUF_LARGE_PREFILL='1',NINFER_GGUF_SILU_MMQ='1',
        NINFER_SM89_SHARED_VECTOR='0',NINFER_SM89_ATTN_SPLIT_VECTOR='0',
        NINFER_SM89_GDN_PRENORM='0',NINFER_HOST_DECODE_OVERLAP='0')
    binary=ROOT/'build/tests/ninfer_qwen3_6_generated_history_test'
    results=dict(state='RUNNING',power_w=200,model=model,pi=str(pi),provider=str(module),
                 scope=__doc__,environment=env|{},profiles=[])
    # Keep only benchmark overrides in the receipt, never unrelated process environment.
    results['environment']={k:v for k,v in env.items() if k.startswith('NINFER_')}
    dump(out/'results.json',results)
    for name in args.profiles:
        source=args.source/name/'emperor/pi.jsonl';fixture=out/f'{name}.fixture.json'
        needle,left=('下雪','下') if name=='mtp4' else ('大秦始皇','大秦')
        subprocess.run(['node','--input-type=module','-e',CONVERT,str(module),str(source),
                        str(fixture),needle,left,str(32804 if name=='mtp4' else 43643)],check=True)
        row=dict(name=name,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),runs=[])
        results['profiles'].append(row);dump(out/'results.json',results)
        for mode in ['canonical','candidate','original']:
            backing=out/f'{name}-{mode}.backing';backing.mkdir()
            run_env=env|dict(NINFER_GENERATED_HISTORY_REPLAY='1' if mode=='candidate' else '0',
                            NINFER_KVMEM_COLD_DIR=str(backing))
            receipt=out/f'{name}-{mode}.json'
            command=[str(binary),model,str(fixture),name,mode,str(receipt)]
            with (out/f'{name}-{mode}.log').open('w') as log:
                proc=subprocess.run(command,env=run_env,stdout=log,stderr=subprocess.STDOUT)
            run=dict(mode=mode,command=command,exit_code=proc.returncode)
            if receipt.exists():run['result']=json.loads(receipt.read_text())
            row['runs'].append(run);dump(out/'results.json',results)
            if proc.returncode != 0:
                results['state']='FAILED';dump(out/'results.json',results)
                raise RuntimeError(f'{name}/{mode} failed; see log')
            print(name,mode,run['result']['ttft_s'],run['result']['reused_tokens'],flush=True)
        base,candidate,original=[r['result'] for r in row['runs']]
        assert candidate['generated_ids']==original['generated_ids'],'original-token continuation changed'
        assert candidate['reused_tokens']==original['reused_tokens']
        row['original_token_output_exact']=True
        row['ttft_reduction_pct']=(1-candidate['ttft_s']/base['ttft_s'])*100
        dump(out/'results.json',results)
    results['state']='PASS';dump(out/'results.json',results)

if __name__=='__main__':main()
