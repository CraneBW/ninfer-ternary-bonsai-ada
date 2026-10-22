"""Create or restore the frozen 213K/256K complete canonical prefix archive.

Uses only the public native Serve/Engine route. Native q0 archives hold all KV
owners, immutable K, mean/profile state and the complete recurrent StateImage.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
from tools.bench.kvmem_async import signature, write_json
from tools.bench import kvmem_qualification

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT.parent.parent
PROFILE=dict(NINFER_R26_BUNDLE='0',NINFER_R27_BUNDLE='1',NINFER_R28_BUNDLE='0',NINFER_R29_BUNDLE='1',
 NINFER_PROMPT_I8_PIPELINE='1',NINFER_KVMEM_SOURCE_READ_AHEAD='1',NINFER_KVMEM_SOURCE_LAYER_BATCH='1',
 NINFER_KVMEM_SOURCE_COPY_BATCH='1',NINFER_GGUF_MMQ_BF16='1',NINFER_GGUF_IQ2_S_MMQ_BF16='1',
 NINFER_GGUF_IQ3_S_MMQ_I64='1',NINFER_GGUF_IQ3_XXS_MMQ_I64='1',NINFER_KVMEM_ORIGINAL_POSITIONS='0',
 NINFER_KVMEM_SOURCE_EVENT_PIPELINE='0',NINFER_KVMEM_TRANSFER_PIPELINE='0',
 NINFER_KVMEM_CANONICAL_PREPARE='0',NINFER_KVMEM_CANONICAL_QUERY_REUSE='0')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def power(value):
 subprocess.run(['sudo','-n','nvidia-smi','-i','0','-pl',str(value)],check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['create','restore','prepare']);p.add_argument('--case',choices=['213k','256k'],required=True)
 p.add_argument('--output',type=Path,required=True,help='fresh receipt directory')
 p.add_argument('--checkpoint',type=Path,help='archive path; defaults beside frozen input')
 p.add_argument('--checkpoint-stage',choices=['auto','raw','canonical'],default='auto',
  help='auto prefers an available qualified ABI2 pack with query reuse; raw selects the ABI1 reference; prepare derives ABI2 without repeating the long cold prefill')
 p.add_argument('--fixtures',type=Path,default=WORK/'benchmarks/frozen-long-context')
 p.add_argument('--binary',type=Path,default=ROOT/'build/apps/ninfer-serve')
 p.add_argument('--draft-tokens',type=int,choices=range(1,16),default=15,
  metavar='1..15',help='explicit MTP draft window; archived state must match on restore (default 15)')
 p.add_argument('--power-limit',type=int,choices=[200,300],default=200,
  help='200W for routine tests/tuning; 300W only for final release validation; always restore 200W')
 p.add_argument('--page-pipeline',action='store_true',help='opt-in source-event + async D2H candidate; default remains sync')
 p.add_argument('--canonical-prepare',action='store_true',help='opt-in exact canonical replay preparation candidate')
 p.add_argument('--canonical-query-reuse',action=argparse.BooleanOptionalAction,default=True,
  help='qualified canonical-view continuation for these frozen cases (default on); --no-canonical-query-reuse selects the reference; original-position mode remains off')
 p.add_argument('--checkpoint-load-pipeline',action=argparse.BooleanOptionalAction,default=None,
  help='private COW extents + two-slot H2D + direct Host reads; default on for paired-qualified213k, opt-in for256k (quality-only); --no-checkpoint-load-pipeline selects the reference')
 p.add_argument('--checkpoint-io',choices=['legacy','indexed'],default='indexed',
  help='legacy FNV twice or indexed XXH3-128 full-file plus per-consume verification')
 a=p.parse_args()
 if a.checkpoint_load_pipeline is None:a.checkpoint_load_pipeline=a.case=='213k'
 f=read(a.fixtures/a.case/'manifest.json')
 out=a.output.resolve();out.mkdir(parents=True);(out/'backing').mkdir()
 selected_stage=a.checkpoint_stage
 if a.action=='restore' and selected_stage=='auto' and a.canonical_query_reuse:
  available=a.checkpoint or a.fixtures/a.case/'canonical-ready.nkv'
  if available.exists():
   saved=read(str(available)+'.json')
   assert saved['archive_abi'] in (1,2),'unknown checkpoint state ABI'
   selected_stage='canonical' if saved['archive_abi']==2 else 'raw'
 canonical=a.action=='prepare' or selected_stage=='canonical'
 assert not canonical or a.canonical_query_reuse,'canonical checkpoint requires query reuse'
 assert not (a.action=='create' and canonical),'use prepare to derive canonical state from the existing raw archive'
 checkpoint=(a.checkpoint or a.fixtures/a.case/('canonical-ready.nkv' if canonical else 'prefix.nkv')).resolve()
 metadata=Path(str(checkpoint)+'.json');model=Path(f['model']['path'])
 assert signature(model)==f['model'],'frozen model changed'
 for r in f['requests']:assert sha(r['file'])==r['sha256'],'frozen input changed'
 binary=a.binary.resolve();binary_hash=sha(binary)
 if a.action in ('create','prepare'):
  assert not checkpoint.exists() and not metadata.exists(),'never overwrite a checkpoint'
  if a.action=='create':
   # A failed full-prefix write can otherwise spend several minutes prefilling
   # before discovering that the destination lacks room for the archive plus
   # transient writer metadata. The frozen model/case carries the qualified
   # archive byte size; leave a bounded 2 GiB safety floor beyond it.
   expected_archive=int(f.get('snapshot',{}).get('archive_bytes',0))
   if expected_archive:
    required=expected_archive+2*(1<<30)
    available=shutil.disk_usage(checkpoint.parent).free
    assert available>required,(
     f'checkpoint create requires {required} free bytes including 2 GiB staging headroom; '
     f'{available} available at {checkpoint.parent}')
   digest=sha(model)
  else:
   source=(a.fixtures/a.case/'prefix.nkv').resolve();m=read(str(source)+'.json')
   assert m['state']=='SAVE_ORIGINAL6_PASS' and m['archive_abi']==1 and m['case']==a.case
   assert m['model']==signature(model),'source snapshot model changed'
   digest=m['model_sha256']
   # Full native archive plus four StateImages, immutable-K temporary backing,
   # and a 2 GiB free-space floor. No old pack is removed.
   required=source.stat().st_size+4*153954304+5_000_000_000+2*(1<<30)
   assert shutil.disk_usage(checkpoint.parent).free>required,'insufficient room for derived archive plus bounded backing'
 else:
  m=read(metadata);assert m['state']=='SAVE_ORIGINAL6_PASS' and m['case']==a.case
  assert m['model']==signature(model) and m['archive_abi']==(2 if canonical else 1),'snapshot model/state ABI changed'
  assert m.get('draft_tokens',15)==a.draft_tokens,'snapshot MTP draft window mismatch; create a separate archive'
  if canonical:
   assert m['canonical_query_reuse'] and m['canonical_prepare']==a.canonical_prepare
   assert m['page_pipeline']==a.page_pipeline,'canonical checkpoint pipeline profile changed'
  digest=m['model_sha256']
 mode='load' if a.action=='restore' else 'save'
 env=dict(PROFILE,NINFER_KVMEM_COLD_DIR=str(out/'backing'),
  NINFER_BENCH_CHECKPOINT_MODEL_SHA256=digest)
 key='NINFER_BENCH_CANONICAL_CHECKPOINT' if canonical else 'NINFER_BENCH_PREFIX_CHECKPOINT'
 env[key]=str(checkpoint);env[key+'_MODE']=mode
 if a.action=='prepare':
  env.update(NINFER_BENCH_PREFIX_CHECKPOINT=str(source),NINFER_BENCH_PREFIX_CHECKPOINT_MODE='load')
 if a.page_pipeline:env.update(NINFER_KVMEM_SOURCE_EVENT_PIPELINE='1',NINFER_KVMEM_TRANSFER_PIPELINE='1')
 if a.canonical_prepare:env['NINFER_KVMEM_CANONICAL_PREPARE']='1'
 if a.canonical_query_reuse:env['NINFER_KVMEM_CANONICAL_QUERY_REUSE']='1'
 env['NINFER_BENCH_CHECKPOINT_LOAD_PIPELINE']='1' if a.checkpoint_load_pipeline else '0'
 env['NINFER_BENCH_CHECKPOINT_FAST_VERIFY']='1' if a.checkpoint_io=='indexed' else '0'
 args=[str(binary),str(model),'--model-id','qualification','--host','127.0.0.1','--port','18118',
  '--max-context',str(f['context_limit']),'--kv-device-tokens','36864','--prefill-chunk','512',
  '--host-kv-mib','0','--max-concurrency','1','--kv-dtype','int8','--greedy','--no-thinking','--log-level','info',
  '--kvmem','--kvmem-budget','32768','--kvmem-gen-reserve','512','--kvmem-action-lrfu',
  '--kvmem-lrfu-lambda','0.5','--kvmem-lrfu-top-frac','0.3','--kvmem-transfer-mode','async' if a.page_pipeline else 'sync',
  '--spec','mtp','--draft-tokens',str(a.draft_tokens),'--lm-head-draft','--request-log-jsonl',str(out/'requests.jsonl')]
 stage=dict(id=a.case+'-'+mode,args=args,requests=f['requests'],env=env,context=f['context_limit'],mode='async' if a.page_pipeline else 'sync')
 manifest=dict(binary=signature(binary),model=signature(model),stages=[stage],max_calls=6,no_retry=True,
  trace=False,request_timeout_s=600,startup_timeout_s=120,reason='complete prefix archive, not cold-prefill throughput')
 write_json(out/'manifest.json',manifest)
 profile='page-pipeline-candidate' if a.page_pipeline else 'qualified-sync'
 if a.checkpoint_load_pipeline:profile+='+checkpoint-load-pipeline'
 if a.canonical_prepare:profile+='+canonical-prepare-candidate'
 if a.canonical_query_reuse:profile+='+canonical-query-reuse'
 result=dict(state='RUNNING',archive_abi=2 if canonical else 1,checkpoint_stage='canonical' if canonical else 'raw',
  canonical_query_reuse=a.canonical_query_reuse,canonical_prepare=a.canonical_prepare,page_pipeline=a.page_pipeline,case=a.case,action=a.action,model=signature(model),model_sha256=digest,
  draft_tokens=a.draft_tokens,
  binary_sha256=binary_hash,checkpoint=str(checkpoint),power_test_w=a.power_limit,profile=profile,
  checkpoint_io=a.checkpoint_io,checkpoint_load_pipeline=a.checkpoint_load_pipeline)
 write_json(out/'checkpoint-receipt.json',result)
 try:
  power(a.power_limit)
  # Reuse the existing owned-server HTTP/stream/usage/exact-answer runner.
  kvmem_qualification.run(out)
  data=read(out/'results.json');s=data['stages'][0]
  assert data['sent']==6 and s['status']=='PASS' and s['server_exit']==0
  assert all(r['correct'] is True and r['terminal'] and r['usage'] for r in s['requests'])
  log=(out/(stage['id']+'.log')).read_text()
  assert f'KVMEM_BENCH_CHECKPOINT action={mode} ' in log and 'complete=1' in log
  assert checkpoint.exists() and checkpoint.stat().st_size>0
  metrics=s['metrics']
  for k,v in metrics.items():
   if any(x in k for x in ('kvmem_inflight_pages','kvmem_inflight_bytes','requests_active')):assert v==0
  restored=metrics['ninfer:restored_prefill_tokens_total']
  assert (restored==0) if a.action=='create' else (restored>0)
  result.update(state='SAVE_ORIGINAL6_PASS' if mode=='save' else 'RESTORE_ORIGINAL6_PASS',correct=6,
   archive_bytes=checkpoint.stat().st_size,restored_tokens=restored,metrics=metrics,
   checkpoint_event=next(x for x in log.splitlines() if f'KVMEM_BENCH_CHECKPOINT action={mode} ' in x))
  if mode=='save':write_json(metadata,result)
 except BaseException as e:
  result.update(state='FAILED_STOPPED_NO_RETRY',error=repr(e));raise
 finally:
  power(200);result['restored_power_limit_w']=200;write_json(out/'checkpoint-receipt.json',result)
 print(result['state'],result['checkpoint_event'],flush=True)
if __name__=='__main__':main()
