"""Small real-model regression for continuation publication after generation reclaim.

This is a runtime/cache test, not a shortened creative quality acceptance. Compare saved
before/after receipts; the long Pi xhigh workload retains its separate 128K output budget.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from tools.bench.sm89_profiles import LocalServer, ROOT, dump, stream_request
from tools.bench.prefix_checkpoint import PROFILE


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--serve', type=Path, default=ROOT/'build/apps/ninfer-serve')
    ap.add_argument('--backend', choices=['mtp4', 'ar'], default='mtp4')
    ap.add_argument('--expect-reuse', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert float(subprocess.check_output(['nvidia-smi', '--query-gpu=power.limit',
                  '--format=csv,noheader,nounits'], text=True).strip()) == 200
    env = dict(PROFILE, NINFER_KVMEM_HOST_PAGES='128',
               NINFER_KVMEM_COLD_DIR=str(out/'backing'), NINFER_PREFIX_TRACE='1',
               NINFER_GGUF_LARGE_PREFILL='1', NINFER_GGUF_SILU_MMQ='1',
               NINFER_R29_BUNDLE='0')
    for key in list(os.environ):
        if key.startswith('NINFER_') and key not in env:
            del os.environ[key]
    os.environ.update(env)
    (out/'backing').mkdir()
    log = out/'serve.jsonl'
    command = [str(args.serve.resolve()),
               '/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer',
               '--model-id', 'qualification', '--host', '127.0.0.1', '--port', '18165',
               '--max-context', '8192', '--kv-device-tokens', '4096',
               '--kvmem', '--kvmem-budget', '2048', '--kvmem-gen-reserve', '512',
               '--kvmem-transfer-mode', 'sync', '--host-kv-mib', '128',
               '--prefill-chunk', '512', '--kv-dtype', 'int8', '--greedy',
               '--no-thinking', '--preserve-thinking', '--presence-penalty', '0',
               '--seed', '42', '--max-concurrency', '1', '--request-log-jsonl', str(log)]
    if args.backend == 'mtp4':
        command += ['--spec', 'mtp', '--draft-tokens', '4', '--lm-head-draft']
    result = dict(state='RUNNING', command=command, environment=env, power_w=200,
                  scope='generation-reclaim continuation regression; not quality qualification', requests=[])
    dump(out/'results.json', result)
    messages = [dict(role='system', content='Follow the requested output format exactly.'),
                dict(role='user', content='Write numbered records from 0001 through 9999, one per line. '
                     'Each record must be: item NNNN: verified. Do not abbreviate or stop early.')]
    server = LocalServer(command, '127.0.0.1', 18165, log)
    try:
        with server:
            startup = server.wait_until_ready()
            result['startup'] = startup
            for step, limit in [('generate', 3072), ('continue', 16), ('changed_history', 16)]:
                payload = dict(model='qualification', messages=messages, max_tokens=limit,
                               temperature=0, presence_penalty=0, seed=42)
                client = stream_request(18165, payload)
                event = server.wait_for_request_done(startup['server_instance_id'])
                row = dict(step=step, payload=payload, client=client, event=event)
                result['requests'].append(row)
                dump(out/'results.json', result)
                response = client['response']['choices'][0]
                if step != 'generate':
                    assert response['message']['content'].strip() == 'ACK', 'continuation answer changed'
                if step == 'generate':
                    assert event['result']['completion_tokens'] == limit, 'fixture did not cross the reclaim window'
                    messages = messages + [dict(role='assistant', content=response['message']['content']),
                                           dict(role='user', content='Reply exactly with ACK.')]
                elif step == 'continue':
                    if args.expect_reuse:
                        assert event['result']['prefix_reuse_path'] == 'private_endpoint'
                        assert event['result']['computed_prefill_tokens'] < 128
                    messages = [dict(messages[0], content='A changed instruction: answer briefly.')] + messages[1:]
                else:
                    assert event['result']['prefix_cache_hit_tokens'] == 0, 'changed prefix reused an invalid state'
                print(step, event['result'], flush=True)
        assert server.process.returncode == 0, 'Serve did not exit cleanly'
        result['state'] = 'PASS'
    except BaseException as error:
        result.update(state='FAILED', error=str(error))
        raise
    finally:
        result['server_exit'] = server.process.returncode if server.process else None
        dump(out/'results.json', result)


if __name__ == '__main__':
    main()
