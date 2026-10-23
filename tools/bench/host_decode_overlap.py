"""Directed C1 publication lifetime regression; not a creative quality benchmark.

Uses the existing owned Serve/SSE harness. AR, MTP4 and short-capacity DFlash2 compare identical
normal output, partial-stop output and tool arguments with the overlap switch
off/on; cancellation is followed by a fresh request in the same process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from tools.bench.continuation_lifecycle_smoke import post
from tools.bench.prefix_checkpoint import PROFILE
from tools.bench.sm89_profiles import LocalServer, ROOT, dump


def plain_post(port, payload):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps(dict(payload, stream=False)).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        assert response.status == 200
        return json.load(response)


def run(out, backend, mode):
    out.mkdir()
    env = dict(PROFILE, NINFER_HOST_DECODE_OVERLAP=str(mode),
               NINFER_R29_BUNDLE='0', NINFER_SM89_SHARED_VECTOR='0',
               NINFER_GGUF_LARGE_PREFILL='1', NINFER_GGUF_SILU_MMQ='1')
    for key in list(os.environ):
        if key.startswith('NINFER_') and key not in env:
            del os.environ[key]
    os.environ.update(env)
    port = 18177
    log = out/'requests.jsonl'
    command = [str(ROOT/'build/apps/ninfer-serve'),
        '/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer',
        '--model-id', 'qualification', '--host', '127.0.0.1', '--port', str(port),
        '--max-context', '8192', '--kv-capacity', '8192', '--max-concurrency', '1',
        '--kv-dtype', 'int8', '--prefill-chunk', '1024', '--greedy', '--no-thinking',
        '--presence-penalty', '0', '--no-prefix-reuse', '--request-log-jsonl', str(log)]
    if backend == 'mtp4':
        command += ['--spec', 'mtp', '--draft-tokens', '4', '--lm-head-draft']
    elif backend == 'dflash2':
        command += ['--spec', 'dflash2', '--draft-tokens', '3', '--lm-head-draft']
    report = dict(state='RUNNING', backend=backend, mode=mode, command=command,
                  environment=env, power_w=200)
    dump(out/'report.json', report)
    base = dict(model='qualification', temperature=0, seed=42, max_tokens=256,
        messages=[dict(role='system', content='Follow the requested format exactly.'),
                  dict(role='user', content='Begin with ALPHA. Explain why unit tests and '
                       'integration tests serve different purposes. Give many concrete '
                       'software examples; continue until stopped.')])
    server = LocalServer(command, '127.0.0.1', port, log)
    try:
        with server:
            report['startup'] = server.wait_until_ready()
            assert report['startup']['memory']['available_after_startup_bytes'] >= 256*1024**2
            normal = post(port, base)
            report['normal'] = normal
            assert normal['complete'] and len(normal['text']) > 100
            marker = normal['text'][len(normal['text'])//2:len(normal['text'])//2+20]
            stopped = post(port, dict(base, stop=[marker]))
            report.update(stopped=stopped, stop_marker=marker)
            assert stopped['complete'] and stopped['finish'] == ['stop']
            assert stopped['text'] == normal['text'].split(marker, 1)[0]
            tool_payload = dict(base, messages=[dict(role='user', content=
                'Call the emit function with value 42 and label "publication". Do not explain.')],
                tools=[dict(type='function', function=dict(name='emit',
                    description='Store one value and its label.', parameters=dict(type='object',
                    properties=dict(value=dict(type='integer'), label=dict(type='string')),
                    required=['value','label'], additionalProperties=False)))])
            tool = plain_post(port, tool_payload)
            report['tool'] = tool
            calls = tool['choices'][0]['message'].get('tool_calls', [])
            assert len(calls) == 1 and calls[0]['function']['name'] == 'emit'
            arguments = json.loads(calls[0]['function']['arguments'])
            assert arguments == dict(value=42, label='publication')
            cancelled = post(port, dict(base, max_tokens=1024), True, cancel_after=100)
            report['cancelled'] = cancelled
            assert cancelled['cancelled']
            after = post(port, dict(base, max_tokens=8,
                messages=[dict(role='user', content='Reply exactly DELTA.')]))
            report['after'] = after
            assert after['complete'] and after['text'] == 'DELTA'
            for _ in range(50):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=3) as response:
                    metrics = response.read().decode()
                if re.search(r'^ninfer:requests_active 0$', metrics, re.M):
                    break
                time.sleep(.1)
            assert re.search(r'^ninfer:requests_active 0$', metrics, re.M)
            (out/'metrics.txt').write_text(metrics)
            report.update(normal=normal, stopped=stopped, stop_marker=marker,
                          tool=tool, tool_arguments=arguments, cancelled=cancelled, after=after)
        assert server.process.returncode == 0
        lines = re.findall(r'HOST_DECODE_OVERLAP (.+)', log.with_suffix('.console.log').read_text())
        report['overlap'] = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
                             for line in lines]
        if mode:
            assert len(report['overlap']) == 1
            counters = report['overlap'][0]
            assert counters['scheduled'] == counters['processed']
            assert counters['overlapped'] > 0 and counters['pending'] == counters['failures'] == 0
        events = [json.loads(line) for line in log.read_text().splitlines()]
        report['request_events'] = [e for e in events if e.get('event') in ('request_done','request_error')]
        if backend == 'mtp4':
            specs = [e['speculative'] for e in events if e.get('event') == 'request_done']
            assert any(0 < e['accepted_tokens'] < e['drafted_tokens'] for e in specs)
        report['state'] = 'PASS'
    except BaseException as error:
        report.update(state='FAILED', error=str(error))
        raise
    finally:
        report['server_exit'] = server.process.returncode if server.process else None
        dump(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert float(subprocess.check_output(['nvidia-smi','--query-gpu=power.limit',
        '--format=csv,noheader,nounits'], text=True).strip()) == 200
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(state='RUNNING', pairs=[])
    for backend in ('mtp4', 'ar', 'dflash2'):
        pair = [run(out/f'{backend}-{mode}', backend, mode) for mode in (0, 1)]
        for key in ('normal', 'stopped', 'after'):
            assert pair[0][key]['text'] == pair[1][key]['text']
            assert pair[0][key]['usage'] == pair[1][key]['usage']
        assert pair[0]['tool_arguments'] == pair[1]['tool_arguments']
        result['pairs'].append(dict(backend=backend, same_output_and_usage=True, arms=pair))
        dump(out/'comparison.json', result)
        print(backend, 'PASS', flush=True)
    result['state'] = 'PASS'
    dump(out/'comparison.json', result)


if __name__ == '__main__':
    main()
