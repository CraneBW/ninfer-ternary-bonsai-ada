"""Three paired MTP4/MTP15 native Engine tests at pp4K/32K + tg128, 200W.

Each binary invocation contains both context cases with Graph prime, one warmup
and one measured repetition. Reuse an already recorded first MTP15 arm to avoid
repeating an unchanged run. Preserve every raw ninfer_bench JSON response.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from tools.bench.prefix_checkpoint import ROOT, sha


def dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def measures(path: Path, width: int) -> dict:
    raw = json.loads(path.read_text())
    cfg = raw['config']
    assert raw['artifact_type'] == 'ninfer_bench_report' and raw['tool'] == 'ninfer_bench'
    assert cfg['draft_tokens'] == width and cfg['warmup'] == cfg['repetitions'] == 1
    assert cfg['max_context'] == 36864 and cfg['prefill_chunk'] == 512
    assert cfg['kv_cache'] == 'int8-group64' and cfg['use_cuda_graph']
    assert cfg['decode_graph_prime']['primed'] and cfg['decode_graph_prime']['output_tokens'] > 0
    assert cfg['corpus_tokens'] >= 32768
    assert [(t['n_prompt'], t['n_gen']) for t in raw['tests']] == [(4096, 128), (32768, 128)]
    rows = {}
    for t in raw['tests']:
        assert t['requested_output_tokens'] == 129 and len(t['reps']) == 1
        rep = t['reps'][0]
        assert rep['generated_output_tokens'] == 129 and rep['decode_output_tokens'] == 128
        s = t['speculative']
        assert s['draft_window'] == width and s['drafted_tokens'] >= s['accepted_tokens']
        rows[str(t['n_prompt'])] = {
            'prefill_tok_s': t['prefill_tok_s_mean'],
            'decode_tok_s': t['decode_output_tok_s_mean'],
            'prefill_s': t['prefill_seconds_mean'],
            'decode_s': t['decode_seconds_mean'],
            'prepare_s': t['prepare_seconds_mean'],
            'session_measurement_s': t['total_seconds_mean'],
            'ttft_compute_proxy_s': t['prefill_seconds_mean'] + t['prepare_seconds_mean'],
            'decode_tokens': rep['decode_output_tokens'],
            'generated_tokens': rep['generated_output_tokens'],
            'rounds': s['rounds'],
            'drafted': s['drafted_tokens'],
            'accepted': s['accepted_tokens'],
            'acceptance_rate': s['acceptance_rate'],
            'full_output_ids_in_report': False,
        }
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'build/bench/ninfer_bench'
    model = ROOT.parent.parent / 'real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer'
    corpus = ROOT / 'bench/fixtures/bench_corpus.ids'
    power = subprocess.run(['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert abs(float(power.splitlines()[0]) - 200) < 0.1, f'200W required, got {power}'
    raw_first = a.reference_report.resolve()
    assert raw_first.is_file()
    summary = dict(state='RUNNING', binary=str(binary), binary_sha256=sha(binary),
                   model=str(model), model_sha256=sha(model), corpus=str(corpus),
                   corpus_sha256=sha(corpus), power_limit_w=200, physical_gpu_count=1,
                   max_context=36864, kv_dtype='int8', prefill_chunk=512, graph=True,
                   warmup=1, repetitions=1, n_prompt=[4096, 32768], n_gen=128,
                   input_portion='fixed corpus token IDs', sequence='15,4,4,15,15,4',
                   runs=[], pairs=[], report_limitations='ninfer_bench records token counts and timings, not output IDs; original six-fact Serve runs separately verify exact answers')
    dump(out / 'summary.json', summary)

    sequence = [('mtp15', 15), ('mtp4', 4), ('mtp4', 4),
                ('mtp15', 15), ('mtp15', 15), ('mtp4', 4)]
    for i, (label, width) in enumerate(sequence):
        pair = i // 2 + 1
        report = out / f'pair{pair}-{label}.json'
        if i == 0:
            shutil.copy2(raw_first, report)
            elapsed = None
            log_path = None
            print(f'REUSE pair=1 {label} qualified reference {raw_first}', flush=True)
        else:
            args = [str(binary), '--weights', str(model), '--corpus', str(corpus),
                    '-pg', '4096,128;32768,128', '--max-ctx', '36864', '-r', '1',
                    '--warmup', '1', '--kv-dtype', 'int8', '--prefill-chunk', '512',
                    '--spec', 'mtp', '--draft-tokens', str(width), '--lm-head-draft',
                    '--output', 'json', '--output-file', str(report)]
            log_path = out / f'pair{pair}-{label}.log'
            print(f'START pair={pair} {label}', flush=True)
            t0 = time.perf_counter()
            with log_path.open('w') as log:
                completed = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - t0
            if completed.returncode != 0:
                summary['state'] = 'STOPPED_ON_FAILED_ARM'
                summary['failed_arm'] = dict(pair=pair, label=label, returncode=completed.returncode,
                                             log=str(log_path), report=str(report))
                dump(out / 'summary.json', summary)
                raise SystemExit(f'FAILED pair={pair} {label} in {log_path}')
        metrics = measures(report, width)
        summary['runs'].append(dict(pair=pair, label=label, width=width, output=str(report),
                                    log=str(log_path) if log_path else None,
                                    process_wall_s=elapsed, cases=metrics))
        dump(out / 'summary.json', summary)
        print(f'PASS pair={pair} {label} 4K/32K complete129/decode128', flush=True)

    for context in ('4096', '32768'):
        ratios = {'prefill_tok_s': [], 'decode_tok_s': [], 'session_reduction': [],
                  'ttft_compute_proxy_reduction': []}
        for pair in range(1, 4):
            arm = [r for r in summary['runs'] if r['pair'] == pair]
            b = next(r['cases'][context] for r in arm if r['label'] == 'mtp15')
            c = next(r['cases'][context] for r in arm if r['label'] == 'mtp4')
            ratios['prefill_tok_s'].append(c['prefill_tok_s'] / b['prefill_tok_s'] - 1)
            ratios['decode_tok_s'].append(c['decode_tok_s'] / b['decode_tok_s'] - 1)
            ratios['session_reduction'].append(1 - c['session_measurement_s'] / b['session_measurement_s'])
            ratios['ttft_compute_proxy_reduction'].append(1 - c['ttft_compute_proxy_s'] / b['ttft_compute_proxy_s'])
        summary['pairs'].append(dict(n_prompt=int(context), per_pair=ratios,
                                     medians={k: statistics.median(v) for k, v in ratios.items()}))
    summary['state'] = 'THREE_PAIRS_BOTH_CONTEXTS_12_OF_12_MEASURED'
    dump(out / 'summary.json', summary)
    print('THREE_PAIRS_BOTH_CONTEXTS_12_OF_12_MEASURED', flush=True)


if __name__ == '__main__':
    main()
