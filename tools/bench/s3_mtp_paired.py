"""Run three 200W paired, whole-session MTP-width comparisons on frozen 256K facts.

The two independently qualified raw prefix checkpoints must be on one filesystem.
Every arm launches a new native Serve through prefix_checkpoint.py; no checkpoint
or model is rebuilt, and every original frozen answer must match exactly.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.bench.prefix_checkpoint import ROOT, read, sha


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def arm_measurements(folder: Path) -> dict:
    result = read(folder / 'results.json')
    stage = result['stages'][0]
    assert result['sent'] == 6 and stage['status'] == 'PASS' and stage['server_exit'] == 0
    assert all(r['correct'] and r['terminal'] and r['status'] == 200 for r in stage['requests'])
    receipts = [read(folder / f'256k-load-{i}.receipt.json') for i in range(6)]
    ttfts = []
    for r in receipts:
        events = [e for e in r['events'] if e['kind'] == 'model_output' and e['output']]
        assert events and any(e['kind'] == 'terminal' for e in r['events'])
        ttfts.append((events[0]['received_ns'] - r['http']['sent_ns']) / 1e9)
    metrics = stage['metrics']
    return {
        'correct': 6,
        'first_ttft_s': ttfts[0],
        'later_ttfts_s': ttfts[1:],
        'later_ttft_sum_s': sum(ttfts[1:]),
        'session_wall_s': (receipts[-1]['http']['ended_ns'] - receipts[0]['http']['sent_ns']) / 1e9,
        'request_wall_s': [(r['http']['ended_ns'] - r['http']['sent_ns']) / 1e9 for r in receipts],
        'draft_proposed': metrics['ninfer:draft_tokens_total'],
        'draft_accepted': metrics['ninfer:draft_accepted_tokens_total'],
        'history_replay_tokens': metrics['ninfer:kvmem_history_replay_tokens_total'],
        'physical_prefill_tokens': metrics['ninfer:physical_prefill_tokens_total'],
        'restored_prefill_tokens': metrics['ninfer:restored_prefill_tokens_total'],
        'inflight_pages': metrics['ninfer:kvmem_inflight_pages'],
        'inflight_bytes': metrics['ninfer:kvmem_inflight_bytes'],
        'active_requests': metrics['ninfer:requests_active'],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='qualified MTP15 ABI1 snapshot')
    p.add_argument('--candidate', type=Path, required=True, help='qualified MTP4 ABI1 snapshot')
    p.add_argument('--output', type=Path, required=True, help='new receipt directory')
    a = p.parse_args()
    base, candidate = a.baseline.resolve(), a.candidate.resolve()
    for path, width in [(base, 15), (candidate, 4)]:
        assert path.is_file(), f'missing snapshot: {path}'
        m = read(Path(str(path) + '.json'))
        assert m['state'] == 'SAVE_ORIGINAL6_PASS' and m['case'] == '256k' and m['archive_abi'] == 1
        assert m.get('draft_tokens', 15) == width, f'wrong draft width in {path}'
    assert base.stat().st_dev == candidate.stat().st_dev, 'checkpoint media differ'
    assert read(Path(str(base) + '.json'))['model'] == read(Path(str(candidate) + '.json'))['model']

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sequence = [('mtp15', 15, base), ('mtp4', 4, candidate),
                ('mtp4', 4, candidate), ('mtp15', 15, base),
                ('mtp15', 15, base), ('mtp4', 4, candidate)]
    summary = dict(state='RUNNING', binary_sha256=sha(ROOT / 'build/apps/ninfer-serve'),
                   power_limit_w=200, case='256k', physical_kv_tokens=36864,
                   kvmem_budget_tokens=32768, gen_reserve_tokens=512,
                   checkpoint_stage='raw', checkpoint_io='indexed',
                   checkpoint_load_pipeline=False, canonical_query_reuse=True,
                   same_device=base.stat().st_dev, baseline_checkpoint=str(base),
                   candidate_checkpoint=str(candidate), runs=[], pairs=[])
    write(out / 'summary.json', summary)

    for i, (label, width, checkpoint) in enumerate(sequence):
        pair_number = i // 2 + 1
        folder = out / f'pair-{pair_number}-{label}'
        cmd = [sys.executable, '-m', 'tools.bench.prefix_checkpoint', 'restore',
               '--case', '256k', '--draft-tokens', str(width), '--checkpoint-stage', 'raw',
               '--checkpoint', str(checkpoint), '--output', str(folder),
               '--power-limit', '200', '--checkpoint-io', 'indexed',
               '--no-checkpoint-load-pipeline', '--canonical-query-reuse']
        print(f'START pair={pair_number} {label}', flush=True)
        started = time.perf_counter()
        with (out / f'pair-{pair_number}-{label}-runner.log').open('w') as log:
            finished = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        arm = dict(pair=pair_number, label=label, width=width, runner_exit=finished.returncode,
                   process_wall_s=elapsed, output=str(folder), command=cmd)
        if finished.returncode != 0:
            summary['runs'].append(arm)
            summary['state'] = 'STOPPED_ON_FAILED_ARM'
            write(out / 'summary.json', summary)
            raise SystemExit(f'FAILED pair={pair_number} {label}, see {folder}')
        arm.update(arm_measurements(folder))
        assert arm['inflight_pages'] == arm['inflight_bytes'] == arm['active_requests'] == 0
        summary['runs'].append(arm)
        write(out / 'summary.json', summary)
        print(f'PASS pair={pair_number} {label} session={arm["session_wall_s"]:.3f}s '
              f'first_TTFT={arm["first_ttft_s"]:.3f}s elapsed={elapsed:.3f}s', flush=True)

    improvements = []
    for pair in range(1, 4):
        arms = [r for r in summary['runs'] if r['pair'] == pair]
        b = next(r for r in arms if r['label'] == 'mtp15')
        c = next(r for r in arms if r['label'] == 'mtp4')
        ratio = 1 - c['session_wall_s'] / b['session_wall_s']
        improvements.append(ratio)
        summary['pairs'].append(dict(pair=pair, first=arms[0]['label'],
                                     mtp15_session_wall_s=b['session_wall_s'],
                                     mtp4_session_wall_s=c['session_wall_s'],
                                     mtp4_reduction_fraction=ratio))
    summary['median_mtp4_session_reduction_fraction'] = statistics.median(improvements)
    summary['state'] = 'THREE_PAIRS_36_OF_36_PASS'
    write(out / 'summary.json', summary)
    print(f'THREE_PAIRS_36_OF_36_PASS median_session_reduction={statistics.median(improvements):.5%}',
          flush=True)


if __name__ == '__main__':
    main()
