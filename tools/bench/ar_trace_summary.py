"""Summarize GPU kernels inside ordinary AR NVTX rounds (not host gaps)."""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('database',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=sqlite3.connect(f'file:{a.database.resolve()}?mode=ro',uri=True)
    ranges=c.execute("select e.start,e.end from NVTX_EVENTS e left join StringIds s on e.textId=s.id "
                     "where coalesce(e.text,s.value)='decode.ordinary_round' order by e.start").fetchall()
    assert ranges,'no ordinary AR ranges; do not substitute an MTP trace'
    starts=[r[0] for r in ranges]; kernels=defaultdict(lambda:[0,0]);groups=defaultdict(lambda:[0,0])
    for start,end,name in c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                                   'join StringIds s on s.id=k.demangledName order by k.start'):
        i=bisect.bisect_right(starts,start)-1
        if i<0 or ranges[i][1] is None or end>ranges[i][1]:continue
        if any(v in name for v in ('quantize_q8','quantize_vector_kernel')):group='activation_q8_quantization'
        elif any(v in name.lower() for v in ('mul_mat_vec','gemv','matvec','vector_product','::gguf::detail::vec::kernel')):group='vector_products'
        elif any(v in name.lower() for v in ('gated_delta','gdn','recurrent')):group='gdn_state'
        elif any(v in name.lower() for v in ('attention','attn')):group='attention'
        else:group='other'
        for target,key in [(kernels,name),(groups,group)]:
            target[key][0]+=1;target[key][1]+=end-start
    total=sum(v[1] for v in kernels.values());assert total>0
    def rows(mapping):
        return [dict(name=k,count=v[0],gpu_ms=v[1]/1e6,fraction=v[1]/total)
                for k,v in sorted(mapping.items(),key=lambda item:-item[1][1])]
    report=dict(scope='GPU kernel duration sum inside ordinary AR rounds; probe overhead and gaps excluded',
        database=str(a.database.resolve()),round_count=len(ranges),kernel_gpu_ms=total/1e6,
        groups=rows(groups),kernels=rows(kernels))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='kernels'}),flush=True)

if __name__=='__main__':main()
