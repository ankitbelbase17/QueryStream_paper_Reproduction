"""Small server-side control plane; JSON IO for the laptop orchestrator."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from experiment import read,write,key,feature_path

def status():
    subset=read(ROOT/'data/subset.json'); cal=read(ROOT/'data/calibration.json')
    calvid=set(s['video_id'] for s in cal); allvid=set(s['video_id'] for s in subset)
    out={'feature_total':len(allvid),'feature_done':sum(feature_path(v).exists() for v in allvid),
         'calibration_feature_total':len(calvid),'calibration_feature_done':sum(feature_path(v).exists() for v in calvid),
         'validated':all((ROOT/'validation'/n).exists() for n in ['qwen3_equivalence.json','metric_checks.json','inference_smoke.json']),
         'final':(ROOT/'results/FINAL_RESULTS.json').exists()}
    for stage in ['calibration','test']:
        jobs=ROOT/f'jobs_{stage}.json'
        out[stage+'_jobs']=len(read(jobs)) if jobs.exists() else 0
        out[stage+'_done']=len(list((ROOT/'predictions'/stage).glob('*.json')))
    out['pending_judgments']=sum(not (ROOT/'judge_results'/p.name).exists() for p in (ROOT/'judge_requests').glob('*.json'))
    out['judge_done']=len(list((ROOT/'judge_results').glob('*.json')))
    out['errors']=len(list((ROOT/'errors').glob('*.json')))
    return out

def judges():
    result=[]
    for p in sorted((ROOT/'judge_requests').glob('*.json')):
        if not (ROOT/'judge_results'/p.name).exists(): result.append(read(p))
        if len(result)>=12: break
    return result

def ingest():
    verdicts=json.load(sys.stdin)
    for v in verdicts:
        assert len(v['key'])==64 and all(c in '0123456789abcdef' for c in v['key'])
        assert type(v['score']) is int and 1<=v['score']<=5
        assert (ROOT/'judge_requests'/f"{v['key']}.json").exists()
        write(ROOT/'judge_results'/f"{v['key']}.json",v)
    return {'written':len(verdicts)}

def launch(stage):
    """Start up to 4 workers on this node, only on GPUs with >=40GB free."""
    if stage!='features':
        jobs=read(ROOT/f'jobs_{stage}.json')
        if all((ROOT/'predictions'/stage/f"{key(stage+'|'+j['policy']+'|'+j['sample']['id'])}.json").exists() for j in jobs):
            return {'host':os.uname().nodename,'stage':stage,'started':[],'complete':True}
    elif all(feature_path(s['video_id']).exists() for s in read(ROOT/'data/subset.json')):
        return {'host':os.uname().nodename,'stage':stage,'started':[],'complete':True}
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    starts=[]
    for line in info.strip().splitlines():
        gpu,free=map(int,line.split(','))
        if free<40000: continue
        # Never colocate this experiment's feature/inference stages on one GPU.
        busy=any(subprocess.run(['tmux','has-session','-t',f'qs-{s}-{gpu}'],capture_output=True).returncode==0
                 for s in ['features','calibration','test'])
        if busy: continue
        session=f'qs-{stage}-{gpu}'
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0: continue
        log=ROOT/'logs'/f'{stage}_{os.uname().nodename}_{gpu}.log'
        command=f'bash {ROOT}/worker.sh {gpu} {stage} >> {log} 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
        starts.append(gpu)
    return {'host':os.uname().nodename,'stage':stage,'started':starts}

def reclaim():
    """Recover this host's interrupted work only, after verifying PID death."""
    recovered=[]
    for owner in (ROOT/'claims').glob('*/*/owner.json'):
        record=read(owner)
        if record['host']!=os.uname().nodename: continue
        try: os.kill(record['pid'],0)
        except ProcessLookupError:
            directory=owner.parent
            stage=directory.parent.name
            completed=(ROOT/'features'/f'{directory.name}.pt').exists() if stage=='features' else (ROOT/'predictions'/stage/f'{directory.name}.json').exists()
            if completed: continue
            archive=ROOT/'claim_archive'/stage
            archive.mkdir(parents=True,exist_ok=True)
            prior=len(list(archive.glob(directory.name+'_*')))
            if prior>=3: continue
            directory.rename(archive/f'{directory.name}_{time.time_ns()}')
            recovered.append({'stage':stage,'key':directory.name})
        except PermissionError: pass
    return recovered

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['status','judges','ingest','launch','reclaim']);p.add_argument('--stage',default='features'); a=p.parse_args()
    result={'status':status,'judges':judges,'ingest':ingest,'reclaim':reclaim}.get(a.command,lambda:launch(a.stage))()
    print(json.dumps(result),flush=True)
