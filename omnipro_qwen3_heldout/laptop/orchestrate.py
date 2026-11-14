"""Laptop: orchestration and text-only OpenCode judge. Heavy work stays remote."""
from concurrent.futures import ThreadPoolExecutor
import json
import fcntl
import os
from pathlib import Path
import subprocess
import time
ROOT='/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114'
HERE=Path(__file__).resolve().parent
MODEL='github-copilot/gemini-3.8-flash'
HOSTS=['clariden-ln001','clariden-ln002','clariden-ln003','clariden-ln004']

def ssh(command,host=HOSTS[0],data=None,timeout=120):
    p=subprocess.run(['ssh','-o','BatchMode=yes',host,command],input=data,text=True,capture_output=True,timeout=timeout)
    if p.returncode: raise RuntimeError(f'{host}: {p.stderr[-3000:]} {p.stdout[-3000:]}')
    return p.stdout
def rpc(cmd,host=HOSTS[0],data=None):
    return json.loads(ssh(f'python3 {ROOT}/control.py {cmd}',host,data))
def judge(job):
    # Preserve exact official prompt. Fresh isolated judging session, no tools.
    p=subprocess.run(['opencode','run','--pure','--dir',str(HERE),'--model',MODEL,'--format','json',job['prompt']],
                     text=True,capture_output=True,timeout=180)
    if p.returncode: raise RuntimeError(p.stderr[-2000:])
    text=[]; session=None; cost=0
    for line in p.stdout.splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        session=event.get('sessionID',session)
        if event.get('type')=='text': text.append(event['part']['text'])
        if event.get('type')=='step_finish': cost+=event.get('part',{}).get('cost',0)
    raw='\n'.join(text).strip()
    start=raw.find('{');end=raw.rfind('}')
    result=json.loads(raw[start:end+1])
    assert type(result['score']) is int and 1<=result['score']<=5
    return {**result,'key':job['key'],'model':MODEL,'session_id':session,'raw_response':raw,'reported_cost':cost}

def main():
    lock=(HERE/'orchestrator.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shortlisting=False; lastlaunch=0; lastscore=0
    while True:
        try:
            st=rpc('status'); print(time.strftime('%F %T'),json.dumps(st),flush=True)
            (HERE/'status.json').write_text(json.dumps(st,indent=2))
            if st['final']:
                # Only small final summaries return to the laptop.
                for name in ['FINAL_RESULTS.json','RESULTS.md']:
                    content=ssh(f'cat {ROOT}/results/{name}')
                    (HERE/name).write_text(content)
                print('EXPERIMENT COMPLETE',flush=True); return
            if not st['validated']:
                time.sleep(30); continue
            if st['calibration_feature_done']==st['calibration_feature_total'] and not st['calibration_jobs'] and not shortlisting:
                ssh(f'tmux new-session -d -s qs-shortlist "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 {ROOT}/experiment.py shortlist > {ROOT}/logs/shortlist.log 2>&1"')
                shortlisting=True
            if time.time()-lastlaunch>180:
                stage='test' if st['test_jobs'] else 'calibration' if st['calibration_jobs'] else 'features'
                if stage=='test' and st['feature_done']<st['feature_total']: stage='features'
                for h in HOSTS:
                    recovered=rpc('reclaim',h)
                    if recovered: print('RECOVERED',h,recovered,flush=True)
                    print('LAUNCH',rpc(f'launch --stage {stage}',h),flush=True)
                lastlaunch=time.time()
            if st['calibration_jobs'] and time.time()-lastscore>60:
                stage='test' if st['test_jobs'] else 'calibration'
                # score returns 2 while pending; successful collection launches
                # next stage automatically via the server-written jobs file.
                p=subprocess.run(['ssh','-o','BatchMode=yes',HOSTS[0],f'OMP_NUM_THREADS=4 python3 {ROOT}/experiment.py score --stage {stage}'],text=True,capture_output=True,timeout=180)
                print('SCORE',p.returncode,p.stdout[-1500:],p.stderr[-500:] if p.returncode not in [0,2] else '',flush=True)
                lastscore=time.time()
            pending=rpc('judges')
            if pending:
                results=[]
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures=[pool.submit(judge,j) for j in pending]
                    for f in futures:
                        try: results.append(f.result())
                        except Exception as e: print('JUDGE RETRY',repr(e),flush=True)
                if results: print('JUDGED',rpc('ingest',data=json.dumps(results)),flush=True)
            else: time.sleep(30)
        except Exception as e:
            print('ORCHESTRATOR RETRY',repr(e),flush=True);time.sleep(30)

if __name__=='__main__': main()
