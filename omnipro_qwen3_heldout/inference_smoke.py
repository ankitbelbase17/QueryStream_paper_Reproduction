"""Real-video end-to-end smoke, isolated from calibration/test artifacts."""
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from experiment import read,write,feature_path,infer,key
samples=read(ROOT/'data/subset.json')
s=min((s for s in samples if feature_path(s['video_id']).exists()),key=lambda s:s['duration'])
write(ROOT/'jobs_smoke.json',[{'sample':s,'stage':'smoke','policy':'smoke','tau_rel':-1.,'tau_den':0.}])
infer(SimpleNamespace(stage='smoke',once=True))
p=ROOT/'predictions/smoke'/f"{key('smoke|smoke|'+s['id'])}.json"
result=read(p)
assert result['generation_calls']
write(ROOT/'validation/inference_smoke.json',{'passed':True,'sample':s['id'],'calls':result['generation_calls'],'predictions':result['predictions'],'seconds':result['wall_sec']})
print('END TO END SMOKE PASSED',result['generation_calls'],flush=True)
