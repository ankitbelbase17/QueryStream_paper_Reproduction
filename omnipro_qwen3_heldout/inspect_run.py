"""Small remote diagnostic summary; no tensors transferred to the laptop."""
from pathlib import Path
import collections
import json
import sys
import time
ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from control import status
from experiment import read,feature_path,key
print(json.dumps(status(),indent=2))
for s in read(ROOT/'data/subset.json'):
    if not feature_path(s['video_id']).exists():
        owner=ROOT/'claims/features'/key(s['video_id'])/'owner.json'
        print('MISSING FEATURE',s['id'],read(owner) if owner.exists() else 'unclaimed')
for p in sorted((ROOT/'errors').glob('*.json'))[:5]:
    print('ERROR',p.name,p.read_text()[-2500:])
for stage in ['calibration','test']:
    logs=sorted((ROOT/'logs').glob(stage+'_*'))
    for p in logs:
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-1400)); text=f.read().decode(errors='replace')
        print('LOG',p.name,text)
manifest=ROOT/'results/calibration_search.json'
if manifest.exists():
    data=json.loads(manifest.read_text())
    print('SHORTLISTS',json.dumps({t:v['shortlist'] for t,v in data.items()},indent=2))
frozen=ROOT/'results/frozen_thresholds.json'
if frozen.exists(): print('FROZEN',frozen.read_text())
