"""Server-only asset preparation and reproducible split; no inference."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, snapshot_download, HfApi

ROOT = Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
os.environ['HF_HOME'] = str(ROOT / 'hf')
for d in ['data', 'logs', 'features', 'predictions', 'judge_requests', 'judge_results', 'results', 'validation']:
    (ROOT / d).mkdir(exist_ok=True)
p = hf_hub_download('RuixiangZhao/OmniPro', 'benchmark.json', repo_type='dataset', local_dir=ROOT/'data')
all_samples = json.loads(Path(p).read_text())
samples = [s for s in all_samples if s['audio_dependency'] in ['none', 'helpful']]
groups = collections.defaultdict(list)
for s in samples:
    groups[s['task']].append(s)
cal, test, stats = [], [], {}
for task in sorted(groups):
    rows = sorted(groups[task], key=lambda s:s['id'])
    rng = random.Random(42)
    rng.shuffle(rows)
    n = max(1, math.floor(len(rows)*.2))
    cal.extend(rows[:n]); test.extend(rows[n:])
    stats[task] = {'total':len(rows), 'calibration':n, 'test':len(rows)-n}
assert set(s['id'] for s in cal).isdisjoint(s['id'] for s in test)
for name, rows in [('calibration',cal),('test',test),('subset',samples)]:
    (ROOT/'data'/f'{name}.json').write_text(json.dumps(rows, indent=2))
video_roots = [Path('/iopsstor/scratch/cscs/dbartaula')/p for p in ['omnipro_data','omni_pro/dataset']]
def get_video(s):
    dest = ROOT/'data'/s['video_path']
    dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.exists(): return 'existing'
    for base in video_roots:
        src = base/s['video_path']
        if src.exists() and src.stat().st_size:
            dest.symlink_to(src)
            return 'linked'
    hf_hub_download('RuixiangZhao/OmniPro', s['video_path'], repo_type='dataset', local_dir=ROOT/'data')
    return 'downloaded'
unique = {s['video_id']:s for s in samples}
with ThreadPoolExecutor(max_workers=12) as pool:
    availability = dict(collections.Counter(pool.map(get_video,unique.values())))
manifest = {'seed':42,'split_unit':'sample within task','rounding':'floor(0.2*n), minimum 1',
            'tasks':stats,'calibration':len(cal),'test':len(test),'subset':len(samples),
            'shared_videos_across_splits':len(set(s['video_id'] for s in cal)&set(s['video_id'] for s in test)),
            'benchmark_sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            'videos':availability,'commits':{}}
for repo in ['reproduction','official_query','official_omni']:
    manifest['commits'][repo] = subprocess.check_output(['git','-C',str(ROOT/repo),'rev-parse','HEAD'],text=True).strip()
(ROOT/'data'/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
# Reuse complete existing model cache; resolve a pinned snapshot, downloading only missing assets.
cache='/iopsstor/scratch/cscs/dbartaula/hf_cache/hub'
model=snapshot_download('Qwen/Qwen3-VL-8B-Instruct',cache_dir=cache)
(ROOT/'data'/'model_path.txt').write_text(model)
print('MODEL', model,flush=True)
