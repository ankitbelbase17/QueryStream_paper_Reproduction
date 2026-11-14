"""Sanity-inspect raw held-out outputs (small text summary only)."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0, str(ROOT))
from experiment import read, key  # noqa: E402
from utils.online_parser import is_standby, parse_streaming_output  # noqa: E402

jobs = read(ROOT / 'jobs_test.json')
by_task = defaultdict(list)
for j in jobs:
    s = j['sample']
    by_task[s['task']].append(read(ROOT / 'predictions/test' / f"{key('test|' + j['policy'] + '|' + s['id'])}.json"))

rng = random.Random(7)
summary = {}
for task, preds in sorted(by_task.items()):
    calls = [c for p in preds for c in p['generation_calls']]
    toks = [c['n_tokens'] for c in calls]
    standby = sum(is_standby(c['raw']) for c in calls)
    parsed_valid = None
    if task in ('explicit_target_grounding', 'snapshot_counting', 'cumulative_counting',
                'dedup_counting', 'realtime_state_monitor'):
        ems = [e for p in preds for e in p['predictions']]
        parsed_valid = sum(parse_streaming_output(e['raw'], task).get('valid', False) for e in ems) / max(len(ems), 1)
    summary[task] = {
        'samples': len(preds), 'calls': len(calls), 'standby_frac': round(standby / max(len(calls), 1), 3),
        'mean_new_tokens': round(statistics.fmean(toks), 1) if toks else 0, 'hit_512': sum(t >= 512 for t in toks),
        'samples_without_emits': sum(not p['predictions'] for p in preds),
        'parsed_valid_frac': None if parsed_valid is None else round(parsed_valid, 3),
        'mean_keep_rate': round(statistics.fmean(p['token_keep_rate'] for p in preds), 3),
    }
    print(f'===== {task} =====')
    for p in rng.sample(preds, 2):
        gts = [(g['trigger_time_sec'], g.get('count', g.get('state_to', g.get('position', g['response'][:70]))))
               for g in p['ground_truth']]
        ems = [(e['t_sec'], e['raw'][:90]) for e in p['predictions'][:5]]
        print('Q:', p['question'][:150])
        print('  GT:', gts[:5])
        print('  EMITS (first 5 of %d):' % len(p['predictions']), ems)
print(json.dumps(summary, indent=1))
