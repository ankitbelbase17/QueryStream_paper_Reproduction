"""Independent final audit: completeness, provenance, recomputed metrics, CI.

Runs on the server. Recomputes every held-out task score from the raw saved
predictions + cached judge verdicts, without calling the scorer's aggregation.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0, str(ROOT))
from experiment import read, write, key  # noqa: E402  (also sets official paths)
from metrics.online.scorer import match_emits_to_gt, _score_content  # noqa: E402
from metrics.llm_judge import _JUDGE_PROMPT  # noqa: E402
from utils.online_parser import is_standby  # noqa: E402
from evaluators.online_evaluator import _merge_continuation_fragments  # noqa: E402

JUDGE = 'github-copilot/gemini-3.8-flash'
final = read(ROOT / 'results/FINAL_RESULTS.json')
jobs = read(ROOT / 'jobs_test.json')
cal = read(ROOT / 'data/calibration.json')
test = read(ROOT / 'data/test.json')
frozen = read(ROOT / 'results/frozen_thresholds.json')

assert final['complete'] is True and len(jobs) == 750 == len(test)
assert sorted(j['sample']['id'] for j in jobs) == sorted(s['id'] for s in test)
assert {s['id'] for s in test}.isdisjoint({s['id'] for s in cal})
assert all(s['audio_dependency'] in ('none', 'helpful') for s in test + cal)
assert final['thresholds'] == frozen

per_task = defaultdict(list)   # rows: (n_emits, n_gt, tp_joint, tp_time)
judge_keys, keep_rates = set(), []
n_calls = n_judged_pairs = n_judged_pass = 0
for j in jobs:
    s = j['sample']
    jid = key('test|' + j['policy'] + '|' + s['id'])
    pred = read(ROOT / 'predictions/test' / f'{jid}.json')
    assert pred['id'] == s['id'] and pred['ground_truth'] == s['ground_truth']
    assert pred['question'] == s['question'] and 'error' not in pred
    th = frozen[s['task']]
    assert pred['thresholds']['tau_rel'] == th['tau_rel']
    assert pred['thresholds']['tau_den'] == th['tau_den']
    assert pred['thresholds']['tau_temp'] == 0.9
    assert all(0 <= c['t_sec'] <= round(s['duration']) for c in pred['generation_calls'])
    assert all(1 <= c['n_tokens'] <= 512 for c in pred['generation_calls'])
    hist = [{'t_sec': c['t_sec'], 'raw': c['raw']} for c in pred['generation_calls']
            if not is_standby(c['raw'])]
    assert hist == pred['raw_history']
    assert _merge_continuation_fragments(hist) == pred['predictions']

    matches, _, _ = match_emits_to_gt(pred['predictions'], s['ground_truth'], 3.0)
    tp = 0
    for i, k, _ in matches:
        emit, gt = pred['predictions'][i], s['ground_truth'][k]
        ok = _score_content(s['task'], emit, gt, s['ground_truth'])
        if ok is None:  # open-ended: exact official judge prompt, cached verdict
            prompt = _JUDGE_PROMPT.format(question=s['question'],
                                          gt_response=gt['response'],
                                          pred_response=emit['raw'])
            h = hashlib.sha256(prompt.encode()).hexdigest()
            v = read(ROOT / 'judge_results' / f'{h}.json')
            assert v['key'] == h and v['model'] == JUDGE and 1 <= v['score'] <= 5
            judge_keys.add(h)
            ok = v['score'] >= 3
            n_judged_pairs += 1
            n_judged_pass += ok
        tp += bool(ok)
    per_task[s['task']].append((len(pred['predictions']), len(s['ground_truth']), tp, len(matches)))
    keep_rates.append(pred['token_keep_rate'])
    n_calls += len(pred['generation_calls'])


def f1(rows, col):
    p = sum(r[0] for r in rows); g = sum(r[1] for r in rows); t = sum(r[col] for r in rows)
    return 2 * t / (p + g) if p + g else 0.0


tasks = sorted(per_task)
assert len(tasks) == 9
metrics = {}
for t in tasks:
    rows = per_task[t]
    p = sum(r[0] for r in rows); g = sum(r[1] for r in rows)
    tpj = sum(r[2] for r in rows); tpt = sum(r[3] for r in rows)
    m = {'n': len(rows), 'emits': p, 'gt_triggers': g, 'tp_joint': tpj, 'tp_time': tpt,
         'joint_precision': tpj / p if p else 0.0, 'joint_recall': tpj / g if g else 0.0,
         'joint_f1': f1(rows, 2), 'time_f1': f1(rows, 3)}
    off = final['tasks'][t]
    assert m['n'] == off['n']
    assert abs(m['joint_f1'] - off['joint_f1']) <= 5.1e-5, (t, m['joint_f1'], off['joint_f1'])
    assert abs(m['time_f1'] - off['time_f1']) <= 5.1e-5, (t, m['time_f1'], off['time_f1'])
    metrics[t] = m

macro_joint = statistics.fmean(m['joint_f1'] for m in metrics.values())
macro_time = statistics.fmean(m['time_f1'] for m in metrics.values())
assert abs(macro_joint - final['macro']['macro_joint_f1']) < 1.1e-4
allrows = [r for t in tasks for r in per_task[t]]
micro_joint = f1(allrows, 2)

# Nonparametric bootstrap over held-out samples, stratified by task.
rng = random.Random(42)
boot = []
for _ in range(2000):
    vals = []
    for t in tasks:
        rows = per_task[t]
        vals.append(f1([rows[rng.randrange(len(rows))] for _ in rows], 2))
    boot.append(statistics.fmean(vals))
boot.sort()

verdicts = [read(p) for p in (ROOT / 'judge_results').glob('*.json')]
audit = {
    'passed': True,
    'test_samples': len(jobs),
    'calibration_samples': len(cal),
    'tasks': len(tasks),
    'macro_joint_f1': macro_joint,
    'macro_joint_f1_bootstrap95': [boot[49], boot[1949]],
    'bootstrap': {'reps': 2000, 'seed': 42, 'unit': 'held-out sample, stratified by task'},
    'macro_time_f1': macro_time,
    'micro_joint_f1': micro_joint,
    'per_task': metrics,
    'mean_sample_token_keep_rate': statistics.fmean(keep_rates),
    'generation_calls': n_calls,
    'judged_test_pairs': n_judged_pairs,
    'judged_test_pass_rate': n_judged_pass / n_judged_pairs if n_judged_pairs else None,
    'unique_test_judge_prompts': len(judge_keys),
    'all_cached_verdicts_incl_calibration_and_smoke': len(verdicts),
    'judge_model': JUDGE,
    'judge_reported_cost_total': sum(v.get('reported_cost', 0) for v in verdicts),
    'test_id_sha256': hashlib.sha256('\n'.join(sorted(s['id'] for s in test)).encode()).hexdigest(),
}
write(ROOT / 'results/FINAL_AUDIT.json', audit)
print(json.dumps(audit, indent=2))
