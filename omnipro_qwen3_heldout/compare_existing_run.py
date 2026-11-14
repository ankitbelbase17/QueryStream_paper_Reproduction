"""Context only: score the repo's existing Qwen3 OmniPro run (runs/op_qwen3) on
exactly this experiment's 750 held-out sample IDs, with the repo's own scorer
and its own (OpenAI) judge verdicts. Different prompts, judge, gating and
thresholds -- this is NOT a controlled comparison of methods.
"""
import json
from pathlib import Path
import sys

ROOT = Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
REPO = ROOT / 'reproduction'
sys.path.insert(0, str(REPO))
import score_omnipro as S  # noqa: E402
from omnipro_eval import judge as J, metrics as M  # noqa: E402

test_ids = set((ROOT / 'pr_bundle/omnipro_qwen3_heldout/splits/test_ids.txt').read_text().split())
records = S.read_records(str(REPO / 'runs/op_qwen3'))
ok = [r for r in records if not r.get('error') and r['sample_id'] in test_ids]
assert len(ok) == len(test_ids) == 750, len(ok)
verdicts = S.load_verdicts(str(REPO / 'runs/op_qwen3/judge_output.jsonl'))
J.attach_verdicts(ok, verdicts)
judge = J.CachedJudge(verdicts, pass_score=3)
agg = M.aggregate([M.score_sample(r, tolerance=3.0, judge=judge) for r in ok])
o = agg['overall']
out = {
    'source': 'runs/op_qwen3 (repo main @ 87391e5), restricted to splits/test_ids.txt',
    'n_samples': len(ok),
    'macro_joint_f1': o['macro_joint_f1'], 'macro_time_f1': o['macro_time_f1'],
    'n_unjudged': o.get('n_unjudged'),
    'emits': o['n_emits'], 'gt': o['n_gt'],
    'per_task_joint_f1': {t: b['joint_f1'] for t, b in sorted(agg['per_task'].items())},
    'per_task_time_f1': {t: b['time_f1'] for t, b in sorted(agg['per_task'].items())},
    'differences': ['repo prompts (event asserted, no history/silence option, 8-80 token caps, '
                    'no state vocabulary)', 'running-minmax relevance, tau_rel=0.60, '
                    'tau_den=0.15 (untuned)', '3 s debounce', 'OpenAI judge',
                    '2 distinct frames per Qwen3 plane', 'supplied Qwen3 wrapper (whole-plane '
                    'M-RoPE shift)', 'repo scorer (reimplementation of the official one)'],
}
(ROOT / 'pr_bundle/omnipro_qwen3_heldout/results/context_existing_op_qwen3_on_heldout.json'
 ).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
