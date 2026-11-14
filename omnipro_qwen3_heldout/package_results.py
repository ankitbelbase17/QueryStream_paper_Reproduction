"""Bundle code, results, validation and compact artifacts for the repository PR.

Runs on the server; produces pr_bundle.tgz (a few MB). Feature caches (~479 GB)
and model weights are deliberately excluded.
"""
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
OUT = ROOT / 'pr_bundle' / 'omnipro_qwen3_heldout'
if OUT.exists():
    shutil.rmtree(OUT)
for d in ['results', 'splits', 'validation', 'logs']:
    (OUT / d).mkdir(parents=True)

code = ['setup_assets.py', 'experiment.py', 'validate.py', 'metric_checks.py',
        'inference_smoke.py', 'judge_smoke.py', 'control.py', 'worker.sh',
        'final_audit.py', 'inspect_run.py', 'inspect_outputs.py', 'wait_status.py',
        'verify_findings.py', 'package_results.py']
for c in code:
    shutil.copy2(ROOT / c, OUT / c)
shutil.copy2(ROOT / 'PROTOCOL.md', OUT / 'PROTOCOL.md')
for r in ['FINAL_RESULTS.json', 'FINAL_AUDIT.json', 'RESULTS.md', 'frozen_thresholds.json',
          'calibration_search.json', 'calibration_progress.json', 'test_progress.json']:
    shutil.copy2(ROOT / 'results' / r, OUT / 'results' / r)
for v in ['qwen3_equivalence.json', 'metric_checks.json', 'inference_smoke.json',
          'findings_checks.json']:
    shutil.copy2(ROOT / 'validation' / v, OUT / 'validation' / v)

shutil.copy2(ROOT / 'data/split_manifest.json', OUT / 'splits/split_manifest.json')
for name in ['calibration', 'test']:
    ids = sorted(s['id'] for s in json.loads((ROOT / 'data' / f'{name}.json').read_text()))
    (OUT / 'splits' / f'{name}_ids.txt').write_text('\n'.join(ids) + '\n')


def dump_jsonl(rows, path):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


for stage, dest in [('calibration', OUT / 'results/predictions_calibration.jsonl'),
                    ('test', OUT / 'results/predictions_test.jsonl'),
                    ('smoke', OUT / 'validation/inference_smoke_prediction.jsonl')]:
    rows = [json.loads(p.read_text()) for p in (ROOT / 'predictions' / stage).glob('*.json')]
    rows.sort(key=lambda r: (r.get('policy', ''), r['id']))
    dump_jsonl(rows, dest)

verdicts = []
for p in sorted((ROOT / 'judge_requests').glob('*.json')):
    req = json.loads(p.read_text())
    res = json.loads((ROOT / 'judge_results' / p.name).read_text())
    assert req['key'] == res['key'] == p.stem
    verdicts.append({'key': req['key'], 'purpose': req.get('purpose', 'benchmark'),
                     'prompt': req['prompt'], **{k: v for k, v in res.items() if k != 'key'}})
assert len(verdicts) == len(list((ROOT / 'judge_results').glob('*.json')))
dump_jsonl(verdicts, OUT / 'results/judge_verdicts.jsonl')

dump_jsonl(sorted((json.loads(p.read_text()) for p in (ROOT / 'logs').glob('feature_*.json')),
                  key=lambda r: r['video']), OUT / 'results/feature_timings.jsonl')
dump_jsonl([{'file': p.name, **json.loads(p.read_text())}
            for p in sorted((ROOT / 'errors').glob('*.json'))], OUT / 'logs/errors.jsonl')

keep = ('INFER', 'FEATURE', 'END TO END', 'METRIC', 'SCORE', 'CALIBRATION JOBS', 'Traceback',
        'Error', 'error')
with open(OUT / 'logs/worker_events.log', 'w') as f:
    for p in sorted((ROOT / 'logs').glob('*.log')):
        for line in p.read_text(errors='replace').splitlines():
            if line.startswith(keep) or 'Error' in line:
                f.write(f'{p.name}\t{line}\n')

with tarfile.open(ROOT / 'pr_bundle.tgz', 'w:gz') as tar:
    tar.add(OUT, arcname='omnipro_qwen3_heldout')
print('bundle', (ROOT / 'pr_bundle.tgz').stat().st_size, 'bytes')
