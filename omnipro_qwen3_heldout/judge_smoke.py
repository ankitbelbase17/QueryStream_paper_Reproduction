"""Enqueue an isolated official judge request to test the complete round trip."""
from pathlib import Path
import sys
import hashlib
ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from experiment import write
from metrics.llm_judge import _JUDGE_PROMPT
p=_JUDGE_PROMPT.format(question='Narrate what happens.',gt_response='A person opens a red door.',pred_response='A person opens a red door.')
k=hashlib.sha256(p.encode()).hexdigest()
write(ROOT/'judge_requests'/f'{k}.json',{'key':k,'prompt':p,'model':'github-copilot/gemini-3.8-flash','purpose':'plumbing_smoke_not_benchmark'})
write(ROOT/'validation/judge_smoke_key.json',{'key':k})
print(k)
