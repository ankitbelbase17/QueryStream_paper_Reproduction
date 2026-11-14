"""Independent hand-calculated cases for official scorer and split integrity."""
import sys
from pathlib import Path
ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from experiment import load_module,read,write,trigger_times
import torch
sc=load_module('qs_scorer_checks',ROOT/'official_omni/scripts/compute_online_metrics.py')
from evaluators.online_evaluator import _merge_continuation_fragments
from utils.prompts import render_observe_prompt
from utils.online_parser import is_standby

sample={'id':'synthetic','task':'snapshot_counting','question':'Count people when a door opens.',
        'ground_truth':[{'trigger_time_sec':10,'count':3},{'trigger_time_sec':30,'count':5}],
        'predictions':[{'t_sec':10,'raw':'3'},{'t_sec':29,'raw':'4'},{'t_sec':60,'raw':'7'}]}
r=sc.score_sample(sample,3.)['score']
assert r['tp_time']==2 and r['tp_content']==1
assert r['time_f1']==.8 and r['joint_f1']==.4,r
sample['task']='event_narration'
sample['ground_truth']=[{'trigger_time_sec':10,'response':'A person opens the door.'}]
sample['predictions']=[{'t_sec':13,'raw':'Someone opens a door.'}]
assert sc.score_sample(sample,3.,gpt_judge=lambda *a:{'score':3})['score']['joint_f1']==1.
assert sc.score_sample(sample,3.,gpt_judge=lambda *a:{'score':2})['score']['joint_f1']==0.
sample['predictions'][0]['t_sec']=13.01
assert sc.score_sample(sample,3.,gpt_judge=lambda *a:{'score':5})['score']['joint_f1']==0.
signal={'relevance':torch.tensor([.2,.3,.3,.1,.3]),'density':torch.tensor([.4,.4,.2,.4,.4])}
assert trigger_times(signal,.2,.2)==[1,4]  # strict >; consecutive positives group
assert _merge_continuation_fragments([{'t_sec':1,'raw':'a'},{'t_sec':3,'raw':'b'},{'t_sec':6,'raw':'c'}])==[{'t_sec':1,'raw':'a b'},{'t_sec':6,'raw':'c'}]
assert is_standby('')
cal=read(ROOT/'data/calibration.json');test=read(ROOT/'data/test.json')
assert len(cal)==182 and len(test)==750
assert {s['id'] for s in cal}.isdisjoint(s['id'] for s in test)
assert all(s['audio_dependency'] in ['none','helpful'] for s in cal+test)
# Official state list is a protocol input, not our fabricated open vocabulary.
state=next(s for s in cal if s['task']=='realtime_state_monitor')
prompt=render_observe_prompt(state['task'],state,8,[{'t_sec':5,'raw':'old_state'}])
assert 'old_state' in prompt and 'Possible states:' in prompt
write(ROOT/'validation/metric_checks.json',{'passed':True,'cases':['wrong-content-denominator','judge-pass-3','tolerance-boundary','strict-RTAR-gates','official-fragment-merging','disjoint-sample-split','official-history-state-vocab']})
print('METRIC CHECKS PASSED')
