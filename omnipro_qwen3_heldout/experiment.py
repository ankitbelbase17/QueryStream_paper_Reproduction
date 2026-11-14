"""Audited QueryStream/OmniPro experiment. All heavy work runs on scratch.

Subcommands: features, shortlist, infer, score, status. Persistent task queue
uses atomic mkdir claims, one worker per GPU. No held-out labels used to fit.
"""
import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ.get('QS_ROOT','/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114'))
sys.path.insert(0,str(ROOT/'reproduction'))
sys.path.insert(0,str(ROOT/'official_omni'))

def read(p): return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)
def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return mod
def official_functions():
    # Execute only pure published QDP functions; avoid the old Qwen2.5 imports.
    import torch
    import torch.nn.functional as F
    names={'_overlap_weights','_pool_clip_features_to_qwen_grid','_make_qdp_masks_from_aligned_features'}
    source=ROOT/'official_query/querystream/demo/infer.py'
    tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    ns={'torch':torch,'F':F,'math':math,'Tuple':__import__('typing').Tuple,'List':__import__('typing').List}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    # Overlap geometry is constant per video. Memoize the unmodified official
    # function's output to avoid rebuilding thousands of CUDA scalar entries
    # for every frame. Feature arithmetic and masks remain unchanged.
    original=ns['_overlap_weights']; cache={}
    def cached_overlap(source_hw,target_hw,device,dtype):
        k=(source_hw,target_hw,str(device),dtype)
        if k not in cache: cache[k]=original(source_hw,target_hw,device,dtype)
        return cache[k]
    ns['_overlap_weights']=cached_overlap
    return ns

def setup_torch():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch

def feature_path(vid): return ROOT/'features'/f'{key(vid)}.pt'

def claim(stage,k):
    p=ROOT/'claims'/stage/k
    p.parent.mkdir(parents=True,exist_ok=True)
    try:
        p.mkdir(); write(p/'owner.json',{'pid':os.getpid(),'host':os.uname().nodename,'time':time.time()})
        return True
    except FileExistsError: return False

def sample_frames(sample):
    """Exact official index/tick policy, OpenCV decoder for ARM (no decord wheel)."""
    import cv2
    import numpy as np
    import torch
    cap=cv2.VideoCapture(str(ROOT/'data'/sample['video_path']))
    if not cap.isOpened(): raise RuntimeError('Cannot open '+sample['video_path'])
    fps=cap.get(cv2.CAP_PROP_FPS) or 1.
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ticks=range(int(round(float(sample['duration'])))+1)
    indices=[max(0,min(int(round(t*fps)),total-1)) for t in ticks]
    desired=set(indices); found={}; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx in desired: found[idx]=frame[:,:,::-1].copy()
        idx+=1
    cap.release()
    if any(i not in found for i in indices): raise RuntimeError('Incomplete video decode')
    return torch.from_numpy(np.stack([found[i] for i in indices])).permute(0,3,1,2).contiguous(),indices,fps

def get_qwen():
    import torch
    from querystream.config import BackboneConfig
    from querystream.qwen3_wrapper import Qwen3VLWrapper

    class AuditedQwen(Qwen3VLWrapper):
        def prepare_context(self,n,hw,masks):
            self.ctx_n=n; self.ctx_hw=hw; self.ctx_masks=masks

        def build_prompt_ids(self,query,plane_counts,system=None,fps=None):
            # Build dense prompt, obtain positions from upstream, THEN slice.
            # This preserves absolute positions even when an entire frame drops.
            tok=self.processor.tokenizer
            messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':query}]}]
            text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            m=self.ctx_hw[0]*self.ctx_hw[1]
            blocks=''.join(f'<{j:.1f} seconds>{self.vision_start_token}{self.video_token*m}{self.vision_end_token}' for j in range(self.ctx_n))
            triple=f'{self.vision_start_token}{self.video_token}{self.vision_end_token}'
            assert triple in text
            text=text.replace(triple,blocks,1)
            dense=tok(text,return_tensors='pt').input_ids.to(self.device)
            types=torch.zeros_like(dense); types[dense==self.video_pad_id]=2
            grids=torch.tensor([[1,self.ctx_hw[0]*2,self.ctx_hw[1]*2]]*self.ctx_n,device=self.device)
            pos,_=self.model.model.get_rope_index(dense,mm_token_type_ids=types,video_grid_thw=grids)
            keep=torch.ones(dense.shape[1],dtype=torch.bool,device=self.device)
            keep[dense[0]==self.video_pad_id]=self.ctx_masks.flatten().to(self.device)
            self.sparse_pos=pos[:,:,keep]
            sparse=dense[:,keep]
            assert int((sparse==self.video_pad_id).sum())==sum(n for _,n in plane_counts)
            self.last_dense_ids=dense; self.last_dense_pos=pos; self.last_keep=keep
            return sparse

        def build_position_ids(self,input_ids,mrope,dense_llm_grid_hw):
            return self.sparse_pos,int(self.sparse_pos.max())+1-input_ids.shape[1]

    model_path=(ROOT/'data/model_path.txt').read_text().strip()
    return AuditedQwen(BackboneConfig(model_id=model_path),device='cuda:0',dtype=torch.bfloat16,
                       attn_implementation='sdpa',max_pixels=128*32*32,fps=1.)

def features(args):
    torch=setup_torch()
    from querystream.open_clip_wrapper import OpenCLIPWrapper
    from querystream.config import EncoderConfig
    from PIL import Image
    funcs=official_functions()
    clip=OpenCLIPWrapper(EncoderConfig(device='cuda:0',dtype=torch.float32))
    qwen=get_qwen()
    samples=read(ROOT/'data/subset.json'); groups=defaultdict(list)
    for s in samples: groups[s['video_id']].append(s)
    # Calibration videos first. Within groups process longest first for load balance.
    calvid={s['video_id'] for s in read(ROOT/'data/calibration.json')}
    vids=sorted(groups,key=lambda v:(v not in calvid,-groups[v][0]['duration'],v))
    if args.once:
        vids=sorted(vids,key=lambda v:groups[v][0]['duration'])
    if getattr(args,'video',None): vids=[args.video]
    for vid in vids:
        out=feature_path(vid)
        if out.exists() or not claim('features',key(vid)): continue
        start=time.time()
        try:
            rows=groups[vid]; frames,indices,native_fps=sample_frames(rows[0])
            n=frames.shape[0]; tokens=[]; rawpatch=[]; hw=None
            for j in range(0,n,8):
                batch=frames[j:j+8]
                # Each 1-fps frame is a causal image observation. Duplicate only
                # the SAME frame to satisfy Qwen's temporal patch size of two.
                chunk=qwen.encode_chunk(batch.repeat_interleave(2,dim=0))
                if hw is None: hw=chunk.llm_grid_hw
                assert hw==chunk.llm_grid_hw
                tokens.append(chunk.tokens.reshape(len(batch),hw[0]*hw[1],-1).cpu())
                # Released CLIP PIL preprocessing, not reproduction's distortion resize.
                images=[Image.fromarray(f.permute(1,2,0).numpy()) for f in batch]
                pixels=torch.stack([clip.preprocess(im) for im in images]).cuda()
                patches,_=clip._patch_tokens(pixels)
                rawpatch.append(patches)
            patches=torch.cat(rawpatch)
            aligned=torch.stack([funcs['_pool_clip_features_to_qwen_grid'](torch.nn.functional.normalize(p,dim=-1),hw) for p in patches])
            # Official QDP implementation normalizes aligned vectors before EMA.
            aligned=torch.nn.functional.normalize(aligned,dim=-1)
            simtemp=[]; hist=None
            for f in aligned:
                simtemp.append(torch.ones(f.shape[0],device=f.device) if hist is None else (f*torch.nn.functional.normalize(hist,dim=-1)).sum(-1))
                hist=f.clone() if hist is None else .1*f+.9*hist
            temp=torch.stack(simtemp)
            signals={}
            for s in rows:
                query=clip.encode_text(s['question'])
                sem=aligned@query
                masks=(sem>sem.mean(-1,keepdim=True))&(temp<.9)
                masks[0]=True  # official demo's explicit first-frame policy
                rel=torch.nn.functional.normalize(aligned.mean(1),dim=-1)@query
                # Feed repeated feature rows to unmodified official pair-selection
                # function so each causal 1fps observation is checked once.
                ref=funcs['_make_qdp_masks_from_aligned_features'](aligned.repeat_interleave(2,0),query,n,.1,.9)
                refmask=torch.stack(ref[0])
                # Near a strict mean threshold, vectorized GEMM and the
                # published per-frame matvec can differ by a float32 ULP.
                # Use the published function's decisions as the source of
                # truth; do not loosen comparisons or tolerate changed masks.
                differing=int((masks.cpu()!=refmask).sum())
                signals[s['id']]={'mask':refmask,'relevance':torch.tensor(ref[2]),
                                  'density':refmask.float().mean(-1),
                                  'vectorized_reference_disagreements':differing}
            payload={'video_id':vid,'tokens':torch.cat(tokens),'hw':hw,'signals':signals,
                     'indices':indices,'native_fps':native_fps,'n_frames':n,'wall_sec':time.time()-start}
            tmp=out.with_suffix(f'.{os.getpid()}.tmp'); torch.save(payload,tmp); tmp.replace(out)
            write(ROOT/'logs'/f'feature_{key(vid)}.json',{'video':vid,'frames':n,'samples':len(rows),'seconds':time.time()-start,'bytes':out.stat().st_size})
            print('FEATURE',vid,n,round(time.time()-start,1),flush=True)
            del payload,tokens,rawpatch,patches,aligned,signals,frames
            torch.cuda.empty_cache()
        except Exception:
            write(ROOT/'errors'/f'features_{key(vid)}.json',{'traceback':traceback.format_exc(),'video':vid})
            print(traceback.format_exc(),flush=True)
            if args.once: raise
        if args.once: break

def trigger_times(signal,rel,den):
    # Paper A.4: one response at the first frame of each contiguous true run.
    # No metric-dependent three-second debouncer.
    r=signal['relevance']; d=signal['density']
    flags=(r>rel)&(d>den)
    starts=flags.clone()
    starts[1:] &= ~flags[:-1]
    return starts.nonzero(as_tuple=True)[0].tolist()

def shortlist(args):
    import torch
    from metrics.online.scorer import match_emits_to_gt
    cal=read(ROOT/'data/calibration.json'); groups=defaultdict(list)
    for s in cal: groups[s['task']].append(s)
    alljobs=[]; tables={}
    for task,rows in sorted(groups.items()):
        signals={}
        for s in rows:
            p=feature_path(s['video_id'])
            if not p.exists(): raise RuntimeError('Calibration features incomplete: '+s['id'])
            data=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            signals[s['id']]=data['signals'][s['id']]
        # Calibration-only feature quantiles give raw cosine thresholds with
        # sensible support, plus literal paper settings as an explicit control.
        vals=torch.cat([z['relevance'] for z in signals.values()])
        rels=sorted(set([round(float(torch.quantile(vals,q)),6) for q in torch.linspace(0,1,17)]+[.5,.6,.7]))
        dens=[0.,.05,.1,.15,.2,.25,.3,.4,.5,.6]
        ranked=[]
        for r in rels:
            for d in dens:
                tp=npred=ngt=0; timeline=[]
                for s in rows:
                    ts=trigger_times(signals[s['id']],r,d)
                    # Mimic official continuation merging for timing shortlist.
                    merged=[]; last=None
                    for t in ts:
                        if last is None or t-last>2: merged.append({'t_sec':t})
                        last=t
                    matches,_,_=match_emits_to_gt(merged,s['ground_truth'],3.)
                    tp+=len(matches); npred+=len(merged); ngt+=len(s['ground_truth'])
                    timeline.append(ts)
                ranked.append({'tau_rel':r,'tau_den':d,'time_f1':2*tp/(npred+ngt) if npred+ngt else 0.,'n_emits':npred,'signature':key(json.dumps(timeline))})
        ranked.sort(key=lambda x:(-x['time_f1'],x['n_emits'],x['tau_rel'],x['tau_den']))
        chosen=[]; seen=set()
        for cfg in ranked:
            if cfg['signature'] in seen: continue
            seen.add(cfg['signature']); chosen.append(cfg)
            if len(chosen)==4: break
        # Fit final thresholds using full content+timing F1 over these four
        # timing-shortlisted policies; NEVER filter generated responses post hoc.
        tables[task]={'search':ranked,'shortlist':chosen,'objective':'calibration joint F1 after timing shortlist of four distinct policies'}
        for i,cfg in enumerate(chosen):
            for s in rows:
                alljobs.append({'sample':s,'stage':'calibration','policy':f'{task}_{i}',**cfg})
    write(ROOT/'results/calibration_search.json',tables)
    write(ROOT/'jobs_calibration.json',alljobs)
    print('CALIBRATION JOBS',len(alljobs),flush=True)

def infer(args):
    torch=setup_torch()
    from utils.prompts import render_observe_prompt
    from utils.online_parser import is_standby
    from evaluators.online_evaluator import _merge_continuation_fragments
    qwen=get_qwen()
    jobs=read(ROOT/f'jobs_{args.stage}.json')
    jobs.sort(key=lambda x:(x['sample']['video_id'],x['policy'],x['sample']['id']))
    lastvid=None; data=None
    for job in jobs:
        s=job['sample']; jid=key(args.stage+'|'+job['policy']+'|'+s['id'])
        out=ROOT/'predictions'/args.stage/f'{jid}.json'
        if out.exists() or not claim(args.stage,jid): continue
        t0=time.time()
        try:
            if s['video_id']!=lastvid:
                data=torch.load(feature_path(s['video_id']),map_location='cpu',weights_only=False,mmap=True)
                lastvid=s['video_id']
            signal=data['signals'][s['id']]
            times=trigger_times(signal,job['tau_rel'],job['tau_den'])
            history=[]; calls=[]; hw=data['hw']; m=hw[0]*hw[1]
            for t in times:
                mask=signal['mask'][:t+1]
                n=t+1
                coords=torch.stack([torch.arange(n).repeat_interleave(m),torch.arange(hw[0]).repeat_interleave(hw[1]).repeat(n),torch.arange(hw[1]).repeat(hw[0]*n)],-1)
                tokens=data['tokens'][:n].reshape(-1,data['tokens'].shape[-1])[mask.flatten()].cuda()
                coords=coords[mask.flatten()].cuda()
                assert tokens.shape[0]>0
                qwen.prepare_context(n,hw,mask)
                prompt=render_observe_prompt(s['task'],s,t,history)
                g=qwen.generate_from_pruned(query=prompt,visual_tokens=tokens,mrope=coords,dense_llm_grid_hw=hw,max_new_tokens=512,do_sample=False,profile_vram=False)
                calls.append({'t_sec':t,'raw':g.text,'n_tokens':g.num_new_tokens,'kept_visual':g.num_visual_tokens})
                if not is_standby(g.text): history.append({'t_sec':t,'raw':g.text})
                del tokens,coords
            pred={**s,'predictions':_merge_continuation_fragments(history),'raw_history':history,'generation_calls':calls,
                  'model':'Qwen/Qwen3-VL-8B-Instruct','eval_mode':'online','policy':job['policy'],'stage':args.stage,
                  'thresholds':{'tau_rel':job['tau_rel'],'tau_den':job['tau_den'],'tau_temp':.9},
                  'token_keep_rate':float(signal['mask'].float().mean()),'wall_sec':time.time()-t0}
            write(out,pred)
            print('INFER',args.stage,s['id'],len(times),len(pred['predictions']),round(time.time()-t0,1),flush=True)
            torch.cuda.empty_cache()
        except Exception:
            write(ROOT/'errors'/f'{args.stage}_{jid}.json',{'job':job,'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
            if args.once: raise
        if args.once: break

def score(args):
    from metrics.llm_judge import _JUDGE_PROMPT
    from metrics.online.scorer import match_emits_to_gt
    scorer=load_module('qs_official_scorer',ROOT/'official_omni/scripts/compute_online_metrics.py')
    jobs=read(ROOT/f'jobs_{args.stage}.json')
    grouped=defaultdict(list); missing=[]; unjudged=0
    for job in jobs:
        jid=key(args.stage+'|'+job['policy']+'|'+job['sample']['id'])
        path=ROOT/'predictions'/args.stage/f'{jid}.json'
        if not path.exists(): missing.append(jid); continue
        pred=read(path)
        def judge(question,gt,answer):
            nonlocal unjudged
            prompt=_JUDGE_PROMPT.format(question=question,gt_response=gt,pred_response=answer)
            jkey=hashlib.sha256(prompt.encode()).hexdigest()
            result=ROOT/'judge_results'/f'{jkey}.json'
            if result.exists():
                got=read(result)
                assert isinstance(got['score'],int) and 1<=got['score']<=5
                return got
            write(ROOT/'judge_requests'/f'{jkey}.json',{'key':jkey,'prompt':prompt,'model':'github-copilot/gemini-3.8-flash'})
            unjudged+=1
            raise LookupError(jkey)
        try: scored=scorer.score_sample(pred,3.,gpt_judge=judge,gpt_pass_score=3)
        except LookupError: continue
        grouped[job['policy']].append(scored)
    # Enumerate all missing judge pairs in a separate pass, not one per sample.
    for path in (ROOT/'predictions'/args.stage).glob('*.json'):
        pred=read(path)
        if pred['task'] not in ['event_narration','sequential_step_instruction']: continue
        matches,_,_=match_emits_to_gt(pred['predictions'],pred['ground_truth'],3.)
        for i,j,_ in matches:
            prompt=_JUDGE_PROMPT.format(question=pred['question'],gt_response=pred['ground_truth'][j]['response'],pred_response=pred['predictions'][i]['raw'])
            k=hashlib.sha256(prompt.encode()).hexdigest()
            if not (ROOT/'judge_results'/f'{k}.json').exists():
                write(ROOT/'judge_requests'/f'{k}.json',{'key':k,'prompt':prompt,'model':'github-copilot/gemini-3.8-flash'})
    report={'stage':args.stage,'expected':len(jobs),'missing_predictions':len(missing),'unjudged_samples':unjudged,'policies':{}}
    for policy,rows in grouped.items():
        report['policies'][policy]=scorer._aggregate_samples(rows)
    write(ROOT/'results'/f'{args.stage}_progress.json',report)
    if missing or unjudged:
        print(json.dumps({'stage':args.stage,'expected':len(jobs),'missing':len(missing),'unjudged':unjudged}),flush=True)
        return False
    if args.stage=='calibration':
        bytask=defaultdict(list)
        for policy,rows in grouped.items():
            # No judge needed if there were zero time-matches: joint F1 is zero.
            metric=report['policies'][policy]
            if metric['joint_f1'] is None and metric['tp_time']==0: metric['joint_f1']=0.
            assert metric['joint_f1'] is not None
            cfg=next(j for j in jobs if j['policy']==policy)
            bytask[rows[0]['task']].append((metric['joint_f1'],policy,cfg,metric))
        selected={}
        for task,opts in bytask.items():
            opts.sort(key=lambda x:(-x[0],x[1]))
            f1,policy,cfg,metric=opts[0]
            selected[task]={'tau_rel':cfg['tau_rel'],'tau_den':cfg['tau_den'],'policy':policy,'calibration_metrics':metric}
        write(ROOT/'results/frozen_thresholds.json',selected)
        testjobs=[{'sample':s,'stage':'test',**selected[s['task']]} for s in read(ROOT/'data/test.json')]
        write(ROOT/'jobs_test.json',testjobs)
    else:
        taskmetrics={rows[0]['task']:report['policies'][policy] for policy,rows in grouped.items()}
        for metric in taskmetrics.values():
            if metric['joint_f1'] is None and metric['tp_time']==0: metric['joint_f1']=0.
        assert all(m['joint_f1'] is not None for m in taskmetrics.values())
        final={'tasks':taskmetrics,'macro':scorer._macro_overall(taskmetrics),'n_samples':len(jobs),
               'split':read(ROOT/'data/split_manifest.json'),'thresholds':read(ROOT/'results/frozen_thresholds.json'),
               'judge':'github-copilot/gemini-3.8-flash','tolerance':3.,'complete':True}
        write(ROOT/'results/FINAL_RESULTS.json',final)
        lines=['# QueryStream + Qwen3-VL-8B: OmniPro Online','',f"Held-out samples: {len(jobs)}. Video-only; none/helpful subset; seed 42.",'',
               '|Task|N|Time F1 (%)|Joint F1 (%)|','|---|---:|---:|---:|']
        for task,m in taskmetrics.items(): lines.append(f"|{task}|{m['n']}|{100*m['time_f1']:.2f}|{100*m['joint_f1']:.2f}|")
        lines+=['',f"**Macro joint F1: {100*final['macro']['macro_joint_f1']:.2f}%**",'',
                'Adapted implementation: official QDP mathematics + reproduction Qwen3 wrapper with audited position slicing. See PROTOCOL.md for complete configuration and differences.']
        (ROOT/'results/RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('SCORE COMPLETE',args.stage,flush=True)
    return True

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['features','shortlist','infer','score']); p.add_argument('--stage',default='calibration',choices=['calibration','test']); p.add_argument('--once',action='store_true'); p.add_argument('--video')
    a=p.parse_args()
    if a.command=='features': features(a)
    elif a.command=='shortlist': shortlist(a)
    elif a.command=='infer': infer(a)
    elif a.command=='score': sys.exit(0 if score(a) else 2)
if __name__=='__main__': main()
