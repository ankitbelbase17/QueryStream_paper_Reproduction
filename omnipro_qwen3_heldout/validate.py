"""Run substantive GPU equivalence checks before the full experiment."""
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from experiment import get_qwen,setup_torch,write
setup_torch()
q=get_qwen()
torch.manual_seed(42)
frames=torch.randint(0,256,(3,3,224,320),dtype=torch.uint8)
paired=frames.repeat_interleave(2,0)
chunk=q.encode_chunk(paired)
hw=chunk.llm_grid_hw; m=hw[0]*hw[1]
allmask=torch.ones(3,m,dtype=torch.bool)
q.prepare_context(3,hw,allmask)
query='Describe the visible colors briefly.'
ids=q.build_prompt_ids(query,[(i,m) for i in range(3)])
pos=q.sparse_pos
pvv,grids=q.preprocess_video(paired)
types=torch.zeros_like(ids); types[ids==q.video_pad_id]=2
with torch.no_grad():
    direct=q.model(input_ids=ids,mm_token_type_ids=types,pixel_values_videos=pvv,video_grid_thw=grids,
                   attention_mask=torch.ones_like(ids),use_cache=False)
    tok,ds=q.split_row(chunk.tokens)
    emb=q.model.get_input_embeddings()(ids).clone()
    emb[0,ids[0]==q.video_pad_id]=tok
    manual=q.model.model.language_model(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones_like(ids),
                                         visual_pos_masks=ids==q.video_pad_id,deepstack_visual_embeds=ds,use_cache=False)
    log=q.model.lm_head(manual.last_hidden_state[:,-1:])
    err=float((direct.logits[:,-1:].float()-log.float()).abs().max())
    assert torch.allclose(direct.logits[:,-1:].float(),log.float(),atol=.06,rtol=.01),err
    assert int(direct.logits[0,-1].argmax())==int(log[0,-1].argmax())
    # Handwritten autoregressive decode must agree with upstream generate.
    native=q.model.generate(input_ids=ids,mm_token_type_ids=types,pixel_values_videos=pvv,video_grid_thw=grids,
                            attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=12)
    native_text=q.processor.tokenizer.decode(native[0,ids.shape[1]:],skip_special_tokens=True).strip()
    ours=q.generate_from_pruned(query=query,visual_tokens=chunk.tokens,mrope=chunk.mrope,dense_llm_grid_hw=hw,
                                max_new_tokens=12,profile_vram=False).text
    assert native_text==ours,(native_text,ours)
    # Dropping all middle-frame tokens must leave all surviving positions
    # EXACTLY equal to their original dense positions (old wrapper fails this).
    sparse=allmask.clone(); sparse[1]=False; sparse[2,::2]=False
    q.prepare_context(3,hw,sparse)
    sparse_ids=q.build_prompt_ids(query,[(0,m),(2,int(sparse[2].sum()))])
    assert torch.equal(q.sparse_pos,q.last_dense_pos[:,:,q.last_keep])
    assert int((sparse_ids==q.video_pad_id).sum())==int(sparse.sum())
    rows=chunk.tokens[sparse.flatten()]
    tokens,deep=q.split_row(rows)
    dense_tokens,dense_deep=q.split_row(chunk.tokens)
    assert torch.equal(tokens,dense_tokens[sparse.flatten()])
    assert all(torch.equal(a,b[sparse.flatten()]) for a,b in zip(deep,dense_deep))
    # Causal features: encoding each observation separately vs in a batch
    # must not inject information from future observations.
    separate=torch.cat([q.encode_chunk(f[None].repeat_interleave(2,0)).tokens for f in frames])
    causality_err=float((separate.float()-chunk.tokens.float()).abs().max())
    assert torch.allclose(separate.float(),chunk.tokens.float(),atol=.12,rtol=.025),causality_err
result={'dense_logits_max_error':err,'greedy_decode_match':True,'native_text':native_text,
        'sparse_mrope_exact':True,'deepstack_row_alignment_exact':True,'batched_vs_frame_feature_max_error':causality_err,
        'torch':torch.__version__,'transformers':__import__('transformers').__version__,'passed':True}
write(ROOT/'validation/qwen3_equivalence.json',result)
print(json.dumps(result,indent=2))
