"""Fused short convolution and differentiable shared-prefix NLI evaluation."""
from collections import OrderedDict
import torch

_functional_cache_classes={}

def enable_fast_convolution():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from fla.modules.convolution import causal_conv1d
    def fused(hidden_states,weight,bias=None,activation=None,**kwargs):
        output,_=causal_conv1d(hidden_states.transpose(1,2),weight,bias,activation=activation,backend='triton')
        return output.transpose(1,2)
    modeling.causal_conv1d_fn=fused

def assign_recurrent(self,recurrent_states,state_idx=0,**kwargs):
    # The inference cache normally mutates h0 in-place. Preserve autograd's saved h0.
    self.recurrent_states[state_idx]=recurrent_states
    self.is_recurrent_states_initialized[state_idx]=True
    return recurrent_states

def shared_prefix_distributions(policy,states):
    from policy import ACTIONS
    device=next(policy.model.parameters()).device
    if not hasattr(policy,'_nli_tokens'):policy._nli_tokens=OrderedDict()
    token_cache=policy._nli_tokens
    missing=list(dict.fromkeys(s for s in states if s not in token_cache))
    if missing:
        texts=[policy.template.format(premise=s,hypothesis=policy.cfg['hypothesis'].format(action=a)) for s in missing for a in ACTIONS]
        ids=policy.tok(texts)['input_ids']
        for i,state in enumerate(missing):
            rows=ids[i*6:i*6+6];split=0;limit=min(map(len,rows))-2
            while split<limit and all(row[split]==rows[0][split] for row in rows):split+=1
            token_cache[state]=(rows[0][:split],[row[split:] for row in rows],max(map(len,rows)))
    prefixes=[];suffixes=[];length=0
    for state in states:
        prefix,suffix,n=token_cache[state];token_cache.move_to_end(state)
        prefixes.append(prefix);suffixes.extend(suffix);length=max(length,n)
    while len(token_cache)>4096:token_cache.popitem(last=False)
    policy.metadata['max_tokens']=max(length,policy.metadata['max_tokens'])
    if length>policy.cfg['max_tokens']:raise RuntimeError(f'Prompt has {length} tokens; no truncation allowed')
    prefix_width=max(map(len,prefixes));suffix_width=max(map(len,suffixes));pad=policy.tok.pad_token_id
    arrays=[
        [[pad]*(prefix_width-len(row))+row for row in prefixes],
        [[0]*(prefix_width-len(row))+[1]*len(row) for row in prefixes],
        [row+[pad]*(suffix_width-len(row)) for row in suffixes],
        [[1]*len(row)+[0]*(suffix_width-len(row)) for row in suffixes],
    ]
    # One asynchronous host-to-device copy, prepared before either model pass.
    flat=[v for array in arrays for row in array for v in row]
    packed=torch.tensor(flat,dtype=torch.long,pin_memory=device.type=='cuda').to(device,non_blocking=True)
    b=len(states);n=b*6
    pieces=packed.split([b*prefix_width,b*prefix_width,n*suffix_width,n*suffix_width])
    prefix_ids,prefix_mask=(v.reshape(b,prefix_width) for v in pieces[:2])
    suffix_ids,suffix_mask=(v.reshape(n,suffix_width) for v in pieces[2:])
    prefix_pos=(prefix_mask.cumsum(-1)-1).clamp_min(0).unsqueeze(0).expand(3,-1,-1)
    mask=torch.cat((prefix_mask.repeat_interleave(6,0),suffix_mask),dim=-1)
    suffix_pos=(prefix_mask.sum(-1).repeat_interleave(6)[:,None]+torch.arange(suffix_width,device=device)[None,:]).unsqueeze(0).expand(3,-1,-1)
    prefix_attention=prefix_mask;suffix_attention=mask
    if policy.cfg.get('explicit_masks',False):
        p=torch.arange(prefix_width,device=device)
        prefix_attention={'full_attention':(p[None,:]<=p[:,None])[None,None,:,:]&prefix_mask[:,None,None,:].bool(),'linear_attention':prefix_mask}
        q=torch.arange(suffix_width,device=device)+prefix_width
        k=torch.arange(prefix_width+suffix_width,device=device)
        suffix_attention={'full_attention':(k[None,:]<=q[:,None])[None,None,:,:]&mask[:,None,None,:].bool(),'linear_attention':suffix_mask}
    model=policy.model.get_base_model()
    cache=model.model(input_ids=prefix_ids,attention_mask=prefix_attention,position_ids=prefix_pos,use_cache=True).past_key_values
    cache.reorder_cache(torch.arange(len(states),device=device).repeat_interleave(6))
    for layer in cache.layers:
        if hasattr(layer,'recurrent_states'):
            # An instance-bound method would create layer -> method -> layer cycles,
            # retaining CUDA cache tensors until Python's cyclic GC runs.
            cls=type(layer)
            if cls not in _functional_cache_classes:
                _functional_cache_classes[cls]=type('Functional'+cls.__name__,(cls,),{'update_recurrent_state':assign_recurrent})
            layer.__class__=_functional_cache_classes[cls]
    hidden=model.model(input_ids=suffix_ids,attention_mask=suffix_attention,position_ids=suffix_pos,past_key_values=cache,use_cache=True).last_hidden_state
    logits=model.score(hidden[torch.arange(n,device=device),suffix_mask.sum(-1)-1]).float()
    ent=torch.log_softmax(logits,-1)[:,1].reshape(len(states),6)
    return ent-torch.logsumexp(ent,-1,keepdim=True)
