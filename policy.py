"""Compact native-observation policy, unchanged 3-way NLI head + backbone LoRA."""
import ctypes as C
import json,time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from peft import LoraConfig,get_peft_model
from kitchen import Kitchen,ROOT

ACTIONS=['stay','up','down','left','right','interact']
class NativeEnv(Kitchen):
    def __init__(self,seed,num_agents=1,layout=0):
        super().__init__(layout=layout,seed=seed,num_agents=num_agents)
        self.lib.bridge_observation.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_float)]
        self.lib.bridge_observation.restype=None
        self.lib.bridge_reward.argtypes=[C.c_void_p,C.c_int];self.lib.bridge_reward.restype=C.c_float
    def observation(self,agent=0):
        data=(C.c_float*43)();self.lib.bridge_observation(self.ptr,agent,data)
        return np.array(data,dtype=np.float32)
    def native_reward(self,agent=0):return float(self.lib.bridge_reward(self.ptr,agent))

def decode(obs):
    # Exact observation schema in PufferLib's overcooked_obs.h. Layout 0 is 5x5.
    face=['up','down','left','right'][int(obs[:4].argmax())]
    hand=['onion','soup','plate','empty'][int(obs[4:8].argmax())] if obs[4:8].sum() else 'unencoded'
    pair=lambda i:tuple(int(round(float(v)*5)) for v in obs[i:i+2])
    flags=['empty','full','cooking','ready']
    return dict(face=face,hand=hand,offsets={n:pair(i) for n,i in zip(['onion_source','plate_source','loose_soup','serve','empty_counter','pot','loose_onion','loose_plate'],range(8,24,2))},soup_ingredients=tuple(int(round(float(v)*3)) for v in obs[24:26]),pot_ingredients=tuple(int(round(float(v)*3)) for v in obs[26:28]),pot_exists=bool(obs[28]),pot_flags=[f for i,f in enumerate(flags) if obs[29+i]],cook_remaining=int(round(float(obs[33])*20)),blocked=''.join(str(int(x)) for x in obs[34:38]),partner=pair(38),position=pair(40),reward=round(float(obs[42]),3))

def relative(pair):
    x,y=pair
    parts=[]
    if x:parts.append(f"{abs(x)} tile{'s' if abs(x)!=1 else ''} {'right' if x>0 else 'left'}")
    if y:parts.append(f"{abs(y)} tile{'s' if abs(y)!=1 else ''} {'down' if y>0 else 'up'}")
    return ' and '.join(parts) or 'not located by the observation'

def holding(hand):
    return {'onion':'an onion','soup':'plated soup','plate':'a plate','empty':'nothing','unencoded':'an unidentified item'}[hand]

def observation_delta(before,after,action):
    a,b=decode(before),decode(after);changes=[]
    if a['position']!=b['position']:changes.append('moved '+relative(tuple(y-x for x,y in zip(a['position'],b['position']))))
    elif a['face']!=b['face']:changes.append('turned '+b['face'])
    if a['hand']!=b['hand']:changes.append('now holding '+holding(b['hand']))
    if a['pot_ingredients']!=b['pot_ingredients']:changes.append(f"pot now has {b['pot_ingredients'][0]} onions")
    if a['pot_flags']!=b['pot_flags']:changes.append('pot now '+(' and '.join(b['pot_flags']) or 'idle'))
    return action+': '+('; '.join(changes) if changes else 'no observed change')+'.'

class Prompt:
    def __init__(self,cfg):self.cfg=cfg
    def build(self,obs,history):
        d=decode(obs)
        names=['Onion supply','Plate supply','Loose soup','Serving station','Empty counter','Pot','Loose onion','Loose plate']
        locations='\n'.join(f"{name}: {relative(pair)}." for name,pair in zip(names,d['offsets'].values()))
        blocked=[name for name,flag in zip(['up','down','left','right'],d['blocked']) if flag=='1']
        teammate=''
        if self.cfg.get('num_agents',1)>1:
            teammate=f" Your teammate is {relative(d['partner'])}."
            adjacent={(0,-1):'up',(0,1):'down',(-1,0):'left',(1,0):'right'}.get(d['partner'])
            if adjacent:
                if adjacent not in blocked:blocked.append(adjacent)
                teammate+=f" The teammate occupies the adjacent {adjacent} tile, so {adjacent} is blocked this tick."
        pot=(f"The pot contains {d['pot_ingredients'][0]} onions and {d['pot_ingredients'][1]} tomatoes; it is {' and '.join(d['pot_flags']) or 'idle'}. Cooking time remaining: {d['cook_remaining']} ticks." if d['pot_exists'] else 'No pot is observed.')
        recent=' '.join(history[-self.cfg['history_window']:]) or 'None yet.'
        return (self.cfg['rules'].strip()+f"\nYou are at {d['position']}, facing {d['face']}, holding {holding(d['hand'])}.{teammate}\n"
                +locations+'\n'+pot+f" Loose soup contains {d['soup_ingredients'][0]} onions and {d['soup_ingredients'][1]} tomatoes.\n"
                +('You cannot move '+', '.join(blocked)+'.' if blocked else 'All four adjacent directions are free.')
                +f" Last environment reward: {d['reward']}.\nRecent actions, oldest first: {recent}\nWhich button should you press now?")

class Policy:
    def __init__(self,cfg):
        path=ROOT/cfg['model_path'];self.cfg=cfg
        if cfg.get('fast_convolution',False):
            from efficient_nli import enable_fast_convolution
            enable_fast_convolution()
        if cfg.get('liger_kernels',False):
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
            apply_liger_kernel_to_qwen3_5(fused_linear_cross_entropy=False)
        self.tok=AutoTokenizer.from_pretrained(path);self.tok.padding_side='right'
        base=AutoModelForSequenceClassification.from_pretrained(path,dtype=torch.bfloat16).to('cuda')
        base.config.get_text_config().pad_token_id=self.tok.pad_token_id
        # Explicit text-only module names include Qwen3.5 attention, delta-net and MLP linears.
        targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and '.language_model.layers.' in n]
        if not targets:raise RuntimeError('No text backbone target modules matched')
        self.model=get_peft_model(base,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=0.,target_modules=targets,modules_to_save=['score'],bias='none'))
        if cfg.get('gradient_checkpointing',True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.template=base.config.nli_template
        self.metadata=dict(dtype='bfloat16',quantized=False,targets=targets,trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad),total=sum(p.numel() for p in self.model.parameters()),head='existing 3-way classifier, fully trainable',max_tokens=0)
        assert all('lora_' in n or 'modules_to_save' in n for n,p in self.model.named_parameters() if p.requires_grad)
    def distribution(self,state):
        return self.distributions([state])[0]
    def distributions(self,states):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.cfg.get('mixed_precision',False)):
            if self.cfg.get('shared_prefix',False):
                from efficient_nli import shared_prefix_distributions
                return shared_prefix_distributions(self,states)
            return self._full_distributions(states)
    def _full_distributions(self,states):
        texts=[self.template.format(premise=state,hypothesis=self.cfg['hypothesis'].format(action=a)) for state in states for a in ACTIONS]
        enc=self.tok(texts,padding=True,return_tensors='pt').to('cuda')
        length=enc.input_ids.shape[1];self.metadata['max_tokens']=max(length,self.metadata['max_tokens'])
        if length>self.cfg['max_tokens']:raise RuntimeError(f'Prompt has {length} tokens, exceeds {self.cfg["max_tokens"]}; no truncation allowed')
        model=self.model.get_base_model()
        hidden=model.model(**enc,use_cache=False).last_hidden_state
        last=enc.attention_mask.sum(-1)-1
        logits=model.score(hidden[torch.arange(len(texts),device='cuda'),last]).float()
        log_entail=torch.log_softmax(logits,dim=-1)[:,1].reshape(len(states),len(ACTIONS))
        log_probs=log_entail-torch.logsumexp(log_entail,dim=-1,keepdim=True)
        return log_probs
    def load(self,path):
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict,get_peft_model_state_dict
        weights=load_peft_weights(str(path),device='cuda')
        expected=set(get_peft_model_state_dict(self.model))
        if set(weights)!=expected:raise RuntimeError(f'Adapter keys differ: {set(weights)^expected}')
        result=set_peft_model_state_dict(self.model,weights)
        if result.unexpected_keys:raise RuntimeError(result.unexpected_keys)
    def save(self,path):
        self.model.save_pretrained(path);self.tok.save_pretrained(path)
