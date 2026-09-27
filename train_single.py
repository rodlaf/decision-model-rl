"""Batched native-action NLI policy training with durable, timed continuation."""
import argparse,atexit,json,os,random,shutil,signal,subprocess,time
from pathlib import Path
import numpy as np
import torch,yaml
from policy import Policy,Prompt,NativeEnv,ACTIONS,observation_delta
from exploration_objectives import EpisodicObservationNovelty,discounted_advantages,adapt_entropy
from single_eval import rollout,atomic_json
from kitchen import ROOT

STOP=False
def stop(signum,frame):
    global STOP
    STOP=True
    print('Stop requested; finishing this update and saving.',flush=True)

def collect(policy,prompt,cfg,iteration,out):
    envs=[NativeEnv(1000+iteration) for _ in range(cfg['group_size'])]
    histories=[[] for _ in envs];traces=[[] for _ in envs];totals=[0. for _ in envs]
    previous=[e.state() for e in envs]
    novelty=[EpisodicObservationNovelty(e.observation(),cfg.get('novelty_bonus',0.),cfg.get('novelty_episode_cap',0.),cfg.get('novelty_quantization',0.1),cfg.get('novelty_segment_cap')) for e in envs]
    last_noop=[None for _ in envs]
    noop_streak=[0 for _ in envs]
    try:
        for tick in range(cfg['rollout_steps']):
            observations=[e.observation() for e in envs]
            states=[prompt.build(o,h) for o,h in zip(observations,histories)]
            with torch.inference_mode():
                batch_size=cfg.get('rollout_batch_states',len(states))
                logp=torch.cat([policy.distributions(states[i:i+batch_size]) for i in range(0,len(states),batch_size)])
                probs=logp.exp()
                actions=torch.multinomial(probs,1).squeeze(1).cpu().tolist()
                logps=logp.cpu().tolist();probabilities=probs.cpu().tolist()
            for i,(env,action) in enumerate(zip(envs,actions)):
                env.step(action);now=env.state()
                reward=env.native_reward()
                after_observation=env.observation()
                intrinsic=novelty[i].observe(after_observation,reward)
                signature=observations[i].tobytes()
                unchanged=np.array_equal(observations[i],after_observation) and reward==0
                repeated_noop=unchanged and last_noop[i]==(signature,action)
                last_noop[i]=(signature,action) if unchanged else None
                noop_streak[i]=noop_streak[i]+1 if unchanged else 0
                no_op_penalty=-cfg.get('repeated_noop_penalty',0.) if repeated_noop else 0.
                report=observation_delta(observations[i],after_observation,ACTIONS[action])
                traces[i].append(dict(state=states[i],action=action,old_logp=logps[i][action],old_distribution=probabilities[i],reward=reward,intrinsic_reward=intrinsic,no_op_penalty=no_op_penalty,noop_streak=noop_streak[i],training_reward=reward+intrinsic+no_op_penalty,native_reward=env.native_reward(),outcome=report,before=previous[i],after=now))
                histories[i].append(report);totals[i]+=reward;previous[i]=now
            if tick%16==0:
                atomic_json(out/'status.json',dict(phase='rollout',iteration=iteration,tick=tick+1,horizon=cfg['rollout_steps'],group_size=len(envs),pid=os.getpid()))
        episodes=[]
        for i,env in enumerate(envs):
            expected=sum(t['native_reward'] for t in traces[i])
            assert abs(totals[i]-expected)<1e-5
            episodes.append(dict(return_=totals[i],soups=env.q('dishes'),seed=1000+iteration,trace=traces[i]))
        return episodes
    finally:
        for e in envs:e.close()

def update(policy,optimizer,cfg,episodes,out,iteration):
    returns=np.array([e['return_'] for e in episodes]);std=float(returns.std())
    advantages=(returns-returns.mean())/(std+1e-6)
    if cfg.get('credit_assignment')=='reward_to_go':
        step_advantages=discounted_advantages([[t['training_reward'] for t in e['trace']] for e in episodes],cfg['discount_gamma'])
        records=[(t,float(a)) for e,aa in zip(episodes,step_advantages) for t,a in zip(e['trace'],aa)]
    else:
        records=[(t,float(a)) for e,a in zip(episodes,advantages) for t in e['trace']]
    token_budget=cfg.get('training_token_budget')
    if token_budget:
        encoded=policy.tok([policy.template.format(premise=t['state'],hypothesis=cfg['hypothesis'].format(action='interact')) for t,a in records])['input_ids']
        for (t,a),ids in zip(records,encoded):t['nli_token_count']=len(ids)
    policy.model.train();parameters=[p for p in policy.model.parameters() if p.requires_grad]
    head=next(p for n,p in policy.model.named_parameters() if p.requires_grad and 'modules_to_save' in n)
    head_before=head.detach().float().cpu().clone()
    lora=next(p for n,p in policy.model.named_parameters() if p.requires_grad and 'lora_B' in n)
    lora_before=lora.detach().float().cpu().clone()
    stats=[];stop_early=False;updates=0
    for epoch in range(cfg['update_epochs']):
        random.shuffle(records)
        for offset in range(0,len(records),cfg['optimizer_batch_size']):
            batch=records[offset:offset+cfg['optimizer_batch_size']]
            optimizer.zero_grad(set_to_none=True)
            sums=torch.zeros(3,device='cuda')
            actions_all=torch.tensor([t['action'] for t,a in batch],device='cuda')
            old_logp_all=torch.tensor([t['old_logp'] for t,a in batch],device='cuda')
            adv_all=torch.tensor([a for t,a in batch],device='cuda')
            old_all=torch.tensor([t['old_distribution'] for t,a in batch],device='cuda')
            stuck_all=torch.tensor([min(t.get('noop_streak',0),cfg.get('stuck_entropy_cap',4)) for t,a in batch],device='cuda')
            start=0
            while start<len(batch):
                n=min(cfg['microbatch_states'],len(batch)-start)
                while token_budget and n>1 and n*max(t['nli_token_count'] for t,a in batch[start:start+n])>token_budget:n=max(1,n//2)
                micro=batch[start:start+n]
                logp=policy.distributions([t['state'] for t,a in micro])
                actions=actions_all[start:start+n]
                old_logp=old_logp_all[start:start+n]
                adv=adv_all[start:start+n]
                chosen=logp.gather(1,actions[:,None]).squeeze(1)
                ratio=(chosen-old_logp).exp()
                entropy=-(logp.exp()*logp).sum(-1)
                entropy_weight=1+cfg.get('stuck_entropy_gain',0.)*stuck_all[start:start+n]
                loss=(-torch.minimum(ratio*adv,ratio.clamp(1-cfg['clip_epsilon'],1+cfg['clip_epsilon'])*adv)-cfg['entropy_coefficient']*entropy_weight*entropy).sum()/len(batch)
                loss.backward()
                old=old_all[start:start+n]
                sums+=torch.stack((loss.detach(),entropy.detach().sum()/len(batch),(old*(old.clamp_min(1e-20).log()-logp.detach())).sum()/len(batch)))
                start+=n
            loss_sum,entropy_sum,kl_sum=sums.cpu().tolist()
            if not all(np.isfinite([loss_sum,entropy_sum,kl_sum])):raise RuntimeError('Nonfinite loss or metrics')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            # Don't make another step if the sampled policy has already drifted too far.
            if kl_sum>cfg['target_kl']:
                optimizer.zero_grad(set_to_none=True);stop_early=True;break
            optimizer.step();updates+=1
            stats.append((loss_sum,entropy_sum,kl_sum,float(norm)))
            atomic_json(out/'status.json',dict(phase='backward',iteration=iteration,epoch=epoch+1,transition=offset+len(batch),total=len(records),optimizer_steps=updates,pid=os.getpid()))
        if stop_early:break
    if not stats:raise RuntimeError('No optimizer updates; investigate policy/log-prob consistency')
    head_change=float((head.detach().float().cpu()-head_before).abs().max())
    lora_change=float((lora.detach().float().cpu()-lora_before).abs().max())
    if iteration==1 and (not head_change or not lora_change):raise RuntimeError('Head or LoRA did not change')
    means=np.mean(stats,axis=0)
    used_entropy=cfg['entropy_coefficient']
    rollout_entropy=float(np.mean([-sum(p*np.log(max(p,1e-20)) for p in t['old_distribution']) for e in episodes for t in e['trace']]))
    if cfg.get('adaptive_entropy',False):
        cfg['entropy_coefficient']=adapt_entropy(used_entropy,rollout_entropy,cfg['entropy_target'],cfg.get('entropy_adaptation_rate',0.2),cfg.get('entropy_min',0.001),cfg.get('entropy_max',0.2))
    return dict(plating_events=sum(t.get('before',{}).get('chef',{}).get('holding')!='plated soup' and t.get('after',{}).get('chef',{}).get('holding')=='plated soup' for e in episodes for t in e['trace']),no_change_fraction=float(np.mean(['no observed change' in t['outcome'] for e in episodes for t in e['trace']])),repeated_noop_fraction=float(np.mean([bool(t.get('no_op_penalty',0.)) for e in episodes for t in e['trace']])),entropy_coefficient=used_entropy,next_entropy_coefficient=cfg['entropy_coefficient'],rollout_entropy=rollout_entropy,intrinsic_return_mean=float(np.mean([sum(t.get('intrinsic_reward',0.) for t in e['trace']) for e in episodes])),returns=returns.tolist(),mean_return=float(returns.mean()),reward_std=std,zero_advantage_group=std<1e-6,soups=sum(e.get('soups',0) for e in episodes),loss=means[0],entropy=means[1],kl_old=means[2],gradient_norm=means[3],optimizer_steps=updates,kl_early_stop=stop_early,head_max_update=head_change,lora_max_update=lora_change)

def save(policy,optimizer,out,iteration,elapsed,best):
    dest=out/f'checkpoint-{iteration:06d}';temp=out/f'.checkpoint-{iteration:06d}.tmp'
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir();policy.save(temp)
    torch.save(dict(optimizer=optimizer.state_dict(),iteration=iteration,elapsed_seconds=elapsed,best_score=best,reward_scheme=policy.cfg.get('reward_scheme'),entropy_coefficient=policy.cfg['entropy_coefficient'],rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),temp/'training_state.pt')
    temp.replace(dest)
    atomic_json(out/'latest-checkpoint.json',dict(path=str(dest),iteration=iteration,elapsed_seconds=elapsed))
    checkpoints=sorted(out.glob('checkpoint-*'))
    for old in checkpoints[:-3]:shutil.rmtree(old)
    return dest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/single.yaml');ap.add_argument('--max-updates',type=int);a=ap.parse_args()
    cfg=yaml.safe_load((ROOT/a.config).read_text());out=ROOT/cfg['run_dir'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if cfg.get('inhibit_suspend',False):
        inhibitor=subprocess.Popen(['/usr/bin/gnome-session-inhibit','--app-id=openjev-training','--inhibit=suspend:idle','--reason=OpenJev-overnight-training','--inhibit-only'])
        atexit.register(inhibitor.terminate)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    policy=Policy(cfg);prompt=Prompt(cfg)
    groups=[dict(params=[p for n,p in policy.model.named_parameters() if p.requires_grad and 'lora_' in n],lr=cfg['learning_rate']),dict(params=[p for n,p in policy.model.named_parameters() if p.requires_grad and 'modules_to_save' in n],lr=cfg['head_learning_rate'])]
    optimizer=torch.optim.AdamW(groups,weight_decay=0.,fused=True)
    iteration=0;elapsed=0.;best=-float('inf')
    latest=out/'latest-checkpoint.json'
    source=Path(json.loads(latest.read_text())['path']) if latest.exists() else (ROOT/cfg['initial_checkpoint'] if cfg.get('initial_checkpoint') else None)
    if source is not None:
        policy.load(source)
        saved=torch.load(source/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        for group,lr in zip(optimizer.param_groups,[cfg['learning_rate'],cfg['head_learning_rate']]):group['lr']=lr;group['fused']=True
        for parameter,state in optimizer.state.items():
            if 'step' in state:state['step']=state['step'].to(parameter.device)
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        if latest.exists():
            iteration=saved['iteration'];elapsed=saved['elapsed_seconds'];best=saved['best_score']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
            if cfg.get('adaptive_entropy',False):cfg['entropy_coefficient']=max(cfg.get('entropy_min',0.),min(cfg.get('entropy_max',float('inf')),saved.get('entropy_coefficient',cfg['entropy_coefficient'])))
        if saved.get('reward_scheme')!=cfg.get('reward_scheme'):best=-float('inf')
    atomic_json(out/'config.json',cfg);atomic_json(out/'model.json',policy.metadata)
    print('START',json.dumps(dict(pid=os.getpid(),resume=str(source),iteration=iteration,completed_hours=elapsed/3600,target_hours=cfg['duration_hours'],microbatch_states=cfg['microbatch_states'],gradient_checkpointing=cfg['gradient_checkpointing'])),flush=True)
    count=0
    while elapsed<cfg['duration_hours']*3600 and not STOP:
        tic=time.monotonic();iteration+=1;count+=1;policy.model.eval()
        episodes=collect(policy,prompt,cfg,iteration,out)
        rollout_seconds=time.monotonic()-tic
        print('ROLLOUT',iteration,[round(e['return_'],3) for e in episodes],flush=True)
        atomic_json(out/'status.json',dict(phase='backward',iteration=iteration,transition=0,total=cfg['group_size']*cfg['rollout_steps'],pid=os.getpid()))
        update_start=time.monotonic()
        metrics=update(policy,optimizer,cfg,episodes,out,iteration)
        update_seconds=time.monotonic()-update_start
        metrics.update(rollout_seconds=rollout_seconds,rollout_moves_per_second=cfg['group_size']*cfg['rollout_steps']/rollout_seconds,update_seconds=update_seconds)
        if iteration%cfg['eval_every']==0 or elapsed+time.monotonic()-tic>=cfg['duration_hours']*3600:
            policy.model.eval();evaluation=rollout(policy,prompt,0,cfg['eval_steps'],sample=False)
            atomic_json(out/'latest-eval.json',dict(iteration=iteration,**evaluation))
            score=evaluation['soups']*1000+evaluation['return_']
            if score>best:
                best=score;policy.save(out/'best-eval-adapter');atomic_json(out/'best-eval.json',dict(iteration=iteration,score=score,soups=evaluation['soups']))
            metrics.update(eval_soups=evaluation['soups'],eval_return=evaluation['return_'])
        elapsed+=time.monotonic()-tic
        metrics.update(iteration=iteration,reward_scheme=cfg.get('reward_scheme'),environment=cfg.get('environment'),seconds=time.monotonic()-tic,elapsed_seconds=elapsed,target_seconds=cfg['duration_hours']*3600,max_tokens=policy.metadata['max_tokens'],peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,pid=os.getpid())
        metrics.update(rollout_moves=cfg['group_size']*cfg['rollout_steps'],end_to_end_moves_per_second=cfg['group_size']*cfg['rollout_steps']/metrics['seconds'])
        checkpoint=save(policy,optimizer,out,iteration,elapsed,best)
        atomic_json(out/'latest-rollouts.json',dict(iteration=iteration,episodes=episodes))
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        atomic_json(out/'status.json',dict(phase='updated',checkpoint=str(checkpoint),**metrics))
        print('UPDATE',json.dumps(metrics),flush=True)
        if a.max_updates and count>=a.max_updates:break
    phase='complete' if elapsed>=cfg['duration_hours']*3600 else 'stopped'
    atomic_json(out/'status.json',dict(phase=phase,iteration=iteration,elapsed_seconds=elapsed,target_seconds=cfg['duration_hours']*3600,pid=os.getpid()))
    print(phase.upper(),iteration,elapsed,flush=True)
if __name__=='__main__':main()
