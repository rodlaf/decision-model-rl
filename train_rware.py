"""Sequential RWARE training, continuing the cooking adapter and checking retention."""
import argparse,hashlib,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch,yaml
import train_single as core
from train import optimizer_for,rollout_duo
from policy import Policy,Prompt as CookingPrompt
from rware import Warehouse,Prompt,outcome
from exploration_objectives import EpisodicObservationNovelty
from kitchen import ROOT


def episode(policy,prompt,cfg,seed,steps):
    env=Warehouse(seed,cfg['num_agents'],cfg['requests']);history=[[] for _ in range(env.num_agents)]
    totals=np.zeros(env.num_agents);trace=[];tic=time.monotonic()
    try:
        for tick in range(steps):
            obs=[env.observation(a) for a in range(env.num_agents)]
            states=[prompt.build(o,h) for o,h in zip(obs,history)]
            with torch.inference_mode():prob=policy.distributions(states).exp().cpu()
            actions=prob.argmax(-1).tolist();before=env.state();env.step(actions)
            rewards=[env.reward(a) for a in range(env.num_agents)];totals+=rewards
            for a in range(env.num_agents):history[a].append(outcome(obs[a],env.observation(a),actions[a],rewards[a]))
            trace.append(dict(tick=tick,states=states,actions=actions,probabilities=prob.tolist(),rewards=rewards,before=before,after=env.state(),deliveries=env.deliveries,returns=env.returns,pickups=env.pickups))
        return dict(seed=seed,steps=steps,deliveries=env.deliveries,completed_returns=env.returns,pickups=env.pickups,returns=totals.tolist(),seconds=time.monotonic()-tic,trace=trace)
    finally:env.close()


def retention(policy,cfg):
    old_cfg,old_actions=policy.cfg,policy.actions
    cooking=yaml.safe_load((ROOT/cfg['retention_config']).read_text())
    try:
        policy.cfg=cooking;policy.actions=cooking['actions'];policy._nli_tokens.clear()
        return rollout_duo(policy,CookingPrompt(cooking),0,cfg['retention_steps'])
    finally:
        policy.cfg=old_cfg;policy.actions=old_actions;policy._nli_tokens.clear()


def collect(policy,prompt,cfg,iteration,out):
    envs=[Warehouse(1000+iteration,cfg['num_agents'],cfg['requests']) for _ in range(cfg['group_size'])]
    pairs=[(env,a) for env in envs for a in range(env.num_agents)]
    histories=[[] for _ in pairs];traces=[[] for _ in pairs]
    novelty=[EpisodicObservationNovelty(env.observation(a),cfg['novelty_bonus'],cfg['novelty_episode_cap'],cfg['novelty_quantization'],cfg['novelty_segment_cap']) for env,a in pairs]
    last_noop=[None]*len(pairs);streak=[0]*len(pairs);tic=time.monotonic()
    try:
        for tick in range(cfg['rollout_steps']):
            obs=[env.observation(a) for env,a in pairs]
            states=[prompt.build(o,h) for o,h in zip(obs,histories)]
            with torch.inference_mode():
                batch=cfg['rollout_batch_states']
                logp=torch.cat([policy.distributions(states[i:i+batch]) for i in range(0,len(states),batch)])
                probs=logp.exp();actions=torch.multinomial(probs,1).flatten().cpu().tolist()
                logs=logp.cpu().tolist();probabilities=probs.cpu().tolist()
            for i,env in enumerate(envs):env.step(actions[i*env.num_agents:(i+1)*env.num_agents])
            for i,(env,a) in enumerate(pairs):
                after=env.observation(a);reward=env.reward(a)
                intrinsic=novelty[i].observe(after,reward)
                unchanged=np.array_equal(obs[i],after) and reward==0
                signature=(obs[i].tobytes(),actions[i]);repeated=unchanged and last_noop[i]==signature
                last_noop[i]=signature if unchanged else None;streak[i]=streak[i]+1 if unchanged else 0
                penalty=-cfg['repeated_noop_penalty'] if repeated else 0.
                report=outcome(obs[i],after,actions[i],reward);histories[i].append(report)
                traces[i].append(dict(state=states[i],action=actions[i],old_logp=logs[i][actions[i]],old_distribution=probabilities[i],reward=reward,training_reward=reward+intrinsic+penalty,intrinsic_reward=intrinsic,no_op_penalty=penalty,noop_streak=streak[i],outcome=report))
            if tick%16==0:core.atomic_json(out/'status.json',dict(phase='rollout',iteration=iteration,tick=tick+1,horizon=cfg['rollout_steps'],pid=os.getpid()))
        episodes=[dict(return_=sum(t['reward'] for t in trace),trace=trace) for trace in traces]
        return episodes,dict(deliveries=sum(e.deliveries for e in envs),completed_returns=sum(e.returns for e in envs),pickups=sum(e.pickups for e in envs),rollout_seconds=time.monotonic()-tic)
    finally:
        for env in envs:env.close()


def evaluate(policy,prompt,cfg,out,iteration):
    policy.model.eval();rware=episode(policy,prompt,cfg,0,cfg['eval_steps']);cooking=retention(policy,cfg)
    core.atomic_json(out/'latest-eval.json',dict(iteration=iteration,**rware))
    core.atomic_json(out/'latest-retention.json',dict(iteration=iteration,**cooking))
    summary=dict(iteration=iteration,rware_deliveries=rware['deliveries'],rware_completed_returns=rware['completed_returns'],rware_pickups=rware['pickups'],rware_mean_return=float(np.mean(rware['returns'])),overcooked_soups=cooking['soups'])
    with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    print('EVAL',json.dumps(summary),flush=True)
    from render_rware import render
    render(dict(iteration=iteration,**rware),ROOT/cfg['video_dir']/f'checkpoint-{iteration:06d}.mp4',cfg)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/rware.yaml');p.add_argument('--max-updates',type=int);p.add_argument('--skip-evaluation',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text());out=ROOT/cfg['run_dir'];out.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    signal.signal(signal.SIGINT,core.stop);signal.signal(signal.SIGTERM,core.stop)
    policy=Policy(cfg);prompt=Prompt(cfg);optimizer=optimizer_for(policy,cfg)
    latest=out/'latest-checkpoint.json';iteration=0;elapsed=0.;best=-float('inf')
    if latest.exists():
        source=Path(json.loads(latest.read_text())['path']);policy.load(source)
        saved=torch.load(source/'training_state.pt',map_location='cpu',weights_only=False);optimizer.load_state_dict(saved['optimizer'])
        for param,state in optimizer.state.items():
            if 'step' in state:state['step']=state['step'].to(param.device)
        iteration=saved['iteration'];elapsed=saved['elapsed_seconds'];best=saved['best_score'];cfg['entropy_coefficient']=saved['entropy_coefficient']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        source=ROOT/cfg['initial_adapter'];policy.load(source)
        core.atomic_json(out/'lineage.json',dict(source=str(source),source_checkpoint=220,adapter_sha256=hashlib.sha256((source/'adapter_model.safetensors').read_bytes()).hexdigest(),optimizer='fresh; continue existing LoRA and classifier',old_task_training=False))
    core.atomic_json(out/'config.json',cfg);core.atomic_json(out/'model.json',policy.metadata)
    print('START',json.dumps(dict(pid=os.getpid(),source=str(source),iteration=iteration,hours=cfg['duration_hours'])),flush=True)
    if iteration==0 and not args.skip_evaluation:
        baseline=evaluate(policy,prompt,cfg,out,0);core.atomic_json(out/'baseline.json',baseline)
        if baseline['overcooked_soups']!=6:raise RuntimeError('Cooking retention baseline differs from released checkpoint')
    count=0
    while elapsed<cfg['duration_hours']*3600 and not core.STOP:
        start=time.monotonic();iteration+=1;count+=1;policy.model.eval()
        episodes,metrics=collect(policy,prompt,cfg,iteration,out)
        print('ROLLOUT',iteration,json.dumps(metrics),flush=True)
        tic=time.monotonic();metrics.update(core.update(policy,optimizer,cfg,episodes,out,iteration));metrics['update_seconds']=time.monotonic()-tic
        elapsed+=time.monotonic()-start
        # Save before evaluation/rendering so those failures cannot discard an update.
        checkpoint=core.save(policy,optimizer,out,iteration,elapsed,best)
        if iteration%cfg['eval_every']==0 and not args.skip_evaluation:
            tic=time.monotonic();evaluation=evaluate(policy,prompt,cfg,out,iteration);elapsed+=time.monotonic()-tic
            score=evaluation['rware_completed_returns']*10000+evaluation['rware_deliveries']*100+evaluation['rware_mean_return']
            if score>best:
                best=score;policy.save(out/'best-eval-adapter');core.atomic_json(out/'best-eval.json',evaluation)
                core.atomic_json(out/'best-eval-trace.json',json.loads((out/'latest-eval.json').read_text()))
            metrics.update(evaluation)
        metrics.update(iteration=iteration,elapsed_seconds=elapsed,agent_moves_per_second=cfg['group_size']*cfg['num_agents']*cfg['rollout_steps']/metrics['rollout_seconds'],max_tokens=policy.metadata['max_tokens'],peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        core.atomic_json(out/'status.json',dict(phase='updated',checkpoint=str(checkpoint),**metrics));print('UPDATE',json.dumps(metrics),flush=True)
        if args.max_updates and count>=args.max_updates:break
    core.atomic_json(out/'status.json',dict(phase='stopped' if core.STOP or args.max_updates else 'complete',iteration=iteration,elapsed_seconds=elapsed))

if __name__=='__main__':main()
