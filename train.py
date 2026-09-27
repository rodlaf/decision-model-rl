"""Shared-policy two-chef training from native PufferLib observations and rewards."""
import argparse,atexit,json,os,random,signal,subprocess,time
from pathlib import Path

import numpy as np
import torch,yaml

import train_single as core
from exploration_objectives import EpisodicObservationNovelty
from policy import Policy,Prompt,NativeEnv,ACTIONS,observation_delta
from kitchen import ROOT


def joint_outcome(before,after,own_action,other_action):
    change=observation_delta(before,after,ACTIONS[own_action])
    return f"You pressed {ACTIONS[own_action]}; teammate pressed {ACTIONS[other_action]}. {change}"


def collect(policy,prompt,cfg,iteration,out):
    n_agents=cfg['num_agents']
    envs=[NativeEnv(1000+iteration,num_agents=n_agents) for _ in range(cfg['group_size'])]
    histories=[[[] for _ in range(n_agents)] for _ in envs]
    traces=[[[] for _ in range(n_agents)] for _ in envs]
    totals=np.zeros((len(envs),n_agents),dtype=np.float64)
    previous=[[env.state(agent) for agent in range(n_agents)] for env in envs]
    novelty=[[EpisodicObservationNovelty(env.observation(agent),cfg.get('novelty_bonus',0.),cfg.get('novelty_episode_cap',0.),cfg.get('novelty_quantization',0.1),cfg.get('novelty_segment_cap')) for agent in range(n_agents)] for env in envs]
    last_noop=[[None for _ in range(n_agents)] for _ in envs]
    noop_streak=np.zeros((len(envs),n_agents),dtype=np.int32)
    try:
        for tick in range(cfg['rollout_steps']):
            observations=[[env.observation(agent) for agent in range(n_agents)] for env in envs]
            states=[[prompt.build(observations[e][agent],histories[e][agent]) for agent in range(n_agents)] for e in range(len(envs))]
            flat_states=[state for pair in states for state in pair]
            with torch.inference_mode():
                batch=cfg.get('rollout_batch_states',len(flat_states))
                logp=torch.cat([policy.distributions(flat_states[i:i+batch]) for i in range(0,len(flat_states),batch)])
                probs=logp.exp()
                flat_actions=torch.multinomial(probs,1).squeeze(1).cpu().tolist()
                flat_logps=logp.cpu().tolist();flat_probs=probs.cpu().tolist()
            actions=[flat_actions[e*n_agents:(e+1)*n_agents] for e in range(len(envs))]
            for e,env in enumerate(envs):env.step_joint(actions[e])
            for e,env in enumerate(envs):
                for agent in range(n_agents):
                    idx=e*n_agents+agent
                    after_obs=env.observation(agent);now=env.state(agent)
                    reward=env.native_reward(agent)
                    intrinsic=novelty[e][agent].observe(after_obs,reward)
                    signature=observations[e][agent].tobytes()
                    unchanged=np.array_equal(observations[e][agent],after_obs) and reward==0
                    repeated=unchanged and last_noop[e][agent]==(signature,actions[e][agent])
                    last_noop[e][agent]=(signature,actions[e][agent]) if unchanged else None
                    noop_streak[e,agent]=noop_streak[e,agent]+1 if unchanged else 0
                    penalty=-cfg.get('repeated_noop_penalty',0.) if repeated else 0.
                    other=1-agent
                    report=joint_outcome(observations[e][agent],after_obs,actions[e][agent],actions[e][other])
                    traces[e][agent].append(dict(state=states[e][agent],action=actions[e][agent],old_logp=flat_logps[idx][actions[e][agent]],old_distribution=flat_probs[idx],reward=reward,intrinsic_reward=intrinsic,no_op_penalty=penalty,noop_streak=int(noop_streak[e,agent]),training_reward=reward+intrinsic+penalty,native_reward=reward,outcome=report,before=previous[e][agent],after=now,agent=agent,teammate_action=actions[e][other]))
                    histories[e][agent].append(report);totals[e,agent]+=reward;previous[e][agent]=now
            if tick%16==0:
                core.atomic_json(out/'status.json',dict(phase='rollout',iteration=iteration,tick=tick+1,horizon=cfg['rollout_steps'],environments=len(envs),agents=n_agents,pid=os.getpid()))
        episodes=[];team_soups=0
        for e,env in enumerate(envs):
            soups=env.q('dishes');team_soups+=soups
            for agent in range(n_agents):
                expected=sum(t['native_reward'] for t in traces[e][agent])
                assert abs(totals[e,agent]-expected)<1e-5
                episodes.append(dict(return_=float(totals[e,agent]),soups=soups if agent==0 else 0,team=e,agent=agent,trace=traces[e][agent]))
        return episodes,team_soups
    finally:
        for env in envs:env.close()


def rollout_duo(policy,prompt,seed,steps,sample=False):
    env=NativeEnv(seed,num_agents=2);histories=[[],[]];totals=[0.,0.];trace=[];start=time.perf_counter()
    try:
        for tick in range(steps):
            observations=[env.observation(agent) for agent in range(2)]
            states=[prompt.build(observations[agent],histories[agent]) for agent in range(2)]
            decision_start=time.perf_counter()
            with torch.inference_mode():logp=policy.distributions(states)
            latency_ms=(time.perf_counter()-decision_start)*1000
            probs=logp.exp();actions=[int(torch.multinomial(p,1)) if sample else int(p.argmax()) for p in probs]
            before=[env.state(agent) for agent in range(2)];env.step_joint(actions)
            agents=[]
            for agent in range(2):
                after_obs=env.observation(agent);reward=env.native_reward(agent);totals[agent]+=reward
                report=joint_outcome(observations[agent],after_obs,actions[agent],actions[1-agent]);histories[agent].append(report)
                agents.append(dict(agent=agent,state=states[agent],action=actions[agent],distribution=probs[agent].cpu().tolist(),reward=reward,outcome=report,before=before[agent],after=env.state(agent)))
            trace.append(dict(tick=tick+1,latency_ms=latency_ms,agents=agents,soups=env.q('dishes')))
        return dict(seed=seed,returns=totals,team_return=float(np.mean(totals)),soups=env.q('dishes'),steps=steps,seconds=time.perf_counter()-start,trace=trace)
    finally:env.close()


def optimizer_for(policy,cfg):
    return torch.optim.AdamW([
        dict(params=[p for n,p in policy.model.named_parameters() if p.requires_grad and 'lora_' in n],lr=cfg['learning_rate']),
        dict(params=[p for n,p in policy.model.named_parameters() if p.requires_grad and 'modules_to_save' in n],lr=cfg['head_learning_rate'])
    ],weight_decay=0.,fused=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/duo.yaml');ap.add_argument('--max-updates',type=int);args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text());out=ROOT/cfg['run_dir'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,core.stop);signal.signal(signal.SIGINT,core.stop)
    if cfg.get('inhibit_suspend',False):
        inhibitor=subprocess.Popen(['/usr/bin/gnome-session-inhibit','--app-id=openjev-duo-training','--inhibit=suspend:idle','--reason=OpenJev-duo-training','--inhibit-only'])
        atexit.register(inhibitor.terminate)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    policy=Policy(cfg);prompt=Prompt(cfg);optimizer=optimizer_for(policy,cfg)
    iteration=0;elapsed=0.;best=-float('inf');latest=out/'latest-checkpoint.json';resume=False
    if latest.exists():
        source=Path(json.loads(latest.read_text())['path']);resume=True
        policy.load(source);saved=torch.load(source/'training_state.pt',map_location='cpu',weights_only=False);optimizer.load_state_dict(saved['optimizer'])
        for parameter,state in optimizer.state.items():
            if 'step' in state:state['step']=state['step'].to(parameter.device)
        iteration=saved['iteration'];elapsed=saved['elapsed_seconds'];best=saved['best_score']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        cfg['entropy_coefficient']=saved.get('entropy_coefficient',cfg['entropy_coefficient'])
    else:
        source=ROOT/cfg['initial_adapter'];policy.load(source)
    core.atomic_json(out/'config.json',cfg);core.atomic_json(out/'model.json',policy.metadata)
    core.atomic_json(out/'lineage.json',dict(source=str(source),source_best_iteration=cfg.get('source_best_iteration'),source_best_soups=cfg.get('source_best_soups'),optimizer='resumed' if resume else 'fresh for two-agent task'))
    print('START',json.dumps(dict(pid=os.getpid(),source=str(source),resume=resume,iteration=iteration,completed_hours=elapsed/3600,target_hours=cfg['duration_hours'],agents=2,environments=cfg['group_size'])),flush=True)
    count=0
    while elapsed<cfg['duration_hours']*3600 and not core.STOP:
        tic=time.monotonic();iteration+=1;count+=1;policy.model.eval()
        episodes,team_soups=collect(policy,prompt,cfg,iteration,out);rollout_seconds=time.monotonic()-tic
        print('ROLLOUT',iteration,'returns',[round(e['return_'],3) for e in episodes],'team_soups',team_soups,flush=True)
        core.atomic_json(out/'status.json',dict(phase='backward',iteration=iteration,transition=0,total=len(episodes)*cfg['rollout_steps'],pid=os.getpid()))
        update_start=time.monotonic();metrics=core.update(policy,optimizer,cfg,episodes,out,iteration);update_seconds=time.monotonic()-update_start
        metrics.update(team_soups=team_soups,rollout_seconds=rollout_seconds,agent_moves_per_second=cfg['group_size']*2*cfg['rollout_steps']/rollout_seconds,environment_ticks_per_second=cfg['group_size']*cfg['rollout_steps']/rollout_seconds,update_seconds=update_seconds)
        if iteration%cfg['eval_every']==0:
            policy.model.eval();evaluation=rollout_duo(policy,prompt,0,cfg['eval_steps'],sample=False)
            core.atomic_json(out/'latest-eval.json',dict(iteration=iteration,**evaluation));score=evaluation['soups']*1000+evaluation['team_return']
            if score>best:
                best=score;policy.save(out/'best-eval-adapter');core.atomic_json(out/'best-eval.json',dict(iteration=iteration,score=score,soups=evaluation['soups']))
            metrics.update(eval_soups=evaluation['soups'],eval_team_return=evaluation['team_return'])
        elapsed+=time.monotonic()-tic
        metrics.update(iteration=iteration,reward_scheme=cfg['reward_scheme'],environment=cfg['environment'],prompt_version=cfg.get('prompt_version'),seconds=time.monotonic()-tic,elapsed_seconds=elapsed,target_seconds=cfg['duration_hours']*3600,max_tokens=policy.metadata['max_tokens'],peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,pid=os.getpid(),agent_moves=cfg['group_size']*2*cfg['rollout_steps'])
        checkpoint=core.save(policy,optimizer,out,iteration,elapsed,best)
        core.atomic_json(out/'latest-rollouts.json',dict(iteration=iteration,episodes=episodes))
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        core.atomic_json(out/'status.json',dict(phase='updated',checkpoint=str(checkpoint),**metrics));print('UPDATE',json.dumps(metrics),flush=True)
        if args.max_updates and count>=args.max_updates:break
    phase='complete' if elapsed>=cfg['duration_hours']*3600 else 'stopped';core.atomic_json(out/'status.json',dict(phase=phase,iteration=iteration,elapsed_seconds=elapsed,target_seconds=cfg['duration_hours']*3600,pid=os.getpid()));print(phase.upper(),iteration,elapsed,flush=True)


if __name__=='__main__':main()
