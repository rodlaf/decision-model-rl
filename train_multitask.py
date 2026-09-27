"""Alternate fresh on-policy updates on both games, using one adapter and optimizer."""
import argparse,hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch,yaml
import train_single as core
import train as cooking
import train_rware as warehouse
from kitchen import ROOT
from policy import Policy,Prompt as CookingPrompt
from rware import Prompt as WarehousePrompt


def activate(policy,cfg):
    policy.cfg=cfg;policy.actions=list(cfg['actions'])
    if hasattr(policy,'_nli_tokens'):policy._nli_tokens.clear()


def task_for(iteration,schedule):
    return schedule[(iteration-1)%len(schedule)]


def evaluate(policy,tasks,settings,out,iteration):
    policy.model.eval();results={}
    for name,cfg in tasks.items():
        activate(policy,cfg)
        if name=='overcooked':result=cooking.rollout_duo(policy,CookingPrompt(cfg),0,cfg['eval_steps'])
        else:result=warehouse.episode(policy,WarehousePrompt(cfg),cfg,0,cfg['eval_steps'])
        results[name]=result;core.atomic_json(out/f'eval-{name}.json',dict(iteration=iteration,**result))
    summary=dict(iteration=iteration,overcooked_soups=results['overcooked']['soups'],rware_deliveries=results['rware']['deliveries'],rware_completed_returns=results['rware']['completed_returns'],rware_pickups=results['rware']['pickups'],rware_mean_return=float(np.mean(results['rware']['returns'])))
    with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    print('EVAL',json.dumps(summary),flush=True)
    from render import export
    from render_rware import render
    export(out/'eval-overcooked.json',iteration,ROOT/settings['video_dir']/'overcooked',ROOT/settings['overcooked_config'])
    render(dict(iteration=iteration,**results['rware']),ROOT/settings['video_dir']/'rware'/f'checkpoint-{iteration:06d}.mp4',tasks['rware'])
    return summary


def load_state(policy,optimizer,source):
    policy.load(source)
    saved=torch.load(source/'training_state.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(saved['optimizer'])
    for param,state in optimizer.state.items():
        if 'step' in state:state['step']=state['step'].to(param.device)
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/multitask.yaml');p.add_argument('--max-updates',type=int);p.add_argument('--skip-evaluation',action='store_true');args=p.parse_args()
    settings=yaml.safe_load((ROOT/args.config).read_text());out=ROOT/settings['run_dir'];out.mkdir(parents=True,exist_ok=True)
    tasks={name:yaml.safe_load((ROOT/settings[name+'_config']).read_text()) for name in ['overcooked','rware']}
    if set(settings['schedule'])!=set(tasks):raise ValueError('Schedule must train both games')
    random.seed(settings['seed']);np.random.seed(settings['seed']);torch.manual_seed(settings['seed']);torch.set_num_threads(4)
    signal.signal(signal.SIGINT,core.stop);signal.signal(signal.SIGTERM,core.stop)
    policy=Policy(tasks['rware']);optimizer=cooking.optimizer_for(policy,tasks['rware'])
    latest=out/'latest-checkpoint.json';resume=latest.exists()
    source=Path(json.loads((latest if resume else ROOT/settings['source_run']/'latest-checkpoint.json').read_text())['path'])
    saved=load_state(policy,optimizer,source)
    iteration=saved['iteration'] if resume else 0;elapsed=saved['elapsed_seconds'] if resume else 0.
    task_updates=saved.get('task_updates',dict(overcooked=0,rware=0)) if resume else dict(overcooked=0,rware=0)
    if resume:
        for name,value in saved['task_entropy'].items():tasks[name]['entropy_coefficient']=value
    else:
        tasks['rware']['entropy_coefficient']=saved['entropy_coefficient']
        core.atomic_json(out/'lineage.json',dict(source=str(source),source_iteration=saved['iteration'],adapter_sha256=hashlib.sha256((source/'adapter_model.safetensors').read_bytes()).hexdigest(),optimizer='continued from source',schedule=settings['schedule'],shared_adapter=True))
    core.atomic_json(out/'config.json',dict(settings=settings,tasks=tasks));core.atomic_json(out/'model.json',policy.metadata)
    print('START',json.dumps(dict(source=str(source),resume=resume,iteration=iteration,schedule=settings['schedule'],duration_hours=settings['duration_hours'])),flush=True)
    bests=json.loads((out/'best-scores.json').read_text()) if (out/'best-scores.json').exists() else {}
    def keep_bests(evaluation):
        cook=evaluation['overcooked_soups'];returned=evaluation['rware_completed_returns'];delivered=evaluation['rware_deliveries']
        scores=dict(overcooked=[cook,returned,delivered],rware=[returned,delivered,cook],balanced=[min(cook/6,returned/2),cook/6+returned/2,delivered])
        for name,score in scores.items():
            if name not in bests or score>bests[name]:
                bests[name]=score;policy.save(out/f'best-{name}-adapter');core.atomic_json(out/f'best-{name}.json',evaluation)
        core.atomic_json(out/'best-scores.json',bests)
    if not resume and not args.skip_evaluation:
        baseline=evaluate(policy,tasks,settings,out,0);core.atomic_json(out/'baseline.json',baseline);keep_bests(baseline)
    count=0
    while not core.STOP and (settings['duration_hours'] is None or elapsed<settings['duration_hours']*3600):
        tic=time.monotonic();iteration+=1;count+=1;name=task_for(iteration,settings['schedule']);cfg=tasks[name]
        activate(policy,cfg);policy.model.eval();task_updates[name]+=1
        core.atomic_json(out/'active-task.json',dict(iteration=iteration,task=name,task_updates=task_updates))
        start=time.monotonic()
        if name=='overcooked':
            episodes,soups=cooking.collect(policy,CookingPrompt(cfg),cfg,task_updates[name],out);rollout=dict(soups=soups)
        else:episodes,rollout=warehouse.collect(policy,WarehousePrompt(cfg),cfg,task_updates[name],out)
        rollout['rollout_seconds']=time.monotonic()-start
        print('ROLLOUT',iteration,name,json.dumps(rollout),flush=True)
        start=time.monotonic();metrics=core.update(policy,optimizer,cfg,episodes,out,iteration)
        metrics.update(rollout,update_seconds=time.monotonic()-start,task=name,iteration=iteration,task_update=task_updates[name],agent_moves_per_second=cfg['group_size']*cfg['num_agents']*cfg['rollout_steps']/rollout['rollout_seconds'])
        elapsed+=time.monotonic()-tic
        checkpoint=core.save(policy,optimizer,out,iteration,elapsed,0.,extra_state=dict(task_entropy={name:c['entropy_coefficient'] for name,c in tasks.items()},task_updates=task_updates.copy()))
        if iteration%settings['eval_every']==0 and not args.skip_evaluation:
            start=time.monotonic();evaluation=evaluate(policy,tasks,settings,out,iteration);keep_bests(evaluation);metrics.update(evaluation);elapsed+=time.monotonic()-start
        metrics.update(elapsed_seconds=elapsed,max_tokens=policy.metadata['max_tokens'],peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        core.atomic_json(out/'status.json',dict(phase='updated',checkpoint=str(checkpoint),**metrics));print('UPDATE',json.dumps(metrics),flush=True)
        if args.max_updates and count>=args.max_updates:break
    core.atomic_json(out/'status.json',dict(phase='stopped' if core.STOP or args.max_updates else 'complete',iteration=iteration,elapsed_seconds=elapsed))

if __name__=='__main__':main()
