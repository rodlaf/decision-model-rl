"""Run one greedy, two-chef evaluation and save every prompt and action score."""
import argparse
from pathlib import Path
import torch,yaml
from kitchen import ROOT
from policy import Policy,Prompt
from train import rollout_duo
from single_eval import atomic_json

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',default='models/trained',help='Adapter directory, or base for pretrained weights')
    p.add_argument('--config',default='configs/duo.yaml')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--output',type=Path,default=ROOT/'runs/evaluation.json')
    p.add_argument('--iteration',type=int,default=220,help='Label recorded in the trace')
    args=p.parse_args();torch.set_num_threads(4)
    cfg=yaml.safe_load((ROOT/args.config).read_text());policy=Policy(cfg)
    if args.adapter!='base':policy.load(ROOT/args.adapter)
    policy.model.eval()
    result=rollout_duo(policy,Prompt(cfg),args.seed,args.steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,dict(iteration=args.iteration,**result))
    print({k:v for k,v in result.items() if k!='trace'})
