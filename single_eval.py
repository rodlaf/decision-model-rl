"""Single-chef evaluation and atomic JSON output."""
import json,time
import torch
from policy import NativeEnv,ACTIONS,observation_delta

def rollout(policy,prompt,seed,steps,sample=True):
    env=NativeEnv(seed);history=[];trace=[];total=0.;start=time.perf_counter()
    try:
        previous=env.state()
        for t in range(steps):
            obs=env.observation();state=prompt.build(obs,history)
            decision_start=time.perf_counter()
            with torch.no_grad():logp=policy.distribution(state)
            probs=logp.exp();action=int(torch.multinomial(probs,1)) if sample else int(probs.argmax())
            latency_ms=(time.perf_counter()-decision_start)*1000
            env.step(action);now=env.state();next_obs=env.observation()
            reward=env.native_reward()
            total+=reward;report=observation_delta(obs,next_obs,ACTIONS[action]);history.append(report)
            trace.append(dict(state=state,observation=obs.tolist(),action=action,old_logp=float(logp[action]),old_distribution=probs.cpu().tolist(),latency_ms=latency_ms,reward=reward,native_reward=env.native_reward(),outcome=report,before=previous,after=now))
            previous=now
        expected=sum(t['native_reward'] for t in trace)
        assert abs(total-expected)<1e-5
        return dict(seed=seed,return_=total,soups=env.q('dishes'),steps=steps,seconds=time.perf_counter()-start,trace=trace)
    finally:env.close()

def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
