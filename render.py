"""Render a recorded two-chef greedy evaluation without running model inference."""
import hashlib,json,subprocess
from pathlib import Path

import imageio_ffmpeg,yaml
from PIL import Image,ImageDraw

from decision_video import BG,PANEL,TEXT,MUTED,ACCENT,font,wrapped,png_url,write_viewer
from kitchen_view import OvercookedView
from kitchen import ROOT
from policy import NativeEnv,Prompt,ACTIONS
from train import joint_outcome


def duo_dashboard(game,event,metrics,meta):
    """Large game view plus the changing input and clean native-action scores."""
    im=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(im)
    game=game.resize((900,1008),Image.Resampling.NEAREST);im.paste(game,(30,36))
    metric_width=d.textbbox((0,0),metrics,font=font(21))[2]
    d.rounded_rectangle((52,54,min(930,88+metric_width),105),radius=13,fill='#0b1020')
    d.text((70,66),metrics,font=font(21),fill=TEXT)
    for agent,(y0,y1) in enumerate(((20,526),(554,1060))):
        color=ACCENT if agent==0 else '#8fb8f4';x0,x1=960,1900
        d.rounded_rectangle((x0,y0,x1,y1),radius=18,fill=PANEL)
        action=ACTIONS[event['agents'][agent]['action']]
        d.text((x0+24,y0+18),f'Chef {agent}',font=font(25),fill=color)
        d.text((x0+128,y0+22),f'chose {action}',font=font(19),fill=TEXT)
        d.text((x0+24,y0+60),f'What Chef {agent} sees',font=font(14),fill=MUTED)
        d.text((x0+500,y0+60),'Action scores',font=font(14),fill=MUTED)
        d.text((x0+500,y0+84),f'“{event["hypothesis_pattern"]}”',font=font(12),fill=MUTED)
        dynamic=event['agent_states'][agent].split('\n',1)[1]
        prompt_y=wrapped(d,dynamic,(x0+24,y0+88),445,12,TEXT,2)
        if prompt_y>y1-18:raise ValueError(f'Chef {agent} observation overflow: {prompt_y}')
        y=y0+115
        for option in event['agent_options'][agent]:
            selected=option['goal']==event['goals'][agent]
            d.text((x0+500,y),option['label'],font=font(14),fill=color if selected else TEXT)
            d.text((x1-68,y),f"{option['probability']*100:4.1f}%",font=font(14),fill=color if selected else TEXT)
            d.rounded_rectangle((x0+500,y+24,x1-24,y+32),radius=3,fill='#27364c')
            if option['probability']>0:
                d.rounded_rectangle((x0+500,y+24,x0+500+(x1-x0-524)*option['probability'],y+32),radius=3,fill=color if selected else '#6481a7')
            y+=62
        d.text((x0+500,y1-31),'Scores are normalized entailment values',font=font(12),fill=MUTED)
    return im


def export(source,iteration,folder,config=ROOT/'configs/duo.yaml'):
    cfg=yaml.safe_load(Path(config).read_text());prompt=Prompt(cfg)
    template=json.loads((ROOT/cfg['model_path']/'config.json').read_text())['nli_template']
    episode=json.loads(Path(source).read_text());assert episode['iteration']==iteration;seed=episode['seed']
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);(folder/'traces').mkdir(exist_ok=True)
    stem=f'checkpoint-{iteration:06d}-seed-{seed}-1x';dest=folder/f'{stem}.mp4';temp=folder/'traces'/f'{stem}.tmp.mp4'
    meta=dict(schema='decision-replay-v1',title=f'Checkpoint {iteration} · two autonomous chefs',subtitle='Native 1× playback',seed=seed,decisions=episode['steps'],model='OpenJev v5 0.8B shared policy',observation_mode='native text per chef',choice_title='Actions',state_title='Exact policy inputs',state_font_size=17,state_line_spacing=3,compact_options=True,controller_description='Both chefs use the same learned policy. One primitive action per chef per step.',probability_note='Scores are normalized entailment values.',checkpoint=dict(iteration=iteration),correct_soups=episode['soups'])
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','32','-i','-','-an','-c:v','libx264','-preset','fast','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    env=NativeEnv(seed,num_agents=2);view=OvercookedView();histories=[[],[]];events=[];frames=[]
    try:
        for i,tick in enumerate(episode['trace']):
            observations=[env.observation(agent) for agent in range(2)]
            states=[prompt.build(observations[agent],histories[agent]) for agent in range(2)]
            agents=tick['agents'];actions=[a['action'] for a in agents]
            for agent in range(2):
                assert states[agent]==agents[agent]['state'];assert env.state(agent)==agents[agent]['before']
            p=env.state(0)['pots'][0];display=(f"Chef 0: {env.state(0)['chef']['position']}, faces {env.state(0)['chef']['facing']}, holds {env.state(0)['chef']['holding']}.\n"
                f"Chef 1: {env.state(1)['chef']['position']}, faces {env.state(1)['chef']['facing']}, holds {env.state(1)['chef']['holding']}.\n"
                f"Pot: {p['onions']}/3 onions, {p['state']}, progress {p['cook_progress']}/20. Soups served: {env.q('dishes')}.\n"
                f"Recent joint outcomes:\n"+'\n'.join(histories[0][-3:] or ['None yet.']))
            options=[];exact=[];goals=[]
            for agent in range(2):
                chosen=f'c{agent}_{ACTIONS[actions[agent]]}';goals.append(chosen)
                for action,probability in zip(ACTIONS,agents[agent]['distribution']):
                    options.append(dict(goal=f'c{agent}_{action}',label=action.capitalize(),effect='one native action',duration=1,probability=probability))
                    exact.append(template.format(premise=states[agent],hypothesis=cfg['hypothesis'].format(action=action)))
            agent_options=[options[:6],options[6:]]
            event=dict(index=i+1,state='\n\n'.join(f'CHEF {agent} PROMPT\n{states[agent]}' for agent in range(2)),display_state=display,agent_states=states,agent_options=agent_options,agents=agents,hypothesis_pattern=cfg['hypothesis'].format(action='[control]'),question='Which action should each chef take?',goal=goals[0],goals=goals,chosen=f'Chef 0 {ACTIONS[actions[0]]}    Chef 1 {ACTIONS[actions[1]]}',latency_ms=tick.get('latency_ms',0.),options=options,exact_prompts=exact)
            events.append(event);game=view.render(env);metrics=f'Step {i:03d}    Soups {env.q("dishes")}'
            board=duo_dashboard(game,event,metrics,meta);raw=board.tobytes()
            encoder.stdin.write(raw);encoder.stdin.write(raw)
            frames.append(dict(time=i/16,event=i,metrics=metrics,image=png_url(game)))
            env.step_joint(actions)
            for agent in range(2):
                assert env.state(agent)==agents[agent]['after']
                report=joint_outcome(observations[agent],env.observation(agent),actions[agent],actions[1-agent]);assert report==agents[agent]['outcome'];histories[agent].append(report)
        assert env.q('dishes')==episode['soups']
        game=view.render(env);metrics=f'Step {episode["steps"]:03d}    Soups {env.q("dishes")}'
        board=duo_dashboard(game,events[-1],metrics,meta)
        for _ in range(32):encoder.stdin.write(board.tobytes())
        frames.append(dict(time=episode['steps']/16,event=len(events)-1,metrics=metrics,image=png_url(game)))
    finally:
        env.close();encoder.stdin.close();code=encoder.wait()
    if code:raise RuntimeError('ffmpeg export failed')
    temp.replace(dest);meta.update(duration=episode['steps']/16+1,verified_decisions=len(events),source_trace_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    write_viewer(folder/f'{stem}.html',meta,events,frames);(folder/'traces'/f'{stem}.json').write_text(json.dumps(dict(meta=meta,episode=episode),indent=2))
    print('VIDEO',dest,'soups',episode['soups'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('iteration',type=int);p.add_argument('--folder',type=Path,default=ROOT/'videos');p.add_argument('--config',type=Path,default=ROOT/'configs/duo.yaml');a=p.parse_args();export(a.source,a.iteration,a.folder,a.config)
