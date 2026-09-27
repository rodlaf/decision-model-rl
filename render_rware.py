"""Recorded native warehouse states with each robot's observation and action scores."""
import subprocess
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw
from decision_video import BG,PANEL,TEXT,MUTED,ACCENT,font,wrapped
from rware import ACTIONS


def render(episode,path,cfg):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.mp4')
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x900','-r','5','-i','-','-an','-c:v','libx264','-preset','fast','-threads','2','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    try:
        for event in episode['trace']:
            im=Image.new('RGB',(1440,900),BG);d=ImageDraw.Draw(im);state=event['before']
            d.text((28,24),f"Step {event['tick']+1}    Delivered {event['deliveries']}    Returned {event['returns']}",font=font(22),fill=TEXT)
            size=62;ox=35;oy=100;colors=['#182334','#456172','#ac4547','#68a779']
            for i,tile in enumerate(state['tiles']):
                x,y=ox+i%10*size,oy+i//10*size
                d.rectangle((x,y,x+size-2,y+size-2),fill=colors[tile])
                if tile in (1,2):d.rectangle((x+12,y+14,x+size-14,y+size-16),outline=TEXT,width=2)
                if tile==3:d.text((x+10,y+20),'GOAL',font=font(13),fill=BG)
            for robot,(pos,facing,load) in enumerate(state['robots']):
                x,y=ox+pos%10*size+size//2,oy+pos//10*size+size//2;color=ACCENT if robot==0 else '#8fb8f4'
                d.ellipse((x-20,y-20,x+20,y+20),fill=BG,outline=color,width=4)
                dx,dy=[(1,0),(0,1),(-1,0),(0,-1)][facing];d.line((x+dx*12,y+dy*12,x+dx*27,y+dy*27),fill=color,width=5)
                d.text((x-5,y-10),str(robot),font=font(16),fill=TEXT)
                if load:d.rectangle((x-23,y+23,x+23,y+29),fill='#ed6466' if load==1 else '#ffc46b')
            d.text((35,812),'Requested shelves are red. Delivery goals are green.',font=font(16),fill=MUTED)
            d.text((35,845),f"Checkpoint {episode['iteration']}    Seed {episode['seed']}    5 ticks/s",font=font(16),fill=MUTED)
            for a in range(2):
                x0,y0=700,18+a*444;color=ACCENT if a==0 else '#8fb8f4'
                d.rounded_rectangle((x0,y0,1420,y0+430),radius=14,fill=PANEL)
                d.text((x0+18,y0+14),f"Robot {a}: {ACTIONS[event['actions'][a]]}",font=font(22),fill=color)
                dynamic=event['states'][a].split('\n',1)[1].split('\nRecent actions, oldest first: ')[0]
                d.text((x0+18,y0+56),'Native observation',font=font(15),fill=MUTED)
                wrapped(d,dynamic,(x0+18,y0+84),380,15,TEXT,4)
                d.text((x0+420,y0+56),'Action scores',font=font(15),fill=MUTED)
                for j,(name,prob) in enumerate(zip(ACTIONS,event['probabilities'][a])):
                    y=y0+90+j*53
                    d.text((x0+420,y),name,font=font(15),fill=TEXT)
                    d.text((x0+647,y),f'{prob:.0%}',font=font(15),fill=color)
                    d.rectangle((x0+420,y+26,x0+687,y+33),fill='#27364c')
                    if prob:d.rectangle((x0+420,y+26,x0+420+267*prob,y+33),fill=color)
                d.text((x0+18,y0+386),'Game rules and eight recent outcomes also enter the prompt.',font=font(14),fill=MUTED)
            writer.stdin.write(im.tobytes())
    finally:
        writer.stdin.close();code=writer.wait()
    if code:raise RuntimeError('Video export failed')
    temp.replace(path);print('VIDEO',path,flush=True)
