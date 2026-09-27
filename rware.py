"""Native 27-value RWARE observations and five primitive controls."""
import ctypes as C
import numpy as np
from kitchen import ROOT

ACTIONS=['wait','move forward','turn left','turn right','toggle load']
DIRECTIONS=['right','down','left','up']
NEIGHBORS=['up','up-right','right','down-right','down','down-left','left','up-left']
OFFSETS=[(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
TILES=['boundary','empty floor','shelf','requested shelf','delivery goal']
LOADS=['nothing','a requested shelf','an emptied shelf']

class Warehouse:
    def __init__(self,seed,num_agents=2,requests=2):
        self.num_agents=num_agents
        self.lib=C.CDLL(str(ROOT/'rware_bridge.so'))
        self.lib.rw_create.argtypes=[C.c_uint,C.c_int,C.c_int];self.lib.rw_create.restype=C.c_void_p
        self.lib.rw_close.argtypes=[C.c_void_p]
        self.lib.rw_step.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
        self.lib.rw_obs.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_float)]
        self.lib.rw_reward.argtypes=[C.c_void_p,C.c_int];self.lib.rw_reward.restype=C.c_float
        self.lib.rw_state.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
        self.ptr=self.lib.rw_create(seed,num_agents,requests)
        if not self.ptr:raise RuntimeError('Invalid RWARE configuration')
        self.deliveries=0;self.returns=0;self.pickups=0
    def close(self):
        if self.ptr:self.lib.rw_close(self.ptr);self.ptr=None
    def observation(self,agent):
        buf=(C.c_float*27)();self.lib.rw_obs(self.ptr,agent,buf);return np.array(buf,dtype=np.float32)
    def reward(self,agent):return float(self.lib.rw_reward(self.ptr,agent))
    def step(self,actions):
        assert len(actions)==self.num_agents and all(0<=a<5 for a in actions)
        before=[int(self.observation(i)[2]) for i in range(self.num_agents)]
        self.lib.rw_step(self.ptr,(C.c_int*self.num_agents)(*actions))
        after=[int(self.observation(i)[2]) for i in range(self.num_agents)]
        self.pickups+=sum(a==0 and b==1 for a,b in zip(before,after))
        self.deliveries+=sum(a==1 and b==2 for a,b in zip(before,after))
        self.returns+=sum(a==2 and b==0 for a,b in zip(before,after))
    def state(self):
        buf=(C.c_int*(110+3*self.num_agents))();self.lib.rw_state(self.ptr,buf)
        return dict(tiles=list(buf[:110]),robots=[list(buf[110+3*i:113+3*i]) for i in range(self.num_agents)])

def decode(obs):
    pos=int(round(float(obs[0])*110));direction=int(round(float(obs[1])*4))-1
    return dict(position=(pos%10,pos//10),direction=DIRECTIONS[direction],load=LOADS[int(obs[2])],neighbors=[(TILES[int(round(float(obs[5+3*j])*4))],DIRECTIONS[int(round(float(obs[4+3*j])*4))-1] if obs[3+3*j] else None) for j in range(8)])

def outcome(before,after,action,reward):
    a,b=decode(before),decode(after);changes=[]
    if a['position']!=b['position']:
        offset=tuple(y-x for x,y in zip(a['position'],b['position']))
        tile=a['neighbors'][OFFSETS.index(offset)][0]
        changes.append(f"moved to {b['position']} onto {tile}")
    if a['direction']!=b['direction']:changes.append('now facing '+b['direction'])
    if a['load']!=b['load']:changes.append('now carrying '+b['load'])
    if reward:changes.append(f'reward {reward:g}')
    return ACTIONS[action]+': '+(', '.join(changes) or 'no observed change')+'.'

class Prompt:
    def __init__(self,cfg):self.cfg=cfg
    def build(self,obs,history):
        d=decode(obs)
        neighbors='\n'.join(f"{name}: {tile}"+(f', robot facing {robot}' if robot else '')+'.' for name,(tile,robot) in zip(NEIGHBORS,d['neighbors']))
        history=' '.join(history[-self.cfg['history_window']:]) or 'None yet.'
        return (self.cfg['rules'].strip()+f"\nYou are at {d['position']}, facing {d['direction']}, carrying {d['load']}.\n"
                +neighbors+'\nRecent actions, oldest first: '+history+'\nWhich button should you press now?')
