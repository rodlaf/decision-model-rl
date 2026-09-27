"""Native 27-value RWARE observations and five primitive controls."""
from collections import deque
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

class ObservationMemory:
    """Per-robot episode memory built exclusively from that robot's observations."""
    def __init__(self,history_window=8):
        self.tick=0;self.tiles={};self.shelf_sites={};self.goals={}
        self.history=deque(maxlen=history_window)
    def observe(self,obs):
        d=decode(obs);x,y=d['position']
        for (dx,dy),(tile,robot) in zip(OFFSETS,d['neighbors']):
            pos=(x+dx,y+dy)
            if tile=='boundary':continue
            self.tiles[pos]=(tile,self.tick)
            if tile=='delivery goal':self.goals[pos]=self.tick
            # A shelf seen without a robot is a storage-site observation.
            # Carried shelves do not create fictitious storage sites.
            if tile in ('shelf','requested shelf') and robot is None:self.shelf_sites[pos]=self.tick
    def record(self,before,after,action,reward):
        self.tick+=1
        self.history.append(f"Step {self.tick}: "+outcome(before,after,action,reward))
        self.observe(after)
    def describe(self,obs,limit=6):
        d=decode(obs)
        def locations(items):
            rows=sorted(items.items(),key=lambda item:(-item[1],item[0]))
            text=', '.join(f"({p[0]},{p[1]}) [age {self.tick-t}]" for p,t in rows[:limit]) or 'none seen'
            return text+(f'; {len(rows)-limit} older locations omitted' if len(rows)>limit else '')
        requested={p:t for p,(tile,t) in self.tiles.items() if tile=='requested shelf' and not (p==d['position'] and d['load']!='nothing')}
        return ("Map memory from your observations only. Ages are steps since last seen; shelves may have moved.\n"
                +'Requested shelves: '+locations(requested)+'.\n'
                +'Delivery goals: '+locations(self.goals)+'.\n'
                +'Shelf storage sites: '+locations(self.shelf_sites)+'.')

class Prompt:
    def __init__(self,cfg):self.cfg=cfg
    def build(self,obs,memory):
        memory.observe(obs);d=decode(obs);x,y=d['position']
        neighbors='\n'.join(f"{name} ({x+dx},{y+dy}): {tile}"+(f', robot facing {robot}' if robot else '')+'.' for name,(dx,dy),(tile,robot) in zip(NEIGHBORS,OFFSETS,d['neighbors']))
        history='\n'.join(memory.history) or 'None yet.'
        return (self.cfg['rules'].strip()+f"\nYou are at {d['position']}, facing {d['direction']}, carrying {d['load']}.\n"
                +neighbors+'\n'+memory.describe(obs,self.cfg.get('memory_locations_per_type',6))
                +'\nRecent actions, oldest first:\n'+history+'\nWhich button should you press now?')
