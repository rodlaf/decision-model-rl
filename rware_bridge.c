// Native PufferLib RWARE. Contiguous buffers match its reset/step contract.
#include "ocean/rware/rware.h"
Env* rw_create(unsigned int seed, int agents, int requests) {
    if (agents < 1 || agents > MAX_AGENTS || requests < 1 || requests > 32) return NULL;
    Env* e=calloc(1,sizeof(Env));
    e->rng=seed;e->map_choice=1;e->num_agents=agents;e->num_requested_shelves=requests;
    float *obs=calloc(agents*OBS_SIZE,sizeof(float)), *act=calloc(agents,sizeof(float));
    float *reward=calloc(agents,sizeof(float)), *done=calloc(agents,sizeof(float));
    for(int i=0;i<agents;i++) {
        e->agents[i].observations=obs+i*OBS_SIZE;e->agents[i].actions=act+i;
        e->agents[i].rewards=reward+i;e->agents[i].terminals=done+i;
    }
    init(e);compute_observations(e);return e;
}
void rw_close(Env* e) {
    if(!e)return;
    puf_close(e);free(e->agents[0].observations);free(e->agents[0].actions);
    free(e->agents[0].rewards);free(e->agents[0].terminals);free(e);
}
void rw_step(Env* e,int* actions) {
    for(int i=0;i<e->num_agents;i++)e->agents[i].actions[0]=actions[i];
    puf_step(e);
}
void rw_obs(Env* e,int agent,float* out){memcpy(out,e->agents[agent].observations,OBS_SIZE*sizeof(float));}
float rw_reward(Env* e,int agent){return e->agents[agent].rewards[0];}
// Full state is for recording/rendering only, never a policy input.
void rw_state(Env* e,int* out) {
    memcpy(out,e->warehouse_states,110*sizeof(int));
    for(int i=0;i<e->num_agents;i++) {
        out[110+3*i]=e->agent_locations[i];out[111+3*i]=e->agent_directions[i];out[112+3*i]=e->agent_states[i];
    }
}
