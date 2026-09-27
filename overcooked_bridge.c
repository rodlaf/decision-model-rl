// Minimal headless bridge to PufferLib 5.0's actual Overcooked implementation.
// Built by ./build_overcooked_bridge.sh against a pinned PufferLib 5.0 checkout.
#include "ocean/overcooked/overcooked.h"

Overcooked* bridge_create_agents(int layout, int num_agents) {
    if (num_agents < 1 || num_agents > 2) return NULL;
    Overcooked* env = calloc(1, sizeof(Overcooked));
    if (!env) return NULL;
    env->layout_id = (LayoutType)layout;
    env->num_agents = num_agents;
    env->observation_size = OBS_SIZE;
    env->rewards_config.dish_served_whole_team = 1.0f;
    env->rewards_config.pot_started = 0.10f;
    env->rewards_config.ingredient_added = 0.10f;
    env->rewards_config.ingredient_picked = 0.0f;
    env->rewards_config.soup_plated = 0.20f;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = calloc(OBS_SIZE, sizeof(float));
        env->agents[i].actions = calloc(1, sizeof(float));
        env->agents[i].rewards = calloc(1, sizeof(float));
        env->agents[i].terminals = calloc(1, sizeof(float));
    }
    init(env);
    puf_reset(env);
    return env;
}

Overcooked* bridge_create(int layout) { return bridge_create_agents(layout, 1); }

void bridge_destroy(Overcooked* env) {
    if (!env) return;
    puf_close(env);
    for (int i = 0; i < env->num_agents; i++) {
        free(env->agents[i].observations);
        free(env->agents[i].actions);
        free(env->agents[i].rewards);
        free(env->agents[i].terminals);
    }
    free(env);
}

void bridge_reset(Overcooked* env) { puf_reset(env); memset(&env->log, 0, sizeof(Log)); }
int bridge_set_start(Overcooked* env, int x, int y, int facing) {
    if (x < 0 || y < 0 || x >= env->width || y >= env->height ||
        env->grid[y * env->width + x] != EMPTY ||
        (is_agent_at(env, x, y) && !((int)env->chefs[0].x == x && (int)env->chefs[0].y == y)) ||
        facing < 0 || facing > 3) return 0;
    clear_agent_position(env, (int)env->chefs[0].x, (int)env->chefs[0].y);
    env->chefs[0].x = x;
    env->chefs[0].y = y;
    env->chefs[0].facing_direction = facing;
    set_agent_position(env, x, y);
    compute_observations(env);
    return 1;
}
void bridge_step(Overcooked* env, int a0, int a1) {
    env->agents[0].actions[0] = (float)a0;
    if (env->num_agents > 1) env->agents[1].actions[0] = (float)a1;
    puf_step(env);
}
int bridge_num_agents(Overcooked* env) { return env->num_agents; }
int bridge_width(Overcooked* env) { return env->width; }
int bridge_height(Overcooked* env) { return env->height; }
int bridge_tile(Overcooked* env, int x, int y) { return env->grid[y*env->width+x]; }
int bridge_x(Overcooked* env, int i) { return (int)env->chefs[i].x; }
int bridge_y(Overcooked* env, int i) { return (int)env->chefs[i].y; }
int bridge_facing(Overcooked* env, int i) { return env->chefs[i].facing_direction; }
int bridge_held(Overcooked* env, int i) { return env->chefs[i].held_item; }
int bridge_held_soup_onions(Overcooked* env, int i) { return env->chefs[i].held_soup_onions; }
int bridge_pot_count(Overcooked* env, int i) { return env->cooking_pots[i].ingredient_count; }
int bridge_pot_state(Overcooked* env, int i) { return env->cooking_pots[i].cooking_state; }
int bridge_pot_progress(Overcooked* env, int i) { return env->cooking_pots[i].cooking_progress; }
int bridge_pot_at(Overcooked* env, int x, int y) { return env->pot_index_grid[y*env->width+x]; }
int bridge_item_type(Overcooked* env, int x, int y) {
    Item* item = get_item_at(env, x, y);
    return item ? item->type : NO_ITEM;
}
int bridge_dishes(Overcooked* env) { return (int)env->log.correct_dishes; }
float bridge_reward(Overcooked* env, int i) { return env->agents[i].rewards[0]; }
// Copy the actual public observation; training policy never reads hidden state.
void bridge_observation(Overcooked* env, int agent, float* out) {
    memcpy(out, env->agents[agent].observations, OBS_SIZE * sizeof(float));
}
