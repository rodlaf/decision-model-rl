"""Native PufferLib environment access. State getters are for replay/logging only."""
import ctypes as C
import random
from pathlib import Path
ROOT = Path(__file__).resolve().parent
EMPTY, COUNTER, STOVE, INGREDIENT, SERVING, WALL, PLATE = 0, 1, 2, 4, 5, 6, 7
NO_ITEM, ONION, DISH, PLATED_SOUP = 10, 12, 13, 15
HELD = {NO_ITEM: "nothing", ONION: "an onion", DISH: "a plate", PLATED_SOUP: "plated soup"}
TILE = {EMPTY: "floor", COUNTER: "counter", STOVE: "pot", INGREDIENT: "onion source",
        SERVING: "serving counter", WALL: "wall", PLATE: "plate source"}


class Kitchen:
    def __init__(self, layout=0, seed=None, num_agents=1):
        lib = C.CDLL(str(ROOT / "overcooked_bridge.so"))
        lib.bridge_create.argtypes = [C.c_int]
        lib.bridge_create.restype = C.c_void_p
        lib.bridge_create_agents.argtypes = [C.c_int, C.c_int]
        lib.bridge_create_agents.restype = C.c_void_p
        lib.bridge_destroy.argtypes = [C.c_void_p]
        lib.bridge_step.argtypes = [C.c_void_p, C.c_int, C.c_int]
        lib.bridge_set_start.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
        lib.bridge_set_start.restype = C.c_int
        for name, args in {
            "width": [], "height": [], "dishes": [], "num_agents": [], "x": [C.c_int], "y": [C.c_int],
            "facing": [C.c_int], "held": [C.c_int], "held_soup_onions": [C.c_int], "pot_count": [C.c_int],
            "pot_state": [C.c_int], "pot_progress": [C.c_int],
            "tile": [C.c_int, C.c_int], "pot_at": [C.c_int, C.c_int],
            "item_type": [C.c_int, C.c_int],
        }.items():
            f = getattr(lib, "bridge_" + name)
            f.argtypes = [C.c_void_p] + args
            f.restype = C.c_int
        self.lib = lib
        self.ptr = lib.bridge_create_agents(layout, num_agents)
        if not self.ptr:
            raise RuntimeError("Could not create Overcooked env")
        self.num_agents = self.q("num_agents")
        self.steps = 0
        if seed is not None:
            w, h = self.q("width"), self.q("height")
            other = (self.q("x", 1), self.q("y", 1)) if self.num_agents > 1 else None
            floors = [(x, y) for y in range(h) for x in range(w)
                      if self.q("tile", x, y) == EMPTY and (x, y) != other]
            rng = random.Random(seed)
            x, y = rng.choice(floors)
            facing = rng.randrange(4)
            if not lib.bridge_set_start(self.ptr, x, y, facing):
                raise RuntimeError("Could not set seeded chef start")
            self.start = {"position": [x, y], "facing": facing}
        else:
            self.start = {"position": [self.q("x", 0), self.q("y", 0)], "facing": self.q("facing", 0)}

    def close(self):
        self.lib.bridge_destroy(self.ptr)

    def q(self, name, *args):
        return getattr(self.lib, "bridge_" + name)(self.ptr, *args)

    def step_joint(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        self.lib.bridge_step(self.ptr, actions[0], actions[1] if self.num_agents > 1 else 0)
        self.steps += 1

    def step(self, action):
        self.step_joint([action] + ([0] if self.num_agents > 1 else []))

    def state(self, agent=0):
        if not 0 <= agent < self.num_agents:
            raise IndexError(agent)
        w, h = self.q("width"), self.q("height")
        pots = []
        features = []
        for y in range(h):
            for x in range(w):
                tile = self.q("tile", x, y)
                if tile == STOVE:
                    i = self.q("pot_at", x, y)
                    pots.append({"position": [x, y], "onions": self.q("pot_count", i),
                                 "state": ["idle", "cooking", "ready"][self.q("pot_state", i)],
                                 "cook_progress": self.q("pot_progress", i)})
                elif tile != EMPTY:
                    features.append(f"{TILE.get(tile, tile)} at ({x},{y})")
                item = self.q("item_type", x, y)
                if item != NO_ITEM:
                    features.append(f"{HELD.get(item, item)} on counter at ({x},{y})")
        result = {"chef": {"position": [self.q("x", agent), self.q("y", agent)],
                          "facing": ["up", "down", "left", "right"][self.q("facing", agent)],
                          "holding": HELD[self.q("held", agent)],
                          "held_soup_onions": self.q("held_soup_onions", agent)},
                "pots": pots, "map_features": features, "dishes_served": self.q("dishes")}
        if self.num_agents > 1:
            other = 1 - agent
            result["other_chef"] = {"position": [self.q("x", other), self.q("y", other)],
                                    "facing": ["up", "down", "left", "right"][self.q("facing", other)],
                                    "holding": HELD[self.q("held", other)], "behavior": "model-controlled"}
        return result
