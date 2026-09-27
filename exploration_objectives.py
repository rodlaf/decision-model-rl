"""CPU-only reward/advantage helpers for native-action exploration training."""
import hashlib
import math
import numpy as np


def _flatten_observation(observation):
    if isinstance(observation, dict):
        return np.concatenate([_flatten_observation(observation[key]) for key in sorted(observation)])
    if isinstance(observation, (tuple, list)):
        return np.concatenate([_flatten_observation(item) for item in observation])
    return np.asarray(observation, dtype=np.float64).reshape(-1)


class EpisodicObservationNovelty:
    """Capped first-visit bonus renewed after positive environment rewards."""
    def __init__(self, initial, bonus=0.01, cap=0.2, quantum=0.1, segment_cap=None):
        if segment_cap is None:
            segment_cap = cap
        if quantum <= 0 or bonus < 0 or cap < 0 or segment_cap < 0:
            raise ValueError('Novelty scales must be positive')
        self.bonus, self.cap, self.quantum, self.segment_cap = bonus, cap, quantum, segment_cap
        self.paid = 0.0
        self.segment_paid = 0.0
        self.seen = {self.key(initial)}

    def key(self, observation):
        values = _flatten_observation(observation)
        values = np.nan_to_num(values, nan=0., posinf=1e6, neginf=-1e6)
        rounded = np.rint(np.clip(values / self.quantum, -1e6, 1e6)).astype('<i4')
        return hashlib.blake2b(rounded.tobytes(), digest_size=16).digest()

    def observe(self, observation, extrinsic_reward=0.0):
        key = self.key(observation)
        if extrinsic_reward > 0:
            self.seen = {key}
            self.segment_paid = 0.0
            return 0.0
        if key in self.seen:
            return 0.0
        self.seen.add(key)
        reward = max(0.0, min(self.bonus, self.cap - self.paid, self.segment_cap - self.segment_paid))
        self.paid += reward
        self.segment_paid += reward
        return reward


def discounted_advantages(rewards, gamma=0.99):
    """Finite-horizon reward-to-go, with an independent leave-one-trajectory-out baseline."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[0] < 2:
        raise ValueError('Expected at least two equal-length trajectories')
    returns = np.zeros_like(rewards)
    running = np.zeros(len(rewards))
    for t in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
    baseline = (returns.sum(axis=0, keepdims=True) - returns) / (len(returns) - 1)
    advantages = returns - baseline
    # Shared scale preserves relative credit across time; do not amplify tiny groups.
    advantages /= max(float(advantages.std()), 0.1)
    return advantages


def adapt_entropy(coefficient, entropy, target, rate=0.2, minimum=0.001, maximum=0.2):
    return float(np.clip(coefficient * math.exp(rate * (target - entropy)), minimum, maximum))
