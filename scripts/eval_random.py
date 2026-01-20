"""
Placeholder: shows intended API shape.
Once you wire into Isaac Lab, replace DummyEnv with the actual environment.
"""

import numpy as np

class DummyEnv:
    def __init__(self, action_dim: int = 16):
        self.action_dim = action_dim
    def reset(self):
        return np.zeros(10, dtype=np.float32)
    def step(self, action):
        obs = np.zeros(10, dtype=np.float32)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

def main():
    env = DummyEnv()
    obs = env.reset()
    for t in range(200):
        action = np.random.uniform(-1, 1, size=(env.action_dim,)).astype(np.float32)
        obs, r, done, info = env.step(action)
        if done:
            obs = env.reset()
    print("Eval script ran (dummy). Wire to Isaac Lab env when available.")

if __name__ == "__main__":
    main()
