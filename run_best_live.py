import time
import torch

from anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING import (
    Env,
    Agent,
    pick_latest_checkpoint,
)

policy_path = pick_latest_checkpoint()
print("Cargando mejor checkpoint:")
print(policy_path)

env = Env(fall_threshold=0.3)

policy = torch.load(policy_path, map_location="cpu", weights_only=False)
policy.eval()

env.attach_viewer()

state = env.reset()
ep_reward = 0.0
episode = 1

try:
    while env.viewer is not None and env.viewer.is_running():
        with torch.no_grad():
            action, _, _ = policy.compute_action(state)

        state, reward, done = env.step(action, render=False)
        ep_reward += reward

        time.sleep(0.01)

        if done:
            print(f"Episodio {episode} | reward: {ep_reward:.2f}")
            episode += 1
            ep_reward = 0.0
            state = env.reset()

finally:
    env.detach_viewer()
