import gym
import numpy as np
import matplotlib.pyplot as plt
from telic_ambiguity_env import TelicAmbiguityEnv
from scipy.stats import entropy

# Configuration
EPISODES = 15
STEPS = 550
VARIANCE = 0.28
BETA = 0.015
NOISE_STD = 0.06
ENTROPY_BOOST_STEP = 450
ENTROPY_BOOST_FACTOR = 0.02
RES_WINDOW = 30

def rolling_entropy(window):
    def func(state_history):
        if len(state_history) < window:
            return 0.0
        counts = np.bincount(state_history[-window:], minlength=env.observation_space.n)
        probs = counts / counts.sum()
        return entropy(probs, base=2)
    return func

def run_episode(env):
    obs = env.reset()
    state_hist = [obs]
    φ_t_vals, entropy_vals, alignments = [], [], []

    for t in range(STEPS):
        # Simple random agent with entropy-boost logic
        if t == ENTROPY_BOOST_STEP:
            env.φ_t += ENTROPY_BOOST_FACTOR
        if t >= 400:
            env.φ_t += np.random.normal(0, NOISE_STD)

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        state_hist.append(obs)

        # Track φ(t) and entropy over rolling window
        φ_t_vals.append(env.φ_t)
        entropy_vals.append(rolling_entropy(RES_WINDOW)(state_hist))
        alignments.append(info.get("alignment", 0))

        if done:
            break

    return φ_t_vals, entropy_vals, alignments

env = TelicAmbiguityEnv(variance=VARIANCE, beta=BETA)
φ_runs, ent_runs, align_runs = [], [], []

for i in range(EPISODES):
    φ, e, a = run_episode(env)
    φ_runs.append(φ)
    ent_runs.append(e)
    align_runs.append(a)

# Plot example run
plt.figure(figsize=(10, 5))
plt.plot(φ_runs[0], label='φ(t)')
plt.plot(ent_runs[0], label='Rolling Entropy')
plt.title("Run 1 – φ(t) and Entropy Over Time")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Value")
plt.grid(True)
plt.show()
