# Telic Ambiguity Environment

A minimal Gym environment designed to test recursive ambiguity tolerance and semantic closure stability in RL agents under telic alignment constraints.

## Files

- `telic_ambiguity_env.py`: Custom environment definition
- `random_agent_demo.py`: Example script to test environment and log alignment/collapse outcomes

## Goal

Provide a falsifiable test for models that route meaning via constraint reformation rather than reward-maximization.  
See discussion with @grok for technical context and simulation results.

## Grok Protocol Simulation

This simulation replicates the protocol discussed with Grok:

- 15 episodes
- 550 steps
- φ(t) updates with variance = 0.28
- Gaussian noise injection at step 400 (std = 0.06)
- Entropy boost at step 450 (factor = 0.02)
- Rolling entropy window = 30
- Tracks KL divergence and φ(t) as stability metrics

Metrics and visualizations will be added in the `/results` folder.
