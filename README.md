

# Humanoid Robot Power Kick – Reinforcement Learning Task (Isaac Lab)

This repository contains the **task logic, reward shaping, termination conditions, observation builder, configuration files, and unit tests** for a humanoid robot *power-kick* reinforcement learning task designed for NVIDIA Isaac Lab / Isaac Sim.

The objective is:

Train a humanoid to deliver a high-impulse kick to a ball while maintaining balance, using physics-based reward shaping and large-scale PPO with vectorized simulation.

The full Isaac Sim environment, assets, and USD files are not included due to size and licensing constraints.
This repository focuses on the **core reinforcement learning components** that define the task.



---

## Task Overview

### Goal

Learn a policy that:

* Approaches the ball efficiently
* Delivers a powerful kick measured via ball kinetic energy
* Maintains dynamic balance after impact
* Remains robust under randomized initial conditions

### Design Principles

* Reward shaping emphasizes **power**, **uprightness**, and **task progression**
* A deterministic **kick latch** detects the first true kick event
* Termination conditions enforce stability without preventing aggressive motion
* Observations include joint states, velocities, gravity projection, and relative ball geometry

---

## Reward Structure

Reward logic is implemented in `humanoid_kick/reward.py` and parameterized via `configs/task.yaml`.

Key components:

### Power Reward

Rewarded via the increase in ball kinetic energy:

```
r_power = log(1 + k · ΔE)
```

### Uprightness Reward

Based on torso alignment with the world up vector.
Clamped to zero below a configurable threshold.

### Approach Shaping

Encourages the kicking foot to move toward the ball prior to contact:

```
r_approach = exp(-α · distance)
```

### Penalties

* Base angular velocity penalty
* Control effort penalty

### Kick Latch

A latch mechanism ensures:

* The kick event triggers once per episode
* Power reward is gated to real contact events
* False positives are suppressed

Latch behavior is verified via unit tests.

---

## Termination Conditions

Defined in `humanoid_kick/termination.py`.

An episode terminates if:

* Base height drops below a threshold
* Uprightness falls below a threshold
* Maximum step count is reached
* The robot clearly falls

All thresholds are configurable via YAML.

---

## Observation Vector

Constructed in `humanoid_kick/obs.py`.

Includes:

* Joint positions and velocities
* Base linear and angular velocities
* Projected gravity
* Relative ball position
* Ball velocity
* Foot-to-ball relative position

Observation components are toggled via configuration.

---

## PPO Training Configuration

PPO hyperparameters are defined in `configs/ppo.yaml`.

Key settings:

* Total timesteps: 20 million
* Parallel environments: 4096
* Rollout length: 32
* Discount factor: 0.99
* GAE lambda: 0.95
* Clipping range: 0.2
* Learning rate: 3e-4

PPO defaults are kept stable while reward shaping and termination logic are tuned.

---

## Unit Tests

Behavior-level tests validate:

* Kick latch triggers exactly once
* Power reward is gated correctly
* Uprightness reward clamps properly
* Termination logic behaves as expected

Run tests with:

```bash
pytest tests/
```

---

## Sanity Runner

`train.py` provides a logic-only sanity check that:

* Builds observations
* Computes rewards
* Evaluates termination logic
* Confirms configuration consistency

Run with:

```bash
python train.py
```

---

## Why the Full Environment Is Not Included

Isaac Sim environments involve large assets, binary files, and licensing constraints.
This repository isolates the **algorithmically relevant components**:

* Reward design
* Termination logic
* Observation structure
* Configuration management
* Testable behavior

The code is structured to integrate cleanly into an Isaac Lab task.

---
