# Urban UAV Navigation

**Abdulazez Abagero · Aerospace engineering researcher**

A personal project exploring goal-directed UAV navigation with reinforcement learning in Isaac Sim and Isaac Lab. The work brings together sensor observations, continuous control, reward design, and training analysis.

[Open portfolio page](index.html) · [Watch training](https://www.youtube.com/watch?v=GY_i5F9RiYg) · [Read V12 code](train_v12.py)

## Demonstration

[![Isaac Lab training video](https://img.youtube.com/vi/GY_i5F9RiYg/hqdefault.jpg)](https://www.youtube.com/watch?v=GY_i5F9RiYg)

The video illustrates my simulation training work. Its exact match to these source snapshots still needs a recorded configuration and checkpoint.

## Project question

How can a UAV learn to move toward a goal while responding to obstacles, limiting abrupt changes in movement commands, and accounting for estimated power consumption?

## Why V12 is featured

V12 offers a clearer development story through revised reward shaping, an expanding start-to-goal curriculum, action-change penalties, and episode time limits. V7 provides the earlier implementation for comparison.

| Design element | V7 | V12 |
| --- | --- | --- |
| Proximity reward | Threshold-based | Exponential distance penalty |
| Action-change penalty | Absent | Included |
| Curriculum | Four configured stages, with fixed goal-region constraints | Five stages, with free goal placement in early stages |
| Arrival radius | 15 m | 30 m |
| Episode time limit | No explicit cap in loop | 4,000 steps |
| Hindsight replay | Absent | Experimental relabeling |
| Path-efficiency logging | Absent | SPL field requiring correction |

This is a design comparison. Neither snapshot has a validated evaluation report in this repository.

## Training workflow

Depth, LiDAR, relative-goal, and battery inputs are assembled into an observation. A policy chooses velocity targets, which a proportional controller converts into simulated external forces. Rewards and transitions are recorded, replay samples update the networks, and checkpoints preserve training progress.

| Component | Implementation |
| --- | --- |
| Depth | 84 × 84 image |
| LiDAR | 64 downsampled readings |
| Goal | Distance and angles from simulator position |
| Battery | State of charge from modeled energy |
| Actions | Four outputs, with three applied as velocity targets |
| Algorithms | Custom TD3, SAC wrapper, experimental TD3 extension |
| Experimental features | Prioritized replay, curiosity, fixed-label CLIP features, critic regularization |

The current code uses ground-truth simulator position. Localization estimation remains a future extension. CLIP compares images with fixed labels rather than accepting mission instructions.

## Urban scenarios

Scenario IDs s1, s2, and s3 select parameters from the external environment builder. Exact building layouts and dimensions require that file. Depth and LiDAR provide inputs for static obstacle awareness. S3 adds wind and maintains moving vehicle and UAV arrays. Connecting those arrays to scene geometry is still pending.

The policy performs local navigation through velocity decisions. Path planning concerns routes through space. Motion planning additionally accounts for timing and physical motion constraints. Global route search and explicit trajectory optimization are further extensions.

## Setup and example command

Required external components are an Isaac Lab installation, compatible quadrotor USD asset, and `env_builder_v7.py`. The environment builder has not yet been included.

Both scripts currently search `~/Desktop/ETD3` for the builder and expect the USD asset at `~/IsaacLab/assets/robots/quadrotor/quadrotor.usd`. Adjust those paths to match your installation.

Python dependencies include PyTorch and NumPy. Experimental branches also use OpenCLIP and Stable-Baselines3/Gymnasium, with optional TensorBoard logging. Exact compatible versions need to be recorded after a successful run.

After the missing dependencies are supplied, this example checks startup and warm-up data collection:

```bash
cd ~/IsaacLab
./isaaclab.sh -p /absolute/path/to/urban-uav-navigation/train_v12.py \
  --algo td3 --scenario s1 --seed 42 --num_envs 1 \
  --total_steps 1000 --headless --enable_cameras \
  --results_dir /absolute/path/to/results
```

This command has not been run during the portfolio preparation. Default V12 settings use 16 environments, 2 million transitions, 25,000 warm-up transitions, and a batch size of 256. Replay memory requirements need checking before larger runs.

## Results and development

The scripts write CSV training statistics, JSON settings and summaries, checkpoints, and optional TensorBoard logs. Current review covers source inspection and syntax checks. Quantitative success, energy, and path-efficiency claims await corrected metrics and fixed-task evaluation.

[Development notes](DEVELOPMENT.md) record the known issues, including vector norms, episode energy accumulation, reset observations, dynamic obstacles, and experimental replay. Code and documentation were prepared with AI assistance.

## Related work

- [Deep Learning-Based MPPT Approach to Enhance CubeSat Power Generation](https://doi.org/10.1109/ACCESS.2025.3546066), IEEE Access, 2025.
- [Accelerated Airfoil Optimization via a Geometry-Based Discriminator Within Particle Swarm Optimization](https://doi.org/10.1109/ACCESS.2026.3683048), IEEE Access, 2026.
- [Crazyflie swarm demonstration](https://youtu.be/aovQiCbUqWQ), a separate hardware activity.

## Publish the portfolio page

Upload this folder's contents to the root of a public repository named `urban-uav-navigation`. In Settings → Pages, select Deploy from a branch, choose main and /(root), then Save. GitHub displays the published URL when deployment completes.

The local README link opens the HTML source on GitHub. Share the deployed Pages URL for the rendered website.

[GitHub Pages setup guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
