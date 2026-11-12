# Development notes

## Changes in this portfolio version

The source files are based on the uploaded V7 and V12 scripts. Long banners and repetitive comments were removed, docstrings shortened, and formatting normalized. The CLI description was changed to a project description. Executable syntax trees were compared after excluding docstrings and that description change.

## Items to check before reporting results

- Some vector-length calculations use `torch.norm(tensor, -1)`. Here `-1` is the norm order. Review these calls and use an explicit dimension where a per-environment length is intended.
- Episode energy adds the already cumulative energy at every step. Correct this before reporting joules.
- Reset environments retain their pre-reset next observations for the following action. Refresh observations after reset.
- Curriculum goals are clamped into fixed goal regions, so the stated stage distance ranges may not be achievable.
- S3 obstacle position arrays need to be written to simulated objects and detected by the sensors.
- The fourth action has no corresponding yaw control in the force application.
- The safety addition to the actor loss depends on detached observations, so it supplies no actor gradient.
- The MMD cross-kernel calculation needs separate squared norms for the two inputs.
- The CLIP projection needs an explicit training/fixed-feature decision and checkpoint handling.
- Sensor failures currently return fallback observations. Validate actual sensor data before interpreting training.
- Check environment offsets, collision geometry, model mass randomization, episode time limits, and SAC wrapper setup in a working simulation.

No runtime or benchmark results were produced during the readability cleanup. The source's existing learning and simulation logic remains the starting point for these checks.

## V12 review

V12 is the featured implementation. V7 is retained as an earlier snapshot. Both files received a readability cleanup and executable AST comparison. Runtime testing still requires the environment builder and compatible assets.

Additional V12 issues include a 30 m goal radius overlapping its 15–40 m first stage, hindsight replay using the same post-step position for both ends of a transition, and relabeled rewards omitting several goal-dependent terms. Collision termination is also discarded by the simplified relabeling.

SPL uses straight-line distance as its reference and leaves first-episode start distances at zero. Path-length and energy metrics are further affected by vector norm and accumulation issues. Time limits are applied after replay insertion, so explicit truncation handling needs review.

V12's smoothness term measures action changes. Its power penalty and energy accumulation use different timestep conventions. Fixed-label CLIP features require validation and checkpoint support. These are implementation details to explain and test before claiming measured benefits.
