# Dynamic Carrier: Raw FastWAM vs Observe-20 TTT

Date: 2026-06-06

This note summarizes the current dynamic-carrier evaluation after tightening the grasp attachment rule and adding a pickup metric. The main result is that TTT does improve the policy, especially pickup behavior, but the full pick-and-place task remains difficult for both methods under the current environment and success criteria.

## Task Setup

The task is a dynamic carrier manipulation problem:

1. Track a moving carrier platform or open low-rim box.
2. Pick up the cream-cheese payload from the moving carrier.
3. Place the payload on a static target region.

Evaluation uses the observe-then-act metadata:

- Metadata: `FastWAM/data/ttt_dynamic_carrier_observe20_200_lerobot/dynamic_carrier_generation_metadata.json`
- Case range: first 50 cases, `case_start=0`, `num_trials=50`
- Environment variants in the 50 cases:
  - `flat_platform`: 22 cases
  - `open_box_low_rim`: 28 cases
  - `line` carrier motion: 30 cases
  - `irregular_loop` carrier motion: 20 cases
- Control frequency: 20 Hz
- `observe_then_act_chunks=20`
- `observe_then_act_interval=10`
- Observation phase length: `20 * 10 = 200` environment steps, about 10 seconds at 20 Hz
- Model action horizon: 32 actions, from `num_frames=33` and `action_horizon=num_frames-1`
- Runtime replanning: only the first 10 actions from each predicted chunk are executed, then the model replans

## Compared Methods

Both methods use the same 50 cases, same seeds, same observe-then-act timing, same camera setup, and same environment/metric code.

### Raw FastWAM

Checkpoint:

```text
FastWAM/runs/ttt_dynamic_carrier_cream_2cam224_ft/20260531_0527_plain_gpu/checkpoints/weights/step_010000.pt
```

Evaluation behavior:

- Runs the same 20 observe chunks and 10-step interval as TTT for fair timing.
- Uses dummy actions during the observe phase.
- Disables video TTT:
  - `model.video_ttt.enabled=false`
  - `model.video_ttt.observation_training=false`
  - `model.video_ttt.switch_chunks=false`
  - `model.loss.lambda_video_ttt=0.0`
- After observation, it starts normal action inference and replans every 10 environment steps.

### Observe-20 TTT

Checkpoint:

```text
FastWAM/runs/ttt_dynamic_carrier_observe20_2cam224_video_ttt/20260604_0127_gpu1_b16_restart/checkpoints/weights/step_017250.pt
```

Evaluation behavior:

- Runs 20 observe chunks with interval 10.
- Uses dummy actions during the observe phase, matching raw.
- Before each observe interval, runs the video-TTT observation update from the current observation.
- Total TTT observation updates per episode: 20.
- After the observe phase, it starts action inference and replans every 10 environment steps.

## Current Environment And Metrics

### Grasp Attachment Rule

The previous attachment rule was too permissive and could make the object snap to the gripper even when the gripper was visually too far away. The current rule is stricter:

- Gripper command must be closing: `gripper_cmd > 0.2`
- XY distance between EEF and payload grasp point must be at most 3.5 cm
- Z distance between EEF and payload grasp point must be at most 5.0 cm

Implementation constants:

```text
grasp_release_distance = 0.035
grasp_release_height = 0.05
```

This reduces the visible teleport/snap artifact during pickup while still allowing a small tolerance for simulation and controller mismatch.

### Pickup Metric

We now separately report pickup success:

```text
pickup_success = episode ever had env.payload_attached_to_gripper == True
```

This metric answers whether the model can actually catch/grasp the moving payload, even if it later fails to place it.

### Task Success Metric

Full task success requires the payload to be released and satisfy the placement check. The placement check is evaluated on either:

1. The current payload pose, or
2. The recorded payload pose at release time.

Using the release pose avoids marking visually successful releases as failures only because the object later penetrates the table or falls through the scene.

The placement pose must satisfy:

- Payload XY is within the target radius: `target_radius = 0.055` m
- Payload is clear of the moving carrier:
  - distance from carrier XY is at least `platform_radius + object_radius`
- Payload Z is plausible:
  - if `target_z` exists, use `target_z_tolerance`
  - otherwise require payload Z to stay within `[carrier_z - 0.20, carrier_z + 0.08]`

The Z sanity check is intended to avoid counting severe table-penetration cases as successful placements.

## 50-Case Evaluation Results

Output directory:

```text
FastWAM/evaluate_results/dynamic_carrier/20260606_attach035_z050_pickup_metric_observe20_vs_raw_step017250_n50_cases000_049
```

Raw run:

```text
raw_plain_observe20_attach035_z050_pickup_metric_n50_cases000_049
```

TTT run:

```text
observe20_step017250_attach035_z050_pickup_metric_n50_cases000_049
```

Exit status:

```text
[eval-exit] rc=0 at 2026-06-06T10:02:52Z
```

### Overall

| Method | Task Success | Pickup Success | Rollout Videos |
|---|---:|---:|---:|
| Raw FastWAM | 0/50 = 0.0% | 5/50 = 10.0% | 50 |
| Observe-20 TTT | 1/50 = 2.0% | 11/50 = 22.0% | 50 |

### By Environment Type

| Method | Environment | Cases | Task Success | Pickup Success |
|---|---|---:|---:|---:|
| Raw FastWAM | flat platform | 22 | 0/22 | 4/22 |
| Raw FastWAM | open box low rim | 28 | 0/28 | 1/28 |
| Observe-20 TTT | flat platform | 22 | 0/22 | 4/22 |
| Observe-20 TTT | open box low rim | 28 | 1/28 | 7/28 |

### By Motion Type

| Method | Motion | Cases | Task Success | Pickup Success |
|---|---|---:|---:|---:|
| Raw FastWAM | line | 30 | 0/30 | 4/30 |
| Raw FastWAM | irregular loop | 20 | 0/20 | 1/20 |
| Observe-20 TTT | line | 30 | 1/30 | 9/30 |
| Observe-20 TTT | irregular loop | 20 | 0/20 | 2/20 |

### By Environment And Motion Pair

| Method | Environment | Motion | Cases | Task Success | Pickup Success |
|---|---|---|---:|---:|---:|
| Raw FastWAM | flat platform | line | 15 | 0/15 | 3/15 |
| Raw FastWAM | flat platform | irregular loop | 7 | 0/7 | 1/7 |
| Raw FastWAM | open box low rim | line | 15 | 0/15 | 1/15 |
| Raw FastWAM | open box low rim | irregular loop | 13 | 0/13 | 0/13 |
| Observe-20 TTT | flat platform | line | 15 | 0/15 | 2/15 |
| Observe-20 TTT | flat platform | irregular loop | 7 | 0/7 | 2/7 |
| Observe-20 TTT | open box low rim | line | 15 | 1/15 | 7/15 |
| Observe-20 TTT | open box low rim | irregular loop | 13 | 0/13 | 0/13 |

## Successful And Pickup Cases

TTT had one full task success:

```text
trial 20
case_id: box_line_medium_speed1.60_v0022_00
first_pickup_step: 245
episode steps: 294
video: observe20_step017250_attach035_z050_pickup_metric_n50_cases000_049/rollout_videos/2026_06_06-09_35_18--episode=trial0020_box_line_medium_speed1.60_v0022_00--success=True--task=track_the_moving_cream_cheese_box_inside_the_open_.mp4
```

Pickup cases:

```text
Raw:
- trial 1:  flat_irregular_loop_medium_speed1.60_v0001_00, first_pickup_step=237
- trial 11: flat_line_medium_speed1.60_v0012_00, first_pickup_step=227
- trial 31: flat_line_medium_speed1.60_v0036_00, first_pickup_step=226
- trial 39: flat_line_medium_speed1.60_v0044_00, first_pickup_step=302
- trial 46: box_line_medium_speed1.60_v0054_00, first_pickup_step=225

TTT:
- trial 1:  flat_irregular_loop_medium_speed1.60_v0001_00, first_pickup_step=234
- trial 5:  box_line_medium_speed1.60_v0006_00, first_pickup_step=225
- trial 15: flat_line_medium_speed1.60_v0016_00, first_pickup_step=271
- trial 20: box_line_medium_speed1.60_v0022_00, first_pickup_step=245, full success
- trial 22: flat_line_medium_speed1.60_v0024_00, first_pickup_step=232
- trial 23: box_line_medium_speed1.60_v0026_00, first_pickup_step=225
- trial 33: box_line_medium_speed1.60_v0038_00, first_pickup_step=225
- trial 36: flat_irregular_loop_medium_speed1.60_v0041_00, first_pickup_step=259
- trial 40: box_line_medium_speed1.60_v0046_00, first_pickup_step=226
- trial 46: box_line_medium_speed1.60_v0054_00, first_pickup_step=225
- trial 49: box_line_medium_speed1.60_v0058_00, first_pickup_step=225
```

## Interpretation

The current result does show a real improvement from TTT:

- Pickup success improves from 10.0% to 22.0%.
- Full task success improves from 0.0% to 2.0%.
- The largest pickup improvement appears in open-box line-motion cases: raw gets 1/15 pickups, while TTT gets 7/15 pickups and the only full success.

However, the full task is still very hard for both methods. Most episodes fail before or after pickup, and many pickups do not become successful placements. This means the task currently combines several hard factors at once:

- moving-object interception,
- precise grasp timing,
- avoiding snap/teleport artifacts with a strict grasp threshold,
- placing onto a static target after a dynamic pickup,
- table/contact stability and object penetration issues.

## Raw Direct Timing Control

After the first 50-case result, we also reran Raw FastWAM on the same first-50
cream metadata cases, but without the observe-then-act wrapper. This controls for
the question of whether the raw model's low pickup rate came from the model
itself or from delaying action by 20 observe chunks.

Output directory:

```text
FastWAM/evaluate_results/dynamic_carrier/20260606_raw_direct_same_prev_env_attach035_z050_n50_cases000_049
```

Run:

```text
raw_plain_direct_same_prev_env_attach035_z050_n50_cases000_049
```

Configuration:

- Same base checkpoint as Raw FastWAM.
- Same first 50 cream metadata cases as the Raw observe-20 run.
- `repeat_eval_from_groups=false`
- `observe_then_act_chunks=0`
- `warmup_passes=0`
- `replan_steps=10`
- grasp XY threshold 3.5 cm and Z threshold 5.0 cm

Result:

| Method | Task Success | Pickup Success | Observe Chunks |
|---|---:|---:|---:|
| Raw FastWAM, observe-20 wrapper | 0/50 = 0.0% | 5/50 = 10.0% | 20 |
| Raw FastWAM, direct execution | 0/50 = 0.0% | 13/50 = 26.0% | 0 |

This confirms that the raw 10.0% pickup number is not purely a property of the
raw checkpoint. On this task, the 200-frame observe delay strongly changes the
execution timing and the carrier phase before policy actions begin. The full
task success remains 0/50 in both raw settings, but the pickup metric is
sensitive to this timing change.

For strict comparisons, Raw direct and observe-then-act TTT should be reported
as different protocols:

- Raw direct measures the original policy's immediate reaction to the dynamic
  scene.
- Raw observe-20 controls for the same delayed action budget used by Type2 TTT.
- Type2 observe-20 tests whether online video-TTT updates during those observed
  chunks help after the same delay.

## OOD Dynamic-Parameter Evaluation

We also created a 50-case eval-only OOD metadata set to compare Raw direct
against Observe-20 Type2 TTT on dynamic parameters outside the original training
range.

Metadata:

```text
FastWAM/data/ttt_dynamic_carrier_ood_eval_50/dynamic_carrier_generation_metadata.json
```

Output directory:

```text
FastWAM/evaluate_results/dynamic_carrier/20260606_ood50_raw_direct_vs_type2_observe20
```

OOD design:

- Same object, same task language, same flat-platform/open-box BDDL scenes.
- Same target region and metric code.
- 50 eval cases, grouped into 5 OOD families with 10 cases each:
  - `fast_period`
  - `slow_period`
  - `large_amplitude`
  - `high_yaw`
  - `novel_shape`
- The OOD parameters move outside the training metadata range:
  - training period range was about 1.55 to 2.66 seconds
  - training X amplitude range was about 0.078 to 0.119 m
  - training Y amplitude range was about 0.0 to 0.080 m
  - training yaw range was about -0.34 to 0.35 radians

Compared methods:

| Method | Checkpoint | Protocol |
|---|---|---|
| Raw FastWAM direct | `ttt_dynamic_carrier_cream_2cam224_ft/.../step_010000.pt` | no observe phase, direct policy execution |
| Type2 Observe-20 TTT | `ttt_dynamic_carrier_observe20_2cam224_video_ttt/.../step_017250.pt` | 20 observe chunks, interval 10, then policy execution |

Overall OOD result:

| Method | Task Success | Pickup Success | Observe Chunks |
|---|---:|---:|---:|
| Raw FastWAM direct | 0/50 = 0.0% | 10/50 = 20.0% | 0 |
| Type2 Observe-20 TTT | 0/50 = 0.0% | 11/50 = 22.0% | 20 |

OOD pickup by family:

| OOD Family | Raw FastWAM Direct | Type2 Observe-20 TTT |
|---|---:|---:|
| fast_period | 3/10 | 2/10 |
| slow_period | 2/10 | 2/10 |
| large_amplitude | 2/10 | 1/10 |
| high_yaw | 2/10 | 2/10 |
| novel_shape | 1/10 | 4/10 |

The OOD result is mixed. Overall pickup is nearly tied, with Type2 only one case
ahead: 22.0% vs 20.0%. The clearest local gain is on `novel_shape`, where Type2
gets 4/10 pickups and Raw direct gets 1/10. This suggests observe-20 adaptation
may be more useful when the trajectory family changes, but this result is still
small and should be treated as an early signal rather than a strong conclusion.

Full task success is still 0/50 for both methods on OOD. That means the current
OOD setting is useful for testing interception and pickup, but remains too hard
for full pick-and-place success under the current policy and simulation metric.

## Short Conclusion

TTT is helping, especially on the intermediate skill of catching or grasping the moving object. The final placement success is still too low to make the full-task metric a sensitive comparison by itself. For clearer TTT-effect demonstrations, the next experiments should either reduce task difficulty or increase the amount/frequency of useful observation for adaptation.

Practical next directions:

- Evaluate easier subsets first, especially open-box line-motion cases where TTT already shows the clearest gain.
- Increase observation frequency or TTT update frequency, for example shorter observe interval than 10 or more observe chunks.
- Try a slower carrier speed or larger target radius as a controlled difficulty sweep.
- Keep reporting both pickup success and full placement success, because pickup success is currently the more informative early signal.
- For OOD analysis, focus on trajectory-family shifts like `novel_shape`, where
  the first OOD sweep showed the clearest Type2 pickup advantage.
