# Dataset Generation Catalog (2026-07-19, hai-machine)

This catalog classifies the dataset and diagnostic scripts added during the
July 2026 experiments. Files remain in `scripts/` and `configs/` because some
entrypoints resolve helper scripts from those exact paths.

## Conventions

- `formal`: reusable LeRobot dataset collector.
- `preview`: small LeRobot rollout used for visual review before a formal run.
- `probe`: controlled experiment for identifying a controller or contact issue.
- `analysis`: reads rollout metadata and produces measurements or plots.
- `build`: derives a new LeRobot dataset or lookup table from existing data.
- `ops`: resume, repair, watch, or environment launch utility.
- Generated datasets, videos, logs, BDDL files, and local runtimes are not Git artifacts.
- Machine-specific additions carry the `hai-machine` suffix.

## Fixed-scene push-box datasets

### Established collectors

- `collect_libero_push_box_formal_6fric_50pair_35_35_direct_lerobot_hai-machine.py`: formal six-friction visible/invisible paired dataset.
- `collect_libero_push_box_event_tap_segmented80_10action_lerobot_2026-07-05_hai-machine.py`: formal segmented-friction event-tap dataset.
- `collect_libero_push_box_70fric_9action_fixed_scene_hidden_lerobot_hai-machine.py`: formal fixed-scene nine-action friction sweep.
- `collect_libero_push_box_15fric_interp_from_6fric_midA210_lerobot_2026-07-08_hai-machine.py`: formal 15-friction interpolation rollout.

### New reusable collectors

- `collect_libero_push_box_20fric_30peak_fixed16_event_tap_lerobot_2026-07-16_hai-machine.py`: 20-friction by 30-action fixed-contact sweep.
- `collect_libero_push_box_mu023_030_fric80_9action_lerobot_2026-07-16_hai-machine.py`: high-friction extension using the fric80 action family.
- `collect_libero_push_box_board_touch_20fric_30action_fixed5cm_A050_lerobot_2026-07-17_hai-machine.py`: board-touch dataset with 5 cm commanded travel and capped action.
- `collect_libero_push_box_board_touch_20fric_30action_full8_A450_lerobot_2026-07-17_hai-machine.py`: eight-frame full-action board-touch sweep.
- `collect_libero_push_box_board_touch_mu0050_30action_fixed5cm_absolute_eef_xyz_lerobot_2026-07-17_hai-machine.py`: absolute-EEF action dataset at fixed friction.

### Push/contact probes

- `collect_libero_push_box_board_touch_fixed_travel_probe_lerobot_2026-07-17_hai-machine.py`
- `collect_libero_push_box_board_touch_fixed5cm_high_force_probe_lerobot_2026-07-17_hai-machine.py`
- `collect_libero_push_box_board_touch_fixed5cm_mu015_A050_matrix_lerobot_2026-07-17_hai-machine.py`
- `collect_libero_push_box_board_touch_fixed5cm_ramp_latched_brake_probe_lerobot_2026-07-17_hai-machine.py`
- `collect_libero_push_box_fric80_noboard_A026_3friction_stability_probe_lerobot_2026-07-19_hai-machine.py`

The matching `configs/libero_push_box_*.json` files define these sweeps. Probe
configs are retained for reproducibility but are not defaults for formal data.

## LIBERO-plus randomized push-box previews

- `collect_libero_plus_push_box_full_trajectory_preview_lerobot_2026-07-18_hai-machine.py`: common full-trajectory preview engine, including event-latched stopping.
- `collect_libero_plus_push_box_native_gripper_preview_lerobot_2026-07-18_hai-machine.py`: native-gripper backgrounds and action previews.
- `collect_libero_plus_push_box_official_assets_full_trajectory_preview_lerobot_2026-07-18_hai-machine.py`: official-scene and official-asset preview path.
- `analyze_libero_plus_push_box_matched_action_velocity_2026-07-19_hai-machine.py`: matched-action velocity analysis across targets, angles, and friction.
- `run_with_libero_plus_2026-07-18_hai-machine.sh`: launches scripts with the shared LIBERO/LIBERO-plus environment.

The `configs/libero_plus_push_box_*.json` files cover official backgrounds,
lighting, target-shape diversity, clutter, directional actions, and matched
action comparisons. `configs/libero_plus_runtime_2026-07-18_hai-machine/` and
`configs/libero_runtime_2026-07-18_hai-machine/` hold machine-local-compatible
runtime configuration without vendoring either runtime.

## Robot action-space datasets

### Collectors

- `collect_libero_imagined_micro_action_demos_12eps_lerobot_2026-07-16_hai-machine.py`
- `collect_libero_joint_position_multi_pose_100eps_lerobot_2026-07-16_hai-machine.py`
- `collect_libero_various_actions_200eps_lerobot_2026-07-15_hai-machine.py`
- `collect_libero_workspace_rich_eef_300eps_lerobot_2026-07-16_hai-machine.py`
- `collect_libero_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine.py`

### Build and operations

- `build_libero_workspace_rich_eef_absolute_xyz_action_400eps_lerobot_2026-07-16_hai-machine.py`: derives absolute EEF XYZ actions.
- `calibrate_libero_eef_pose_transition_graph_2026-07-16_hai-machine.py`: transition calibration helper.
- `resume_libero_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine.py`: interrupted-run continuation.
- `repair_libero_workspace_rich_eef_400eps_metadata_2026-07-16_hai-machine.py`: metadata-only repair.
- `watch_libero_workspace_rich_eef_400eps_2026-07-16_hai-machine.py`: progress monitor.
- `render_libero_action_world_model_diverse_demos_2026-07-15_hai-machine.py`: render-only action-world-model examples.

## Dynamic-carrier derived datasets

- `build_dynamic_carrier_first20_eef_absolute_xyz_action_lerobot_2026-07-18_hai-machine.py`
- `build_dynamic_carrier_first20_execute_only_eef_absolute_xyz_action_lerobot_2026-07-18_hai-machine.py`

These are build tools, not simulator collectors. They preserve source episodes
while deriving alternative action representations.

## Two-box dynamics datasets

### Formal collectors

- `collect_libero_two_box_collision_9speed_20mass_lerobot_2026-07-16_hai-machine.py`
- `collect_libero_two_box_collision_9speed_20mass_linear_theory_distance_lerobot_2026-07-17_hai-machine.py`
- `collect_libero_two_box_mass_friction_balanced100env_9action_lerobot_2026-07-18_hai-machine.py`
- `collect_libero_two_box_mass_friction_boundary_3mass_3friction_9action_lerobot_2026-07-18_hai-machine.py`

Their paired `configs/libero_two_box_*.json` files define mass, friction,
action, and boundary grids.

### Lookup and render utilities

- `build_libero_two_box_distance_lookup_60frame_2026-07-17_hai-machine.py`
- `build_libero_two_box_postimpact_speed_lookup_2026-07-17_hai-machine.py`
- `render_libero_two_box_collision_mass_demo_2026-07-16_hai-machine.py`
- `render_libero_two_box_high_speed_extreme_mass_demo_2026-07-16_hai-machine.py`

## Render-only push-box diagnostics

- `render_libero_push_box_gripped_board_alignment_probe_2026-07-17_hai-machine.py`
- `render_libero_push_box_gripped_board_gap_vs_touch_probe_2026-07-17_hai-machine.py`
- `render_libero_push_box_gripped_board_touch_duration_4_5_6_8_probe_2026-07-17_hai-machine.py`
- `render_libero_push_box_gripped_board_touch_full_first_step_probe_2026-07-17_hai-machine.py`
- `render_libero_push_box_gripped_board_touch_wide_A_sweep_2026-07-17_hai-machine.py`
- `render_libero_push_box_hybrid_force_control_mass_demo_2026-07-16_hai-machine.py`
- `render_libero_push_box_mass_oracle_force_and_lowkp_mu0040_2026-07-16_hai-machine.py`
- `render_libero_push_box_mass_sweep_mu0040_mid40cm_2026-07-16_hai-machine.py`
- `render_libero_push_box_mass_sweep_mu0080_mid40cm_2026-07-16_hai-machine.py`

These scripts produce diagnostics only and must write under `outputs/` or a
temporary directory, never into a formal LeRobot dataset directory.

## Recommended workflow

1. Select a formal collector and copy its dated config for a new machine or experiment.
2. Run a three-case or small-grid LeRobot preview using the same control path.
3. Inspect contact, alignment, action encoding, image resolution, and metadata.
4. Start the formal collector only after the preview configuration is accepted.
5. Keep generated artifacts out of Git and commit the exact script/config pair.
