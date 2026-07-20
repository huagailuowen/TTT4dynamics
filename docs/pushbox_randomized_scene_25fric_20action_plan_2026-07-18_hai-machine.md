# Randomized-scene PushBox 25-friction x 20-action plan

Date: 2026-07-18
Machine: hai-machine
Status: preview only; do not collect the 500-episode formal dataset before review.

## Dataset topology

- 25 friction groups, using the segmented density ratio from the earlier fric80 dataset.
- 20 fixed action IDs at every friction, for 500 formal episodes total.
- One unique visual-domain seed per `(friction_id, action_id)` pair.
- Hidden target, straight PushBox task, 224 x 224 agent and wrist videos at 20 FPS.
- LeRobot action schema uses absolute EEF XYZ targets and relative axis-angle rotation.

## Friction schedule

- 9 values in `[0.002, 0.05)`.
- 13 values in `[0.05, 0.15)`.
- 3 values in `[0.15, 0.20]`.

This is the 30:40:10 fric80 density ratio rounded to 25 groups.

## Action schedule

- Action IDs 0-8: centered +x pushes with A = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50.
- Action IDs 9-19: fixed directional actions covering +/-30, +/-60, +/-90, +/-135, reverse pushes, and one off-center glancing push.
- Every action begins with 0.7A, reaches A, brakes once, and then latches to zero.
- The EEF projected travel target is 0.05 m for every action direction.
- Action ID semantics do not change across friction groups. Randomness belongs to appearance and initial state, not action identity.

## Object variants

All targets use primitive box collision geoms. No sphere or cylinder target is used.

- `standard_box`: approximately 8.2 x 4.2 x 1.8 cm.
- `wide_box`: approximately 6.8 x 5.4 x 2.0 cm.
- `slim_box`: approximately 9.0 x 3.6 x 2.0 cm.

The compiled body mass and inertia are preserved across variants. Contact parameters and friction assignment are identical; only box dimensions and visual color change.

## Initial-state randomization

- Front/back x jitter: +/-0.012 m.
- Image-horizontal y jitter: +/-0.080 m.
- The tool approach target is recomputed from the measured object pose after reset.
- The preparation trajectory and gentle touch are not recorded.

## Visual randomization

- Random floor, table, wall, and table-leg colors.
- Existing textures are independently retained or disabled.
- Two lights receive randomized position, diffuse color, ambient color, and specular intensity.
- Object color is sampled with a contrast constraint against the table.
- Camera poses, robot geometry, collision geoms, gravity, and control frequency remain fixed.

## Directional pusher

The gripped rectangular board is centered at the EEF for this dataset. Its broad face is rotated to align with the selected planar action direction. Gripper collision is disabled against the target so the board remains the only pusher contact geom.

## Required quality gates

- A sampled board-object contact must occur before recording starts.
- Maximum projected EEF travel must be between 0.040 and 0.065 m.
- Absolute perpendicular EEF drift must remain below 0.012 m.
- Board-face alignment error must remain below 3 degrees.
- The action may change sign once for braking, but cannot accelerate again afterward.
- Secondary contact episodes and box lateral/forward motion are recorded in episode metadata.
- Formal collection must be rejected if preview videos show tool/object visual overlap, unreachable reverse approaches, or unstable diagonal impacts.

## Preview

The preview contains nine LeRobot episodes, balanced across the three box variants and covering straight, diagonal, lateral, backward-diagonal, and reverse pushes. Each episode uses a different background, lighting setup, initial position, friction, and action.

## 2026-07-18 validated preview findings

Status: preview only. Formal 25-friction x 20-action collection has **not** started.

### Final object variants

All three target colliders stay inside the original compile-time collider bounds. This is required because expanding \`geom_size\` at runtime can be rejected by MuJoCo's compile-time broad phase even when an OBB overlap test is positive.

| object | world dimensions (length x width x height) | body dynamics |
|---|---:|---|
| \`standard_box\` | about 8.0 x 4.2 x 1.76 cm | original mass and inertia |
| \`wide_box\` | about 6.4 x 4.2 x 1.76 cm | original mass and inertia |
| \`slim_box\` | about 8.0 x 3.0 x 1.76 cm | original mass and inertia |

Only collision/visual dimensions and visual color change. Mass, inertia, friction/contact parameters, and solver settings are unchanged. The object's z pose is adjusted by the exact change in vertical support so every variant rests on the same table plane.

### Randomization used by the preview

- Table, floor, table legs, and walls receive independently sampled materials/colors.
- Texture enablement, light colors, and light positions are randomized.
- Object color is sampled with a minimum RGB contrast from the table.
- Object front/back x jitter is +/-1.2 cm.
- Object image-horizontal y jitter is +/-8.0 cm.
- The robot approaches the measured object pose and measured directional support; it never pushes toward a nominal fixed coordinate.

### Contact preparation guarantee

MuJoCo contact can begin and end inside one control step, so end-of-step \`ncon\` alone is insufficient. The preview uses a dual event latch:

- Trigger on board-box contact, at least 0.1 mm projected box motion, or at least 0.002 m/s projected box speed.
- After the first event, restore a static touch pose with a 20-micron overlap before recording.
- Set target free-joint velocity to zero before the first recorded action.
- Record the trigger source, touch steps, geometry, and launch speed in episode metadata.

All nine validated episodes start with measured planar box speed 0.0 m/s. Touch preparation required 7-9 control steps.

### Distance semantics

The hard command-side guarantee is pusher EEF travel, not final free-sliding box distance:

- Nominal projected EEF travel: 5.0 cm.
- Accepted measured projected EEF travel: 4.0-7.0 cm.
- Maximum perpendicular EEF drift: 1.2 cm.
- Maximum board-normal alignment error: 3 degrees.

Final box displacement must remain a physical outcome of action, friction, shape, and contact. Artificially braking the box would corrupt the world-model target. The selected review demos stay between 9.13 cm and 38.13 cm. A complete fixed-action grid at very low friction can exceed 45 cm; if 45 cm must be a hard box limit, the action must become friction-conditioned, the camera/view must change, or extreme pairs must be rejected.

### Validated nine-episode preview

| ep | object | mu | action | angle | A | measured EEF | box displacement | pass |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | standard | 0.02 | 0 | 0 deg | 0.05 | 5.60 cm | 9.13 cm | yes |
| 1 | wide | 0.08 | 4 | 0 deg | 0.25 | 4.86 cm | 18.84 cm | yes |
| 2 | slim | 0.15 | 8 | 0 deg | 0.50 | 5.94 cm | 25.96 cm | yes |
| 3 | standard | 0.05 | 9 | +30 deg | 0.18 | 5.46 cm | 18.05 cm | yes |
| 4 | wide | 0.10 | 11 | +60 deg | 0.25 | 4.81 cm | 15.14 cm | yes |
| 5 | slim | 0.05 | 13 | +90 deg | 0.30 | 6.78 cm | 38.13 cm | yes |
| 6 | standard | 0.15 | 14 | -90 deg | 0.30 | 6.49 cm | 17.48 cm | yes |
| 7 | wide | 0.08 | 15 | +135 deg | 0.22 | 4.16 cm | 15.62 cm | yes |
| 8 | slim | 0.05 | 17 | 180 deg | 0.25 | 5.28 cm | 20.83 cm | yes |

The preview is a normal-resolution 224 x 224, dual-camera, absolute-EEF-action LeRobot dataset. The preview collector now raises an error immediately when any episode fails a quality gate.

### Proposed formal grid after approval

- Friction: 25 values following the fric80 density ratio: 9 in [0.002, 0.05), 13 in [0.05, 0.15), and 3 in [0.15, 0.20].
- Actions per friction: 20 fixed IDs.
- Straight actions: IDs 0-8 with A = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50.
- Directional actions: +30, -30, +60, -60, +90, -90, +135, -135, 180 (two strengths), and one +15-degree glancing contact.
- Total: 500 episodes, hidden/no target, one LeRobot dataset, friction and all randomization parameters in per-episode metadata.
- Formal collection remains blocked on visual review and the decision about extreme low-friction/high-A box travel.

### Camera observability finding

Visual review found a remaining formal blocker:

- The external agent view is clear for straight, +30/+60-degree, and lateral pushes.
- +135-degree and 180-degree pushes move the box behind the pusher/robot, so the latter part is occluded in agent view.
- The current wrist view does not recover the box; it is dominated by the board and table horizon.

Recommendation: before formal collection, replace the second camera with a fixed overhead or opposite-side workspace camera and preview +135, -135, and 180 degrees again. The alternative is to remove strong backward actions. Do not start the 500-episode formal collection while these actions are visually unobservable.
