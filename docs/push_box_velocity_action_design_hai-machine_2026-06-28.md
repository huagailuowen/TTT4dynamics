# Push-box velocity action design notes for hai-machine, 2026-06-28

## Scope

This note summarizes the design discussion after generating and auditing the hidden-target 9-friction push-box dataset on hai-machine.

The purpose is not to replace the existing dataset immediately. The purpose is to clarify why the current action representation is difficult for a VLA to learn, and what a more learnable action/control interface should look like for fast dynamic pushing.

## Current dataset status

The generated dataset contains 450 hidden-target episodes:

- 9 friction values.
- 50 episodes per friction value.
- 225 straight and 225 angled pushes.
- Short/mid/long displacement quotas follow the previous ratio: 14/20/16 per friction.

The dataset is balanced according to the configured buckets, but the quality audit shows several issues for VLA training.

## Main quality issues observed

### 1. Long episodes contain too much non-push motion

The current episode structure is roughly:

```text
approach -> descend -> push -> retreat -> settle
```

The fixed pre-push portion is large:

```text
approach: 30 frames
descend: 40 frames
pre-push total: 70 frames
```

The actual push window is much shorter:

```text
mean push frames: about 13.8
push frames <= 8: many episodes
push frames <= 12: most episodes
```

This is a core VLA training problem. If the full episode is used for imitation, the model spends most of its action loss on approach, descend, retreat, and settle rather than on the physically important fast pushing behavior.

The dataset should therefore be reorganized around the push/contact window.

A better training clip should look like:

```text
pre-contact context: 10-20 frames
push/contact window: full window
post-contact observation: 20-40 frames
```

Action loss should focus on the pre-contact and push/contact window. Post-contact frames can still provide useful visual dynamics, but the robot action after release is weakly coupled to the sliding box and should be downweighted or masked.

### 2. Some long-displacement samples are too far for reliable vision/action learning

Long displacement is not inherently wrong. Dynamic pushing often requires the object to slide for a while after contact. However, the current dataset includes samples whose final displacement is too large for stable visual learning.

The audit found examples above 50 cm, with a maximum around 56 cm. In several keyframes, the object is very close to the edge of the agent-view image or visually hard to track.

The design conclusion is:

```text
Longer dynamic pushes are useful.
But the dataset should cap displacement to stay inside a visually learnable range.
```

Suggested rule:

```text
preferred displacement: 10-40 cm
manual-review band: 40-45 cm
hard cap: 45 cm
exclude: >45 cm
```

This keeps dynamic sliding behavior while avoiding out-of-frame or near-boundary examples.

### 3. The current action label is not semantically clean for fast pushing

The current action is a low-level controller command, not a clean physical velocity label. For dynamic pushes, especially impulse-like cases, the action often appears as a short high-amplitude controller spike.

This does not necessarily express the behavior we want the model to learn.

The intended behavior is:

```text
Move the EEF into contact, then push quickly in a controlled direction with a controlled speed for a short time.
```

The current label instead tends to encode:

```text
Controller-specific transient command + retreat + object sliding under inertia.
```

This makes the mapping from image to action sparse and temporally misaligned.

### 4. Apparent repeated pushing is partly a representation/rendering gap

Some trajectories visually look like repeated pushing. The metadata does not show multiple push phases. Each trajectory has a single push phase.

The apparent repeated-push effect likely comes from a gap among:

```text
low-level controller action
EEF motion
rendered image frames
object dynamics after contact
```

The robot may already be retreating or settling while the object continues sliding. The renderer displays the object motion, but the current action label at those frames no longer directly describes pushing the object.

For per-frame imitation, this creates ambiguity:

```text
The image still changes because the box is sliding.
The robot action may already be retreating.
The model is asked to imitate an action that no longer causes the visible object motion.
```

This gap is one reason the training clips should be contact-centric and why action loss after release should be reduced.

## Why position-only is not enough

The goal is not merely to learn slow, quasi-static position pushing.

The desired behavior is fast dynamic pushing:

```text
The robot should learn to push at a controlled speed over a short contact window, causing the object to slide to a desired region.
```

The current `impulse` mode is one implementation of fast dynamic pushing, but it is not necessarily the action representation we want the model to imitate directly.

Therefore, the distinction should be:

```text
Task behavior: fast controlled dynamic push.
Current implementation: position mode or impulse mode in the low-level controller.
Desired learning interface: EEF velocity or segmented push primitive.
```

Position-only data would likely make the model better at slow contact pushing, but it may remove the core dynamic behavior we care about.

## Proposed direction: EEF velocity action

A better action label for dynamic pushing is EEF velocity.

Possible action format:

```text
[vx, vy, vz, dax, day, daz, gripper]
```

or:

```text
[vx, vy, vz, wx, wy, wz, gripper]
```

The key change is that translation should become velocity-like. Rotation can remain close to the current representation if that is simpler for compatibility.

The model would learn:

```text
which direction to push
how fast to push
how long to maintain that velocity
when to release or stop pushing
```

This is closer to the physical semantics of dynamic pushing than a controller-internal position/action spike.

## Important limitation: velocity action does not remove the short effective window

Changing the label to EEF velocity does not make the task long-horizon by itself.

The effective control window is still short:

```text
pre-contact: approach/control alignment
contact: high-value push velocity
post-contact: object slides mostly under physics
```

Thus, even with velocity action, full-episode imitation is still suboptimal. The model would still see long stretches where the action is not the causal driver of the sliding object.

Velocity action should be paired with:

- contact-centric clips,
- phase metadata,
- action loss masks,
- or a segmented controller.

## Alternative or complementary direction: segmented control

Another promising design is to use phase-conditioned or segmented control.

Example phases:

```text
phase 1: approach-to-contact
phase 2: fast velocity push
phase 3: release / retreat / observe
```

The model may predict either:

```text
phase-specific low-level actions
```

or a higher-level push primitive:

```text
contact point
push direction
target EEF speed
push duration
release timing
```

A low-level controller then executes the primitive.

This is attractive because the core decision in dynamic pushing is not every frame of controller action. The core decision is the push direction, speed, duration, and contact timing.

## Recommended training representation

For the next dataset or relabeling pass, the recommended representation is:

```text
observation: contact-centric visual clip
action: EEF velocity chunk or segmented push primitive
metadata: push_start, push_end, release frame, displacement, friction, target visibility
loss mask: strong around contact, weak or zero after release
```

Suggested clip construction:

```text
clip_start = push_start - 10 to 20 frames
clip_end = push_end + 20 to 40 frames
```

Suggested action-loss weighting:

```text
pre-contact alignment: 1.0
push/contact: 1.0
post-release sliding: 0.0 to 0.2
retreat-only frames: usually exclude or low weight
```

## Data-quality filters for the next pass

Suggested filters:

```text
displacement <= 45 cm
prefer 10-40 cm
object remains inside agent-view margin
push/contact window is explicitly recorded
avoid extremely short unstructured spikes unless represented as a primitive
exclude frames where action is retreat but visual motion is dominated by box inertia from action loss
```

The existing long-displacement design can remain, but it should be capped. Dynamic behavior is useful; out-of-frame or almost-out-of-frame behavior is not.

## Implications for inference code

If the training action is changed to EEF velocity, inference must change as well.

Future code changes would likely include:

```text
1. Dataset writer/relabeler
   Convert EEF pose differences to velocity labels.

2. Environment wrapper
   Add an EEF velocity control mode.

3. Inference wrapper
   Convert VLA-predicted velocity chunks into executable low-level controls.

4. Normalization
   Recompute action normalization for velocity units and clip ranges.

5. Optional segmented execution
   Let the model predict phase or push primitive parameters, then execute with a low-level velocity controller.
```

This document does not implement these changes. It only records the proposed direction and the reasoning behind it.

## Summary

The current dataset is useful for diagnosing the problem, but it is not yet ideal for VLA training.

The main issue is not that dynamic pushing or impulse-like behavior is wrong. The issue is that the current low-level action representation and full-episode layout make the important contact window too sparse and too hard to learn.

A better direction is:

```text
Keep fast dynamic pushing.
Cap displacement at about 45 cm.
Use contact-centric clips.
Represent push actions as EEF velocity or segmented push primitives.
Reduce or mask non-push frames in the loss.
Update inference to execute velocity or primitive commands.
```
