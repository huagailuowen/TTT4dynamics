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

## Fixed 8-step trapezoid impulse profile proposal, 2026-06-29

The current impulse controller uses a sinusoidal pulse:

```text
action_x[t] = action_end * sin(pi * progress)
```

This is physically smooth, but it is not ideal as a learning target. Different `push_steps` produce different curve shapes, and every frame inside the push window has a different value. A VLA must therefore learn phase, duration, and amplitude at the same time, even though the effective contact window is only a few frames.

A more learnable profile is a fixed-length trapezoid pulse:

```text
push_steps = 8
profile = [0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3]
action_x[t] = A * profile[t]
```

Here `A` is the only primary strength parameter. It is still a dimensionless controller action amplitude, not a metric velocity. The measured EEF velocity must still be computed from observations:

```text
v_x ~= (eef_x[t + 1] - eef_x[t]) / dt
```

This profile has several advantages for imitation learning:

```text
1. The push window has a fixed duration.
2. The shape is shared across all impulse examples.
3. The plateau makes the intended fast-push behavior explicit.
4. The scalar A becomes an interpretable proxy for push strength.
5. The model no longer has to fit many sinusoidal variants with different step counts.
```

For dataset generation, the preferred control variables should become:

```text
friction_mu
push direction / angle
A: impulse amplitude
optional contact timing / pre-contact pose
```

The sweep should avoid changing `push_scale` while testing `A`; otherwise the effect of `A` is confounded with controller-output scaling. If this profile is adopted, `pusher_push_action_delta` must be large enough not to clip the intended 0.3A -> 0.7A -> A transitions.

Initial experiment criteria:

```text
push_steps: fixed at 8
profile: [0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3]
A sweep: test whether final displacement changes monotonically enough to cover short/mid/long pushes
quality checks: no backward push action, no EEF backward steps during push, no unusually large EEF step
```

If the sweep gives stable and reasonably monotonic distances, this profile should replace the current sinusoidal impulse profile for the next dataset version.

### Preliminary sweep result for fixed 8-step trapezoid profile

A temporary simulation sweep was run without modifying the production controller. The experiment monkey-patched only the impulse envelope and fixed:

```text
profile = [0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3]
push_steps = 8
push_scale = 10.0
straight push
```

The sweep varied only `A`.

```text
mu=0.10
A=0.25 -> 22.7 cm
A=0.30 -> 31.8 cm
A=0.35 -> 39.8 cm
A=0.40 -> 60.5 cm
A=0.45 -> 64.1 cm
A=0.50 -> 66.7 cm
A=0.55 -> 100.9 cm, EEF backward steps appear
A=0.60 -> 98.5 cm, EEF backward steps appear
A=0.65 -> 95.7 cm, EEF backward steps appear
```

```text
mu=0.20
A=0.25 -> 13.4 cm
A=0.30 -> 20.1 cm
A=0.35 -> 18.6 cm
A=0.40 -> 33.6 cm
A=0.45 -> 37.0 cm
A=0.50 -> 37.0 cm
A=0.55 -> 46.4 cm, EEF backward steps appear
A=0.60 -> 47.8 cm, EEF backward steps appear
A=0.65 -> 48.0 cm, EEF backward steps appear
```

The action profile itself is clean: no backward push action was produced in the sweep. For the usable range, EEF backward steps were also zero. The problematic cases begin around `A >= 0.55`, where EEF backward steps appear and should be excluded or re-tuned.

Conclusion:

```text
The fixed 8-step trapezoid impulse profile is more learnable than the sine profile and can produce multiple useful displacement regimes.
However, A is not a globally linear distance controller.
The next dataset should still use a calibration table mapping (mu, target distance) -> A.
```

Recommended safe starting bands from this preliminary sweep:

```text
mu=0.10: A around 0.25-0.35 covers roughly 23-40 cm.
mu=0.20: A around 0.25-0.50 covers roughly 13-37 cm.
Avoid A >= 0.55 unless separately validated.
```

This supports replacing the sine profile with the fixed trapezoid profile, but not replacing calibration with a naive global linear formula.

### Follow-up: fixed 13-step lower-peak trapezoid profile

After visual inspection of the 8-step profile videos, the main failure mode at high `A` is that the EEF hits the object too hard and can rebound after contact. A natural alternative is to lengthen the push window and lower the peak action amplitude.

A temporary 13-step profile was tested:

```text
push_steps = 13
profile = [0.2, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.4, 0.2, 0.0]
action_x[t] = A * profile[t]
```

This lowers the peak action while preserving a longer controlled push window.

Preliminary sweep with `push_scale = 10.0`:

```text
mu=0.10
A=0.18 -> 17.1 cm, back_eef=0
A=0.20 -> 20.4 cm, back_eef=0
A=0.22 -> 17.2 cm, back_eef=0
A=0.24 -> 20.0 cm, back_eef=0
A=0.26 -> 24.9 cm, back_eef=1
A=0.28 -> 26.1 cm, back_eef=0
A=0.30 -> 31.8 cm, back_eef=0
A=0.32 -> 34.4 cm, back_eef=0
A=0.34 -> 33.2 cm, back_eef=0
A=0.36 -> 42.5 cm, back_eef=1
```

```text
mu=0.20
A=0.18 -> 12.9 cm, back_eef=0
A=0.20 -> 14.9 cm, back_eef=0
A=0.22 -> 14.4 cm, back_eef=0
A=0.24 -> 16.6 cm, back_eef=0
A=0.26 -> 15.7 cm, back_eef=1
A=0.28 -> 16.4 cm, back_eef=0
A=0.30 -> 20.3 cm, back_eef=0
A=0.32 -> 20.6 cm, back_eef=0
A=0.34 -> 21.7 cm, back_eef=0
A=0.36 -> 24.1 cm, back_eef=1
```

Conclusion:

```text
The 13-step lower-peak profile does reduce peak EEF speed and should reduce hard collision/rebound visually.
However, lower peak amplitude is not dynamically equivalent to the shorter 8-step impulse.
For mu=0.20, this first 13-step profile cannot yet produce 35-40 cm pushes at safe A values.
For mu=0.10, it can approach 34 cm safely and 42 cm with a backward EEF warning.
```

This suggests the correct direction is a medium-length controlled velocity pulse, but the profile still needs tuning. Good next candidates are:

```text
1. Keep push_steps=13 but use a slightly stronger plateau profile, such as [0.3, 0.6, 0.85, 1, ..., 1, 0.85, 0.6, 0.3].
2. Try push_steps=10 or 11 as a compromise between the current 8-step impulse and the weak 13-step push.
3. Keep 13 steps but allow A above 0.36 only if the EEF backward/rebound check remains clean after visual review.
```

The main design point remains valid: the model should see a fixed, simple action shape with one interpretable amplitude parameter, but that profile must still transfer enough momentum to the object without causing rebound.

### High-A check for 13-step profile at mu=0.20

The previous 13-step sweep only tested low peak amplitudes. A follow-up sweep tested whether a 13-step push can still reach about 50 cm at `mu=0.20` by increasing `A`.

The same 13-step profile was used:

```text
profile = [0.2, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.4, 0.2, 0.0]
push_scale = 10.0
mu = 0.20
```

Result:

```text
A=0.38 -> 27.6 cm, back_eef=1
A=0.40 -> 30.1 cm, back_eef=1
A=0.42 -> 33.4 cm, back_eef=1
A=0.44 -> 32.5 cm, back_eef=1
A=0.46 -> 39.0 cm, back_eef=1
A=0.48 -> 37.0 cm, back_eef=2
A=0.50 -> 41.7 cm, back_eef=1
A=0.52 -> 42.6 cm, back_eef=1
A=0.54 -> 31.7 cm, back_eef=1
A=0.56 -> 34.5 cm, back_eef=0
A=0.58 -> 43.1 cm, back_eef=1
A=0.60 -> 43.1 cm, back_eef=0
A=0.65 -> 44.9 cm, back_eef=1
A=0.70 -> 50.8 cm, back_eef=1
A=0.75 -> 47.1 cm, back_eef=1
```

Conclusion:

```text
For mu=0.20 with this 13-step profile, about 50 cm requires A around 0.70.
That is lower than an extreme hard impulse, but it is not clean: EEF backward appears.
The clean no-backward cases in this sweep only reached about 34-43 cm.
```

This suggests that simply extending to 13 steps is not enough by itself. The collision/rebound issue is not only peak amplitude; it also depends on how long the gripper remains in contact while the object starts moving away. A 13-step pulse may keep driving into a moving object long enough to create contact/controller mismatch.

A better candidate is likely a medium-duration profile:

```text
push_steps = 10 or 11
moderate ramp
short plateau
A around 0.50-0.65 for high-friction long pushes
```

This should preserve enough impulse for high-friction 45-50 cm cases while reducing the hard-hit behavior of the original short pulse.

### 10-step medium-duration profile at mu=0.20

A 10-step profile was tested as a compromise between the original short 8-step impulse and the longer 13-step pulse:

```text
push_steps = 10
profile = [0.25, 0.55, 0.85, 1.0, 1.0, 1.0, 1.0, 0.85, 0.55, 0.25]
push_scale = 10.0
mu = 0.20
```

Result:

```text
A=0.40 -> 31.0 cm, back_eef=0
A=0.45 -> 28.1 cm, back_eef=0
A=0.50 -> 37.7 cm, back_eef=0
A=0.55 -> 39.5 cm, back_eef=0
A=0.60 -> 33.6 cm, back_eef=0
A=0.65 -> 46.8 cm, back_eef=0
A=0.70 -> 52.1 cm, back_eef=0
A=0.75 -> 53.8 cm, back_eef=2
A=0.80 -> 66.4 cm, back_eef=2
```

This is better than the tested 13-step profile for high-friction long pushes. It reaches the 45-50 cm regime without EEF backward:

```text
A=0.65 -> 46.8 cm, clean
A=0.70 -> 52.1 cm, clean
```

Conclusion:

```text
The 10-step medium-duration trapezoid profile is currently the best candidate among the tested impulse profiles.
It is simpler and more learnable than sine.
It avoids the hard short collision of the original impulse better than 8 steps.
It avoids the prolonged-contact weakness/rebound risk of the first 13-step profile.
```

Recommended next dataset candidate:

```text
push_steps = 10
profile = [0.25, 0.55, 0.85, 1.0, 1.0, 1.0, 1.0, 0.85, 0.55, 0.25]
A calibrated by friction and target displacement
initial safe high-friction long range: A around 0.65-0.70 for mu=0.20
```

### Non-monotonic A-to-distance diagnosis

A serious issue observed in the 10-step sweep is that final push distance is not perfectly monotonic in `A`. This is not caused by the action label itself. For the fixed 10-step profile, the action sum is exactly linear in `A`:

```text
profile sum = 7.3
sum(action_x) = 7.3 * A
```

For `mu=0.20`, detailed diagnostics show:

```text
A=0.45 -> final 28.1 cm, push-end 24.6 cm, post-slide 3.5 cm
A=0.50 -> final 37.6 cm, push-end 30.8 cm, post-slide 6.9 cm
A=0.55 -> final 39.5 cm, push-end 32.2 cm, post-slide 7.4 cm
A=0.60 -> final 33.5 cm, push-end 28.8 cm, post-slide 4.7 cm
A=0.65 -> final 46.8 cm, push-end 36.8 cm, post-slide 10.1 cm
A=0.70 -> final 52.0 cm, push-end 39.8 cm, post-slide 12.2 cm
A=0.75 -> final 53.8 cm, push-end 40.9 cm, post-slide 12.9 cm
A=0.80 -> final 66.4 cm, push-end 47.3 cm, post-slide 19.1 cm
```

The non-monotonic case `A=0.60` already has a larger commanded action than `A=0.55`, but it produces less forward displacement. The diagnostics show that this is a contact-dynamics issue rather than an action-generation issue:

```text
A=0.55: push-end 32.2 cm, release speed 0.550 m/s, max lateral deviation 1.0 cm
A=0.60: push-end 28.8 cm, release speed 0.432 m/s, max lateral deviation 1.8 cm
```

So the larger command at `A=0.60` transferred less useful forward momentum into the object. More energy likely went into contact mismatch, lateral motion, box rotation, or short contact separation/re-impact. This is expected in discontinuous contact dynamics: larger controller commands do not guarantee larger useful forward object impulse.

A diagnostic plot was saved at:

```text
tmp/trapezoid_impulse_profile_2026-06-29/trapezoid10_mu02_nonmonotonic_diagnostic.png
```

Design consequence:

```text
A should not be treated as a globally monotonic distance label.
A is a control amplitude, not a reliable distance parameter.
The dataset generator must use empirical calibration and acceptance filtering.
```

Recommended mitigation:

```text
1. For each friction and distance bucket, sweep A and select empirically validated samples.
2. Reject samples with excessive lateral drift, EEF backward, or visually out-of-frame final states.
3. Do not interpolate target distance from A with a single formula.
4. Store both commanded A and measured outcome labels: final displacement, push-end displacement, release velocity, lateral drift.
5. For VLA training, consider using a discrete push-strength class or calibrated primitive ID instead of asking the model to regress A as a continuous distance control variable.
```

### Non-monotonic A-to-distance diagnosis

A serious issue observed in the 10-step sweep is that final push distance is not perfectly monotonic in `A`. This is not caused by the action label itself. For the fixed 10-step profile, the action sum is exactly linear in `A`:

```text
profile sum = 7.3
sum(action_x) = 7.3 * A
```

For `mu=0.20`, detailed diagnostics show:

```text
A=0.45 -> final 28.1 cm, push-end 24.6 cm, post-slide 3.5 cm
A=0.50 -> final 37.6 cm, push-end 30.8 cm, post-slide 6.9 cm
A=0.55 -> final 39.5 cm, push-end 32.2 cm, post-slide 7.4 cm
A=0.60 -> final 33.5 cm, push-end 28.8 cm, post-slide 4.7 cm
A=0.65 -> final 46.8 cm, push-end 36.8 cm, post-slide 10.1 cm
A=0.70 -> final 52.0 cm, push-end 39.8 cm, post-slide 12.2 cm
A=0.75 -> final 53.8 cm, push-end 40.9 cm, post-slide 12.9 cm
A=0.80 -> final 66.4 cm, push-end 47.3 cm, post-slide 19.1 cm
```

The non-monotonic case `A=0.60` has a larger commanded action than `A=0.55`, but it produces less forward displacement. The diagnostics show that this is a contact-dynamics issue rather than an action-generation issue:

```text
A=0.55: push-end 32.2 cm, release speed 0.550 m/s, max lateral deviation 1.0 cm
A=0.60: push-end 28.8 cm, release speed 0.432 m/s, max lateral deviation 1.8 cm
```

So the larger command at `A=0.60` transferred less useful forward momentum into the object. More energy likely went into contact mismatch, lateral motion, box rotation, or short contact separation/re-impact. This is expected in discontinuous contact dynamics: larger controller commands do not guarantee larger useful forward object impulse.

A diagnostic plot was saved at:

```text
tmp/trapezoid_impulse_profile_2026-06-29/trapezoid10_mu02_nonmonotonic_diagnostic.png
```

Design consequence:

```text
A should not be treated as a globally monotonic distance label.
A is a control amplitude, not a reliable distance parameter.
The dataset generator must use empirical calibration and acceptance filtering.
```

Recommended mitigation:

```text
1. For each friction and distance bucket, sweep A and select empirically validated samples.
2. Reject samples with excessive lateral drift, EEF backward, or visually out-of-frame final states.
3. Do not interpolate target distance from A with a single formula.
4. Store both commanded A and measured outcome labels: final displacement, push-end displacement, release velocity, lateral drift.
5. For VLA training, consider using a discrete push-strength class or calibrated primitive ID instead of asking the model to regress A as a continuous distance control variable.
```

### Fast-ramp profile with longer contact offset

A follow-up tested the hypothesis that the gripper should reach high speed before first effective contact. The previous 10-step ramp sometimes contacted the object at the 0.85A frame. A faster ramp was tested:

```text
push_steps = 10
profile = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
```

The contact offset was also moved farther behind the object:

```text
pusher_contact_offset_xy[0] in {-0.115, -0.125, -0.135, -0.145}
```

This worked better for contact timing. In almost all tested cases, first effective box movement happened at `profile=1.0`, meaning the gripper was already at peak command when the object started moving.

Useful `mu=0.20` examples:

```text
offset=-0.115, A=0.55 -> final 47.5 cm, push-end 37.6 cm, back_eef=0, lateral=0.4 cm
offset=-0.115, A=0.60 -> final 50.4 cm, push-end 39.5 cm, back_eef=0, lateral=0.3 cm
offset=-0.135, A=0.50 -> final 50.5 cm, push-end 37.6 cm, back_eef=0, lateral=1.0 cm
offset=-0.145, A=0.50 -> final 41.2 cm, push-end 32.3 cm, back_eef=0, lateral=0.3 cm
```

The best current candidate for high-friction long pushes is:

```text
profile = [0.5, 1, 1, 1, 1, 1, 1, 1, 0.5, 0]
push_steps = 10
mu = 0.20
contact_offset_x = -0.115
A = 0.55-0.60
```

This reaches roughly 47-50 cm with no EEF backward and low lateral drift.

Important caveat:

```text
Distance is still not globally monotonic in A and offset.
But fast ramp + slightly longer offset makes contact timing much more consistent.
```

Design implication:

```text
The generator should explicitly target contact timing, not only final displacement.
A good acceptance condition is first_effective_contact_local in the peak plateau, e.g. local frame 1-3 for this fast-ramp profile.
```

For the next dataset version, the profile should likely be changed from the earlier ramp to this faster profile, then calibrated jointly over:

```text
friction_mu
A
contact_offset_x
final displacement bucket
first contact timing
lateral drift
EEF backward count
```

### Monotonicity check for fast-ramp profile

A finer sweep tested whether the fast-ramp profile with a longer contact offset gives monotonic final distance as `A` increases.

Configuration:

```text
profile = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
push_steps = 10
contact_offset_x = -0.115
```

For `mu=0.20`:

```text
A=0.400 -> 32.4 cm, back_eef=1
A=0.425 -> 35.4 cm, back_eef=1
A=0.450 -> 34.2 cm, back_eef=1
A=0.475 -> 42.7 cm, back_eef=1
A=0.500 -> 38.4 cm, back_eef=1
A=0.525 -> 43.5 cm, back_eef=1
A=0.550 -> 47.5 cm, back_eef=0
A=0.575 -> 49.8 cm, back_eef=0
A=0.600 -> 50.4 cm, back_eef=0
A=0.625 -> 63.4 cm, back_eef=0
A=0.650 -> 61.0 cm, back_eef=0
A=0.675 -> 54.3 cm, back_eef=0
A=0.700 -> 70.4 cm, back_eef=0
```

For `mu=0.10`:

```text
A=0.250 -> 22.2 cm, back_eef=0
A=0.275 -> 32.9 cm, back_eef=0
A=0.300 -> 32.2 cm, back_eef=0
A=0.325 -> 36.5 cm, back_eef=0
A=0.350 -> 41.1 cm, back_eef=0
A=0.375 -> 45.5 cm, back_eef=1
A=0.400 -> 49.7 cm, back_eef=1
A=0.425 -> 58.9 cm, back_eef=1
A=0.450 -> 55.5 cm, back_eef=1
A=0.475 -> 74.9 cm, back_eef=1
A=0.500 -> 65.8 cm, back_eef=1
A=0.525 -> 94.0 cm, back_eef=1
A=0.550 -> 98.1 cm, back_eef=0
```

Result:

```text
Fast ramp + longer offset improves contact timing.
It does not make final distance globally monotonic in A.
```

There are useful local monotonic bands, for example:

```text
mu=0.20, A=0.550 -> 0.600 gives 47.5 -> 50.4 cm cleanly.
mu=0.10, A=0.300 -> 0.350 gives 32.2 -> 41.1 cm cleanly.
```

But violations remain, even when first contact happens at peak profile. This means final displacement is still governed by contact dynamics, not just command magnitude.

Design consequence:

```text
Do not train or generate data under the assumption that A is a continuous monotonic distance knob.
Use A as a discrete calibrated primitive parameter.
For each friction and distance bucket, empirically select accepted A/offset pairs.
```

For dataset generation, a safer format is:

```text
push_strength_class: short / mid / long
calibrated_A
calibrated_contact_offset_x
measured_displacement
```

rather than exposing `A` alone as the target variable for distance control.

### Dataset-generation decision: filter by measured outcome, not by A

The current conclusion is that exact monotonicity of `A -> final distance` is not required for the dataset. The trend is good enough for generating diverse push strengths, as long as the generator does not assume `A` directly determines distance.

The dataset should be generated as follows:

```text
1. Sweep candidate controls: A, contact_offset_x, friction_mu, push direction.
2. Run the simulation.
3. Measure the actual outcome.
4. Accept or reject by measured quality and measured distance.
5. Assign the accepted sample to a distance bucket based on actual displacement, not based on A.
```

`A` should therefore be treated as a candidate control parameter:

```text
A is used to propose a push primitive.
A is not used as the source of truth for the distance label.
```

The source of truth is the measured rollout result:

```text
measured final displacement
measured push-end displacement
measured lateral drift
measured first contact timing
measured EEF backward count
visual in-frame quality
```

Recommended hard cap:

```text
final displacement <= 60 cm
```

Samples above this should be rejected for the next dataset version because they are too likely to be visually out of range, too dynamic for stable imitation, or too sensitive to small contact changes.

Recommended distance buckets remain outcome-based, for example:

```text
short: measured displacement around 10-25 cm
mid:   measured displacement around 25-40 cm
long:  measured displacement around 40-60 cm
```

The exact bucket edges can be adjusted, but the assignment must come from measured displacement.

Recommended quality filters:

```text
final displacement <= 60 cm
EEF backward count during push == 0
lateral drift below threshold
first effective contact occurs inside the peak plateau
object remains visible / not near image boundary
```

This makes the dataset robust even if the mapping from `A` to final displacement is only approximately increasing and locally noisy.

### Current recommended primitive parameters from experiments

This section records the current best experimental parameters so the next dataset generation does not depend on implicit memory.

Recommended impulse primitive:

```text
push_mode = impulse
push_steps = 10
control_freq = 20 Hz
push duration = 0.5 s
profile = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
action_x[t] = A * profile[t]
profile_sum = 8.0
```

Recommended starting contact offset:

```text
pusher_contact_offset_xy = (-0.115, 0.0)
```

This offset is longer than the previous `-0.105` setting. The purpose is to let the gripper accelerate before effective contact. In the current best `mu=0.20` trials, this makes the first effective box movement occur during the peak plateau:

```text
first_effective_contact_local = 2
profile[first_effective_contact_local] = 1.0
```

That means the object starts moving after the command has already reached `A`.

Recommended `mu=0.20` high-friction long-push candidates:

```text
A=0.55, contact_offset_x=-0.115
final displacement: 47.5 cm
push-end displacement: 37.6 cm
post-release slide: 9.9 cm
release speed: 0.637 m/s
max box speed during push: 1.374 m/s
lateral drift: 0.38 cm
EEF backward count: 0
first effective contact local frame: 2
```

```text
A=0.60, contact_offset_x=-0.115
final displacement: 50.4 cm
push-end displacement: 39.5 cm
post-release slide: 10.9 cm
release speed: 0.660 m/s
max box speed during push: 1.413 m/s
lateral drift: 0.33 cm
EEF backward count: 0
first effective contact local frame: 2
```

The corresponding action curves are simple and should be easier for a VLA to learn than the sine profile:

```text
A=0.55 action_x:
[0.275, 0.550, 0.550, 0.550, 0.550, 0.550, 0.550, 0.550, 0.275, 0.000]
```

```text
A=0.60 action_x:
[0.300, 0.600, 0.600, 0.600, 0.600, 0.600, 0.600, 0.600, 0.300, 0.000]
```

Measured EEF x movement per push frame for these two candidates:

```text
A=0.55 eef_dx_cm_per_frame:
[0.857, 3.009, 4.972, 4.005, 0.080, 0.002, 0.128, 2.057, 4.262, 3.078]
```

```text
A=0.60 eef_dx_cm_per_frame:
[0.936, 3.131, 5.316, 3.431, 0.046, 0.023, 0.108, 1.772, 4.299, 3.221]
```

Measured box forward position during the push window:

```text
A=0.55 box_forward_cm:
[0.0, 0.0, 2.749, 9.359, 15.396, 20.890, 25.859, 30.297, 34.214, 37.618]
```

```text
A=0.60 box_forward_cm:
[0.0, 0.0, 3.394, 10.181, 16.408, 22.086, 27.213, 31.828, 35.918, 39.488]
```

Measured box speed during the push window:

```text
A=0.55 box_speed_cm_s:
[0.0, 0.0, 137.391, 125.187, 116.296, 104.857, 93.762, 84.365, 73.581, 63.722]
```

```text
A=0.60 box_speed_cm_s:
[0.0, 0.0, 141.255, 131.175, 117.812, 109.662, 98.605, 88.329, 76.165, 65.951]
```

Recommended acceptance rules for the next data generator:

```text
final displacement <= 60 cm
assign distance bucket by measured displacement, not by A
first effective contact local frame inside peak plateau
EEF backward count == 0
lateral drift below threshold
object remains visually in frame
```

A practical initial calibration table entry can start with:

```text
mu=0.20 long bucket: A in [0.55, 0.60], contact_offset_x=-0.115
```

The same profile should be swept for the other friction values. The accepted samples should be chosen by measured rollout outcome rather than by assuming `A` is a monotonic distance controller.
