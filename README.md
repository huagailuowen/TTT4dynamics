# TTT4dynamics

**Test-Time-Learnable Execution Memory for Dynamic Vision-Language-Action Control**

## Introduction

TTT4dynamics studies a concrete weakness of current streaming VLA policies in dynamic manipulation: they can react quickly to new observations, but they often do not maintain a stable sense of what they have already executed, how the local environment is changing, and how the next action chunk should continue the ongoing motion.

DynamicVLA is an important baseline for this problem. It reduces inference latency, overlaps action execution with model inference, and streams new action chunks into the robot control loop. This makes the robot more responsive to moving objects and changing observations. However, faster reaction alone does not solve temporal consistency. If each chunk is generated mainly from the latest observation, the robot can repeatedly re-align to a target, overwrite its previous intent, stall near an intermediate phase, or fail to adapt when the local motion pattern changes during execution.

Our current view is that the missing component is not a large explicit world model, a hand-designed object-centric state channel, or online finetuning of the whole VLA. The missing component is a lightweight execution memory that can be updated while the robot acts.

TTT4dynamics therefore uses three core ideas:

1. **Fast-WAM backbone**

   The first implementation uses Fast-WAM as the backbone instead of treating WAM/VLA choices as open. The backbone handles visual-language-action alignment, nominal action chunk generation, Fast-WAM-style video prediction, and low-latency control.

2. **Fast-WAM video prediction as world prediction**

   We do not introduce a new explicit `world-prediction token segment` in the first version. World prediction is represented through Fast-WAM-style video prediction and future-video latents. In other words, the model's expectation about future scene motion comes from video/future-latent modeling inside the backbone, not from a manually specified object-centric prediction interface.

3. **TTT-QKV SHE Adapter as execution memory**

   We add a small test-time-learnable memory component called the SHE Adapter, where `SHE` means `State / History / Execution`. In the first implementation this is not a new input interface. We keep Fast-WAM's original input `x` unchanged and insert a TTT layer inside the Transformer path as a gated residual module. The memory is updated through the TTT layer's fast weights, then read by the same hidden-state stream before the action head.

The central hypothesis is:

> A streaming VLA becomes more reliable in dynamic manipulation when Fast-WAM-style video/future latents provide an implicit world-prediction representation, while a small TTT-QKV execution memory is updated from recent transitions and conditions the next action chunk.

This design targets failure modes such as:

- repeated re-alignment caused by chunk overwriting;
- delayed inference changing when the next chunk should actually begin;
- objects sliding or drifting during approach or grasp;
- motion patterns that are exploitable within one environment but not known beforehand;
- mid-execution velocity or motion-pattern changes that require fast adaptation.

The project is intentionally scoped. TTT4dynamics is not trying to solve full symbolic replanning, full physical reasoning, tactile-heavy manipulation, semantic long-term memory, or full-backbone online adaptation. The first target is narrower: make a streaming dynamic manipulation policy more temporally coherent and more adaptive within an episode or short adaptation stream.

## Method

TTT4dynamics follows a lightweight training-centered architecture.

```text
Fast-WAM original inputs
        -> Fast-WAM backbone
        -> video backbone hidden state x_video
             -> gated TTT-QKV layer
             -> video/future latents as world-prediction representation
        -> action backbone hidden state x_action
             -> gated TTT-QKV layer
             -> action head
        -> next action chunk
```

### Backbone

The base model is temporarily fixed to Fast-WAM. We use its video prediction / future-latent path as the current world-prediction representation. The first TTT version inserts the layer into both sides of the model:

- the video backbone, where the TTT layer operates on the existing video hidden state `x_video`;
- the action backbone, where the TTT layer operates on the existing action hidden state `x_action`.

At inference time, the policy does not predict future video frames for control. It keeps the observation-side clean latent and uses the action pathway for low-latency control, while both backbones can still carry their own TTT fast-weight state.

### First Insertion Rule

The first version should not redesign the Fast-WAM input features.

`x_video` and `x_action` mean the existing hidden states produced by Fast-WAM at the insertion points. We do not initially concatenate hand-written fields such as previous chunks, elapsed time, executed prefixes, or object-motion summaries into either hidden state. Those signals can be introduced later only if ablations show the inserted TTT layers are insufficient.

The immediate implementation target is:

- keep Fast-WAM tokenizer, observation inputs, language inputs, robot-state inputs, and action-head interface unchanged;
- insert a gated TTT layer into the video backbone;
- insert a gated TTT layer into the action backbone;
- initialize the residual gate close to zero so the starting model behaves like the original Fast-WAM;
- update only the small TTT fast weights at test time;
- train or finetune the TTT projections and gate conservatively while keeping the backbone frozen or mostly frozen.

### TTT-QKV Layer

The memory mechanism follows the TTT Layers / One-Minute Video Generation style:

```text
train_view = theta_K x_branch
label_view = theta_V x_branch
L_inner = || f(train_view; W) - label_view ||^2
test_view = theta_Q x_branch
output = f(test_view; W_updated)
```

In TTT4dynamics, `x_branch` is not a newly engineered input vector. It is the existing Fast-WAM hidden state at the TTT insertion point, with one branch-specific instance for `x_video` and one for `x_action`. `W` is the fast memory learner inside that branch's SHE Adapter. In the first version, the outer loop mainly learns the branch-specific Q/K/V projections that define the internal self-supervised TTT task: `theta_K`, `theta_V`, and `theta_Q`. Residual gates and initialization can be trained conservatively if needed, but the Fast-WAM backbone remains frozen or mostly frozen. During inner-loop updates, only the small TTT learner / fast weights are updated.

The TTT layer should not be treated as a normal external MLP attached after the action head. It is inserted into the Transformer backbone as a gated residual sub-layer, following the TTT Layers style.

First implementation:

- insert one gated TTT-QKV branch into the video backbone;
- insert one gated TTT-QKV branch into the action backbone;
- keep video/action TTT fast weights separate at first, so each branch learns memory in its own representation space;
- initialize residual gates close to zero;
- train primarily the Q/K/V projections that define the inner self-supervised task; keep gates / initialization conservative if they are trained at all;
- keep the backbone frozen or mostly frozen for stability.

Cross-branch extension:

- insert a `video expert` TTT branch to write observation-transition, video-latent, future-latent, and motion evidence;
- insert an `action expert` TTT branch so action tokens and execution tokens read the updated SHE memory before generating the next action chunk.

### Training

The model should be trained in the same style in which it will be used: update while acting.

Stage 0: start from a Fast-WAM backbone trained with behavior cloning and Fast-WAM-style video prediction.

Stage 1: insert TTT-QKV SHE layers into both the video backbone and action backbone with conservative gated residual initialization.

Stage 2: run simulated inner loops along dynamic trajectories. At each step, the model predicts an action chunk, observes the next real transition, updates the TTT learner with a self-supervised transition loss, and uses the updated memory for the next action. Because inference does not predict future video frames, the inner-loop self-supervised loss should not include video future-frame prediction. The inner loop should use real observations, robot state, executed actions, and the action-expert side of the model.

Stage 3: optimize a minimal first-version outer objective:

```text
L = L_action
  + lambda_video * L_video_prediction
```

`L_action` trains post-update action generation. `L_video_prediction` preserves Fast-WAM's world-prediction representation as a training-time signal. The parameters optimized by this outer loop are primarily the Q/K/V projections and the small components defining the internal TTT self-supervised task, not the whole Fast-WAM backbone. Motion consistency, recovery preference, and rollout rewards are later extensions, not first-version losses.

### Trajectory Organization

Training trajectories do not have to be single independent episodes. They should follow the dynamic data types we construct.

- **Single episode**: tests local execution continuity and repair-like behavior.
- **Repeated short episodes under the same motion regime**: allows memory carry-over so the model can exploit reusable environment dynamics.
- **Mid-stream velocity or pattern change**: memory should not be immediately reset; the TTT learner must overwrite stale evidence.
- **Unrelated scene or task switch**: fast weights should be reset, or a boundary/reset token should tell the model not to carry memory.

This reset/carry rule is part of the training design. It should not be hard-coded as “reset after every episode.”

## Contributions

- Uses Fast-WAM-style video prediction / future-video latents as the initial world-prediction representation for dynamic VLA control.
- Introduces a small Transformer-inserted TTT-QKV SHE Adapter as test-time-learnable execution memory.
- Trains the policy with update-while-acting inner loops and post-update outer losses, instead of treating behavior cloning as isolated action-chunk prediction.
- Defines dynamic trajectory streams that force the model to preserve intent, adapt to exploitable motion patterns, and overwrite stale memory when dynamics change.

## Experiments

TODO.
