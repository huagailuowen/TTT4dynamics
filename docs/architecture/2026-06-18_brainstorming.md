# Low-Dimensional Adaptive World Model for Robotic Test-Time Adaptation

## 0. 核心目标

我们想做一个 **robotic adaptation** 课题：机器人在推理时面对的环境和训练时不同，例如：

* 桌面更滑 / 更粗糙；
* 绳子更硬 / 更软；
* 物体质量分布不同；
* 夹爪有滞后 / 锈蚀 / 输出幅度变小；
* 插头和插座几何有偏差；
* 接触模式变化，例如滑动、卡住、jamming、碰撞风险等。

我们的目标不是做一个更大的 VLA，也不是直接用 heavy video generation 作为世界模型，而是希望构建一个：

> **低维、高效、可被 test-time latent 快速调节的 robotic world model。**

它能够从高维 observation 中抽取当前任务真正 relevant 的 working state，在这个低维状态空间中做 prediction、policy learning 和 adaptation，并通过一个隐式 context latent `c` 来适应当前测试环境。

---

## 1. 背后的核心直觉

### 1.1 Language / game AI 成功的重要原因

Language AI 和 game AI 之所以取得巨大成功，很大程度上是因为它们本身运行在高效、低维、结构化的空间中：

* LLM 的输入是 token，而不是 raw acoustic wave 或 raw pixels；
* 围棋 / chess AI 的输入是 board state，而不是摄像头图像；
* 决策空间和状态空间已经被人类或规则系统高度压缩。

相比之下，embodied AI 最大的困难是：

> observation 极其高维，但真正决定动作的 working state 往往非常低维，并且需要极高精度。

例如插插座任务中，真正重要的可能只是：

* 插头末端几何；
* 插座孔洞几何；
* 二者的相对位置和相对方向；
* contact / jamming / clearance 信息；
* 局部碰撞风险。

背景纹理、墙面颜色、插座品牌、全局图像外观大多不是核心变量。

---

### 1.2 Embodied AI 的关键问题：自动发现 task-relevant working state

我们认为 embodied AI 的低效不只是数据少或模型小，而是缺少一种机制：

> 从高维 observation 中自动发现当前任务真正 matter 的低维 working state。

这个 working state 不是固定的。它会随着任务阶段变化。

例如插插座：

1. **粗对齐阶段**

   * 重要变量：plug pose、socket pose、relative alignment。
2. **接触插入阶段**

   * 重要变量：tip clearance、insertion axis、contact force、jamming cue。
3. **即将碰撞桌面阶段**

   * 重要变量：hand-table clearance、collision margin、safe motion direction。

所以我们不应该假设一个全局固定低维状态，而应该假设：

> 每个决策时刻都有一个局部低维、任务相关、动态变化的 working state。

---

## 2. 总体研究定位

我们要做的不是：

* pixel-level video world model；
* object-centric particle world model；
* 纯 memory-based VLA；
* 纯 online RL；
* 直接 test-time fine-tuning 整个 policy；
* 让 video generation 主导机器人推理。

我们真正想做的是：

> **Adaptive working-state world model with test-time physical context adaptation.**

也就是：

1. 从 observation 和 task 中提取当前 relevant 的 working state `z_t`；
2. 通过少量 trial 学出当前环境的隐式 context latent `c`；
3. 用 `c` 同时 steer：

   * working-state extraction；
   * state world model prediction；
   * action policy；
   * auxiliary video generation head；
4. 在 `z_t` 和 `c` 的空间中进行高效 prediction、policy learning 和 test-time adaptation。

---

## 3. 不应过度 object-centric

我们不希望方法过度依赖 object-centric prior，例如：

* 显式 object segmentation；
* particles；
* keypoints；
* object graph；
* contact graph；
* 手工定义功能区。

这些方法在 RoboCook / RoboCraft / deformable manipulation 中很有效，但它们加入了太多人工先验，容易把问题变成 representation engineering。

我们的立场是：

> Object-centric / keypoint / particle representation 可以作为 fallback 或辅助 prior，但不应该成为核心假设。

更好的定位是：

> **weakly-structured / interaction-centric / task-relevant working-state representation。**

也就是说，模型应该尽可能自动从 V-L model、human prior、task instruction、observation、history 和 prediction error 中决定当前哪些变量重要。只有在模型做不到时，才用 human prior 或 VLM 标注辅助。

---

## 4. 核心变量定义

### 4.1 `z_t`：当前 working state

`z_t` 表示当前 observation 在任务相关 working-state space 中的一个点。

形式上：

```text
z_t = R_φ(o_t, g, c)
```

其中：

* `o_t`：当前 observation，例如 RGB-D、proprioception、force/tactile；
* `g`：task goal / language instruction / desired state；
* `c`：当前环境的 physical context latent；
* `R_φ`：working-state extractor / relevance router。

`z_t` 不是固定 object state，而是 task-conditioned、phase-dependent 的低维状态。

---

### 4.2 `c`：隐式 physical context latent

`c` 不是 object embedding，也不是视觉 representation，而是：

> 当前测试环境隐藏动力学的隐式 context / belief。

它表示当前环境和训练平均环境之间的差异，例如：

* friction；
* stiffness；
* mass distribution；
* damping；
* compliance；
* actuator gain；
* delay；
* gripper bias；
* contact mode；
* scale / embodiment change。

`c` 的作用是把初始模型分布适应到当前环境分布。

更直观地说：

> `c` 是模型的环境旋钮。
> 它让 frozen model 在不直接改参数的情况下，临时适应当前世界。

---

### 4.3 `g`：目标或任务条件

`g` 表示当前任务目标，可以来自：

* language instruction；
* goal image；
* desired state；
* reward specification；
* human/VLM provided task prior。

policy 根据 `z_t, g, c` 选择动作。

---

### 4.4 `a_t`：当前动作

policy 输出当前动作：

```text
a_t = π_θ(z_t, g, c)
```

这里 policy 不应该直接看 predicted future `ẑ_future`，因为 action 是根据当前 state、goal 和 context 做决策。

---

## 5. 模型模块设计

### 5.1 Working-State Extractor / Relevance Router

```text
z_t = R_φ(o_t, g, c)
```

它负责从高维 observation 中抽取当前任务真正重要的 working state。

可以由以下信息共同决定：

* 当前图像 / depth；
* proprioception；
* tactile / force；
* language goal；
* history；
* prediction error；
* inferred context latent `c`；
* V-L model prior；
* optional human prior。

关键点：

> `c` 也应该 steer `z_t` 的提取，因为不同环境下模型应该关注不同变量。

例如：

* 桌面更滑时，物体位移残差更重要；
* 绳子更硬时，形变和受力历史更重要；
* 插入卡住时，contact/jamming cue 更重要；
* 快碰撞时，collision margin 更重要。

---

### 5.2 Context-Conditioned Policy

policy 的定义应该是：

```text
a_t = π_θ(z_t, g, c)
```

它根据当前 working state、目标和环境 context 输出动作。

重要原则：

> policy 不应该直接读取 `ẑ_future`。

原因是 `ẑ_future` 本身由候选 action 决定：

```text
ẑ_future = F_θ(z_t, a_t, c)
```

如果 policy 输入 `ẑ_future`，逻辑上会形成循环：

```text
要选 action → 需要 future
要有 future → 需要先知道 action
```

因此，policy 是 state feedback controller：

```text
current working state + goal + context → action
```

---

### 5.3 Context-Conditioned State World Model

state world model 是核心推理模块：

```text
ẑ_{t+1:t+H} = F_θ(z_t, a_{t:t+H-1}, c)
```

它根据当前 working state、candidate action 和 context latent 预测未来 working states。

这里 `c` 的作用是：

> 同样的 state 和 action，在不同环境中会导致不同后果。

例如：

* 低摩擦 `c`：push 后物体滑得更远；
* 高刚度 `c`：拉绳子后形变更小；
* actuator delay `c`：同样 command 实际运动更慢；
* plug offset `c`：插入时接触反馈和修正方向不同。

---

### 5.4 State World Model as Policy Teacher

虽然 policy 不直接看 `ẑ_future`，但 state world model 可以作为 planner / teacher。

例如：

```text
a* = argmax_a Score(F_θ(z_t, a, c), g)
```

world model 在 latent state 空间中评估不同 candidate actions，选出较好的 `a*`，再用它训练 policy：

```text
L_distill = || π_θ(z_t, g, c) - a* ||²
```

也就是说：

```text
world model / planner → 产生 teacher action
policy → 学习成为快速反射式 controller
```

这和 RoboCook 中用 world model 生成 state-goal policy teacher 的思想类似，但我们不希望依赖大量 random action，因为一般 manipulation 任务中 random action 大量无效。更合理的是：

* policy-prior guided exploration；
* short-horizon MPC；
* CEM in latent world model；
* uncertainty / information-gain exploration；
* contact-seeking primitive；
* goal-conditioned action proposal。

---

### 5.5 State-Guided Video Generation Head

我们不是完全不要 video generation，而是不要让 video generation 成为主世界模型。

video head 应该是：

```text
î_{t+1:t+H}
=
V_ψ(o_t, z_t, ẑ_{t+1:t+H}, a_{t:t+H-1}, c)
```

也就是：

```text
current image + current state + predicted future states + action + context
    → future image / future visual feature
```

video head 的作用是：

1. 辅助监督 state world model；
2. 提供视觉一致性检查；
3. 生成可解释 rollout；
4. 提供 uncertainty / failure signal；
5. 作为论文展示的可视化工具。

但它不是控制主线。

我们的原则是：

> `c, action` 指导 state world model；
> `c, action, state world model future` 再指导 video head。

也就是说：

```text
state world model 是主因果推理通道；
video head 是 state-conditioned visual decoder / auxiliary supervision。
```

---

## 6. 避免 video head 反客为主

必须防止 shortcut：

```text
current image + action → video head 直接生成未来
state world model 被架空
```

因此要限制 video head：

### 6.1 State bottleneck video decoding

video head 的 motion / geometry / contact 信息主要来自 `ẑ_future`。

当前图像只提供 appearance / texture，不提供 dynamics。

```text
appearance comes from image
dynamics comes from state world model
```

---

### 6.2 Video loss 作为辅助项

整体 loss 中 video loss 权重不能太大：

```text
L = L_state + L_policy + λ_video L_video + ...
```

`λ_video` 应该小于 state prediction 和 policy loss。更重要的是，video loss 不应该无约束地反向传播到 `z` 之前的 representation extractor，否则 `z` 很容易被 pixel reconstruction 拉向背景纹理，而不是保持为 task-relevant working state。

因此更推荐：

```text
L_video = Loss(VideoHead(stopgrad(z_t), stopgrad(ẑ_future), o_t, a, c), target_video)
```

或者使用 partial stop-gradient：

```text
z_video = stopgrad(z_state_part) + z_visual_adapter_part
```

也就是说，video head 可以用 `z` / predicted state future 作为条件，但 `L_video` 对 backbone、working-state extractor、state world model 的梯度应该被截断或强烈限制。video branch 应该辅助监督和可视化，而不能反过来主导 `z` 的语义。

---

### 6.3 可以用 state 接不同 auxiliary head，而不是只预测完整 RGB

state-guided visual branch 不一定要生成完整视频。更推荐的形式是：用 `z_t`、`ẑ_future`、action 和 `c` 接不同 auxiliary head，输出更接近低层物理响应的信息：

* future visual feature；
* motion field；
* changed-region reconstruction；
* contact-region crop；
* local future patch；
* video latent；
* optical flow / point motion；
* slip / contact / jamming cue。

这样可以减少对背景纹理的浪费，也更贴近我们真正关心的 action-conditioned physical response。这里的重点不是证明模型能生成漂亮视频，而是证明 state / context latent 确实携带了足够的低层运动和接触信息。

---

### 6.4 State-video consistency

video head 输出的未来图像再经过 state extractor，应该和 state world model 预测的 future state 一致：

```text
R(E(î_{t+k}), g, c) ≈ ẑ_{t+k}
```

这个 consistency loss 可以防止 video head 生成视觉上合理但 state 上错误的视频。

---

## 7. `c` 的更新逻辑

### 7.1 `c` 的本质

`c` 是 inner-loop adaptation variable。

它不是静态标签，而是在训练和测试中都可以通过少量 trial 被更新，用来把模型适配到当前环境。

---

### 7.2 World model 侧的 `c` 更新

world model 通过 prediction error 更新 `c`：

```text
L_world(c) =
|| z_{t+1} - F_θ(z_t, a_t, c) ||²
```

更新：

```text
c ← c - α ∇_c L_world(c)
```

含义：

> 通过观察 action 的真实后果，调整 `c`，使得模型在当前环境下预测更准。

---

### 7.3 Policy 侧的 `c` 更新

policy 也应该能通过 `c` 适应当前环境。

训练时，如果有 demo，可以用 BC loss：

```text
L_BC(c) =
|| a_t^demo - π_θ(z_t, g, c) ||²
```

测试时，如果没有 demo，可以用：

* RL reward；
* task success；
* goal progress；
* world-model planner pseudo-label；
* safety / contact / collision proxy。

例如：

```text
L_policy(c) = -R(π_θ(z_t, g, c))
```

或者：

```text
L_policy(c) =
|| π_θ(z_t, g, c) - a*_planner ||²
```

---

### 7.4 联合 inner-loop objective

`c` 的更新目标可以写成：

```text
L_inner(c)
=
λ_W L_world(c)
+
λ_BC L_BC(c)
+
λ_RL L_RL(c)
+
λ_reg ||c - c0||²
```

其中：

* `L_world`：让 prediction 更准；
* `L_BC`：让 policy 更像当前环境 expert；
* `L_RL`：让 policy 在当前环境 reward 更高；
* `L_reg`：防止少量 noisy trial 把 `c` 拉飞。

直觉：

> prediction error 告诉 `c` 当前世界是什么样；
> policy loss 告诉 `c` 当前世界中什么行为有效。

---

## 8. 训练时 outer loop 与推理时 slow learning 的区分

这是整个方法中必须分清的核心概念。

---

### 8.1 训练时 inner loop

训练时 inner loop 只更新 `c`，模拟测试时快速适应。

对于每个训练环境 `e_i`：

```text
support trials S_i
    → update c_i
```

例如：

```text
c_i' = c0 - α ∇_c L_support(θ, c0; S_i)
```

---

### 8.2 训练时 outer loop：目标 1，General Adaptability

训练时 outer loop 更新 `θ_meta`。

它的目标不是让模型直接拟合某个环境，而是：

> 学一个初始权重和条件化架构，使得 inner loop 的少量 `c` 更新可以适应各种环境。

形式上：

```text
min_θ E_{e_i} [
    L_query(θ, U_c(c0, S_i; θ), Q_i)
]
```

其中：

* `S_i`：support trials，用来更新 `c`；
* `Q_i`：query trials，用来评估更新后的 `c` 是否能泛化到同一环境的新轨迹；
* `θ`：模型参数；
* `U_c`：inner-loop c update。

训练 outer loop 学到的是：

> 模型应该长成什么样，才能容易被 `c` 快速调节。

---

### 8.3 测试时 fast loop

进入新测试环境后，先冻结 `θ_meta`，只更新 `c`：

```text
c* = U_c(c0, D_test; θ_meta)
```

此时模型临时变成：

```text
F_{θ_meta}(z, a, c*)
π_{θ_meta}(z, g, c*)
R_{φ_meta}(o, g, c*)
```

这个阶段是 fast adaptation：

> 用少量 trial 得到当前环境的临时 context，使模型快速适应当前测试分布。

---

### 8.4 测试时 slow learning：目标 2 和 3

测试时 slow learning 和训练 outer loop 不是一回事。

训练 outer loop 的目标是：

> 目标 1：General Adaptability。

测试 slow learning 的目标是：

> 目标 2：Environment Specialization。
> 目标 3：Residual Adaptability / c-controllability Preservation。

---

## 9. 测试 slow learning 的目标 2：当前环境特化

fast loop 得到了当前环境的临时最优 `c*`。

slow learning 要做的是把 `c*` 下的临时适应写回模型参数，使默认 `c0` 下也能适应当前环境。

即希望：

```text
F_{θ_env}(z, a, c0) ≈ F_{θ_meta}(z, a, c*)
π_{θ_env}(z, g, c0) ≈ π_{θ_meta}(z, g, c*)
```

直觉：

> 一开始机器人需要通过 `c*` 才知道桌面很滑；
> slow learning 后，它在默认状态下就已经习惯这个桌面很滑。

这相当于：

> 把 fast adaptation 变成 habit。

---

## 10. 测试 slow learning 的目标 3：保留 `c` 的控制能力

如果 slow learning 只追求当前环境特化，它可能破坏 `c` 的作用：

```text
θ_env 完全吸收当前环境
c 失去控制能力
模型只能适应一次
```

所以 slow learning 还必须保留：

> 即使模型形成当前环境的 habit，`c` 仍然能够继续调节 world model 和 policy。

也就是不仅要满足：

```text
θ_env, c0 ≈ θ_meta, c*
```

还要满足局部控制能力：

```text
F_{θ_env}(z, a, c0 + δ)
≈
F_{θ_meta}(z, a, c* + δ)

π_{θ_env}(z, g, c0 + δ)
≈
π_{θ_meta}(z, g, c* + δ)
```

直觉：

> slow update 应该把 `c*` 吸收到默认行为里，但不能压扁 `c` 空间。
> 它应该 re-center context manifold，而不是 collapse context manifold。

---

## 11. Slow learning 的建议 loss

测试 slow learning 可以写成：

```text
θ_env = θ_meta + Δθ_env
```

其中 `Δθ_env` 最好是 LoRA / adapter / small residual module，而不是 full fine-tuning。

目标函数：

```text
L_slow =
    L_specialize
  + λ_distill L_fast_distill
  + γ_control L_c_control
  + ρ_anchor L_anchor
  + η_replay L_replay
```

---

### 11.1 `L_specialize`

让默认 `c0` 下适应当前环境：

```text
L_specialize_world =
|| z_{t+1} - F_{θ_env}(z_t, a_t, c0) ||²
```

policy 部分：

```text
L_specialize_policy =
BC / RL / planner distillation under c0
```

---

### 11.2 `L_fast_distill`

把 fast-adapted teacher 蒸馏到 default context：

```text
teacher = (θ_meta, c*)
student = (θ_env, c0)
```

world model distillation：

```text
|| F_{θ_env}(z, a, c0)
 - stopgrad(F_{θ_meta}(z, a, c*)) ||²
```

policy distillation：

```text
|| π_{θ_env}(z, g, c0)
 - stopgrad(π_{θ_meta}(z, g, c*)) ||²
```

---

### 11.3 `L_c_control`

保留 `c` 的局部控制能力。

local neighborhood distillation：

```text
sample δ ~ small noise

F_{θ_env}(z, a, c0 + δ)
≈
F_{θ_meta}(z, a, c* + δ)

π_{θ_env}(z, g, c0 + δ)
≈
π_{θ_meta}(z, g, c* + δ)
```

或者 Jacobian matching：

```text
∂F_{θ_env}/∂c |_{c0}
≈
∂F_{θ_meta}/∂c |_{c*}

∂π_{θ_env}/∂c |_{c0}
≈
∂π_{θ_meta}/∂c |_{c*}
```

---

### 11.4 `L_anchor`

限制参数不要离 meta-initialization 太远：

```text
L_anchor = || Δθ_env ||²
```

或使用更精细的重要性加权 regularization。

---

### 11.5 `L_replay`

用 replay / imagined contexts 保持泛化能力：

```text
F_{θ_env}(z, a, c_j) ≈ F_{θ_meta}(z, a, c_j)
π_{θ_env}(z, g, c_j) ≈ π_{θ_meta}(z, g, c_j)
```

其中 `c_j` 来自旧环境或 latent imagination。

---

## 12. 推荐整体架构

最终架构可以总结为：

```text
Observation o_t, goal g
        ↓
Rich visual-language-proprio encoder
        ↓
Working-state extractor / relevance router
        z_t = R_φ(o_t, g, c)
        ↓
        ├── Policy:
        │       a_t = π_θ(z_t, g, c)
        │
        ├── State world model:
        │       ẑ_{t+1:t+H} = F_θ(z_t, a_{t:t+H-1}, c)
        │
        └── State-guided video head:
                î_{t+1:t+H}
                = V_ψ(o_t, z_t, ẑ_{t+1:t+H}, a_{t:t+H-1}, c)
```

其中：

* policy 不看 `ẑ_future`；
* state world model 根据 action 和 `c` 预测未来 `z`；
* video head 根据 predicted state future、action 和 `c` 生成未来视觉；
* `c` steer 所有模块，保证 test-time adaptation。

---

## 13. 训练流程

### 13.1 Meta-training episode

对于每个训练环境 `e_i`：

1. 采样 support trials `S_i`；
2. 用 `S_i` 更新 `c_i`；
3. 采样 query trials `Q_i`；
4. 用 query loss 更新 `θ`，让少量 `c` 更新后的模型在同一环境中泛化。

---

### 13.2 Inner-loop loss

```text
L_inner(c)
=
λ_W L_state_pred
+
λ_BC L_BC
+
λ_RL L_RL
+
λ_context L_context
+
λ_reg ||c - c0||²
```

---

### 13.3 Outer-loop loss

```text
L_outer(θ)
=
L_state_pred_query
+
L_policy_query
+
λ_video L_video_aux
+
λ_consistency L_state_video_consistency
+
λ_context L_context_regularization
```

---

### 13.4 Context regularization

为了让 `c` 不是 episode ID，而是真正的 physical context，需要：

* temporal consistency；
* same-env contrastive consistency；
* different-env separation；
* latent smoothness；
* optional factorization；
* optional privileged physical parameter prediction；
* uncertainty calibration。

例如：

```text
same environment → similar c
different dynamics → separated c
nearby c → nearby predicted dynamics
interpolated c → physically plausible dynamics
```

---

## 14. Latent space 的要求

`c` latent space 必须满足几个性质。

### 14.1 Few-trial identifiability

少量 trial 后能定位当前环境。

```text
same action, different outcome
    → prediction error
    → update c
```

---

### 14.2 Expressiveness

latent space 要足够表达多数新环境。

不能太小，否则很多环境找不到对应 `c`；
也不能太大，否则 few-shot optimization 不稳定。

可以考虑 factorized latent：

```text
c = [c_env, c_obj, c_robot, c_contact]
```

其中：

* `c_env`：surface friction、external disturbance；
* `c_obj`：mass、stiffness、deformability；
* `c_robot`：actuator gain、delay、gripper bias；
* `c_contact`：sliding、sticking、jamming、compliance mode。

这不是 object-centric prior，而是 variation-source prior。

---

### 14.3 Smoothness / interpolation

好的 latent space 应该支持插值：

```text
nearby c → nearby dynamics
interpolated c → plausible dynamics
```

---

### 14.4 Prediction-grounded

`c` 的含义必须由 world prediction error grounding，而不是普通 prompt embedding。

```text
c* = argmin_c Σ || z_{t+1} - F(z_t, a_t, c) ||²
```

---

### 14.5 Policy-controllable

`c` 必须能改变 policy 行为：

```text
π(z, g, c_low_friction)
≠
π(z, g, c_high_friction)
```

否则 world model 知道环境变了，但 policy 不会调整动作。

---

## 15. Video head 的训练原则

video head 应作为辅助监督，而不是主模型。

推荐 loss：

```text
L_video =
D(visual_target, V_ψ(o_t, z_t, ẑ_future, a, c))
```

其中 `D` 可以是：

* perceptual feature loss；
* latent video feature loss；
* flow loss；
* changed-region loss；
* contact-region reconstruction；
* future image loss。

同时加入：

```text
L_state_video_consistency =
|| R(E(î_{t+k}), g, c) - ẑ_{t+k} ||²
```

核心原则：

> video generation 应该被 state world model 指导，而不是替代 state world model。

---

## 16. Policy 与 world model 的逻辑关系

### 16.1 Policy

```text
a_t = π(z_t, g, c)
```

policy 是当前状态反馈控制器。

---

### 16.2 World model

```text
ẑ_future = F(z_t, a, c)
```

world model 是 action-conditioned predictor。

---

### 16.3 Planner / teacher

world model 可以评估 candidate actions：

```text
a* = argmax_a Score(F(z_t, a, c), g)
```

然后训练 policy：

```text
π(z_t, g, c) ≈ a*
```

---

### 16.4 Video head

```text
future image = V(o_t, z_t, ẑ_future, a, c)
```

video head 是 state-conditioned visual decoder。

---

## 17. 第一阶段必须验证的四个问题

这一板块比完整系统设计更优先。我们不能一开始就假设 `c`、world model、policy 和 working-state extractor 都自然有效，而应该先用小规模、可控任务验证四个关键问题。

这里的核心适应目标不是 language-level memory，也不是普通历史记忆，而是：

> **language 很难描述清楚的低层物理直觉。**

包括摩擦、刚度、质量分布、接触模式、执行器增益、夹爪迟滞、以及物体在受力后的细微运动轨迹。物体的 subtle trajectory 也属于这种 physical instinct，因为它真正表达的是：

```text
I applied this action, and the world responded in this low-level motion pattern.
```

这类信息很难用语言直接转成可执行控制量。模型必须从 action-conditioned motion residual 中形成隐式物理判断，并把它转化为力、幅度、方向、时序、接触策略或恢复动作的变化。

---

### 17.1 问题一：能否训练出足够可控的 latent `c`

第一个必须验证的问题是：

> 能不能训练出一种 latent `c`，它不是 episode ID，也不是普通 visual embedding，而是真的能 steer world model 和 policy。

我们应该分别尝试两条路线：

1. **Video / WAM 路线**

   直接在 video generation / Fast-WAM-style latent 上测试 `c` 是否能改变未来预测。例如固定当前 observation 和 action，只改变 `c`，看未来运动是否符合物理直觉：

   * high friction `c` → 物体滑动距离变短；
   * low friction `c` → 物体滑动距离变长；
   * weak gripper / rusty claw `c` → 同样 action 下夹爪或物体位移更小；
   * stiffness `c` → deformable object 的形变模式改变。

2. **State model 路线**

   第一版可以先使用 human prior 抽取低维 state，例如 object position、velocity、pose、end-effector displacement、contact / slip cue。然后验证：

   ```text
   ẑ_{t+1} = F(z_t, a_t, c)
   ```

   中的 `c` 是否能稳定调节 state transition。

成功标准不是 visual reconstruction 好看，而是 `c` 对 motion / state transition 的影响方向正确、连续、可插值，并且能覆盖不同 physical property 的变化。

这里必须特别警惕一个失败模式：

> `c` 变成 shortcut。

如果 rollout 很少，`c` 很容易不去表示 friction、stiffness、actuator gain 这类真正的 physical property，而是直接记住 support rollout 里的视觉特征、初始位置、某个固定轨迹片段或某个 episode identity。这样它在 support loss 上可能有效，但本质上只是过拟合了那几个样本。

因此验证 `c` 时必须检查：

* support rollout 上更新出来的 `c` 是否能泛化到 query rollout；
* 同一环境下不同初始状态 / 不同动作序列得到的 `c` 是否一致；
* 改变 `c` 是否产生连续、物理合理的 motion change，而不是离散记忆某个 episode；
* `c` 的维度和更新步数是否足够小，避免把整段 rollout 信息塞进 latent；
* `c` 是否不能直接访问未来 query 信息，避免训练时泄漏。

如果这一点做不到，后面的 test-time adaptation 没有基础，因为即使我们在测试时找到了某个 `c`，它也无法真正改变模型对当前世界的预测和动作。

---

### 17.2 问题二：能否用少量真实 rollout 快速识别 `c`

第二个必须验证的问题是：

> 给模型极少量当前测试环境的 trajectory / rollout，能不能高效找到一个有用的 `c`。

这对应 test-time adaptation 的核心。我们要验证的不是离线 supervised prediction，而是 few-rollout adaptation：

```text
c ← c - α ∇_c L_inner
```

其中 `L_inner` 应该主要来自低层物理响应误差，例如：

* predicted state / actual state mismatch；
* predicted object motion / actual object motion mismatch；
* optical flow / point motion residual；
* end-effector proprioceptive transition residual；
* slip / jamming / contact outcome mismatch；
* action outcome consistency。

关键是这些 loss 不能只是让模型“记住刚才发生了什么”，而要让 `c` 变成当前环境的 physical context。也就是说，更新后的 `c` 应该能提升后续 trajectory 的预测和控制，而不是只解释已经看过的片段。

这里和 17.1 一样存在 shortcut concern，而且在 few-rollout 设置下更严重。因为 inner-loop optimization 本身就有能力把少量 support rollout 压进 `c`，所以我们要验证的是：

```text
c represents transferable physical response,
not compressed support-rollout memory.
```

可能的防护方式包括：

* 明确区分 support rollout 和 query rollout，所有关键指标都在 query 上评估；
* 用同一 hidden property 下的不同初始状态、不同动作、不同物体姿态做 query；
* 控制 `c` 维度、更新步数和 learning rate；
* 对 `c` 加 temporal smoothness / norm regularization / information bottleneck；
* 用 same-environment contrastive consistency，让同一物理环境下不同 rollout 的 `c` 靠近；
* 用 different-environment separation，让不同摩擦、刚度、增益条件下的 `c` 可区分；
* 避免让 `c` 直接接收 raw image patch 或完整 future trajectory，降低它记样本细节的机会。

初步实验可以做成：

```text
0 rollout / 1 rollout / 3 rollouts / 5 rollouts
```

观察：

* world prediction error 是否下降；
* latent traversal 是否更符合当前环境；
* 后续 policy success 是否提升；
* 对未见过的动作和状态是否仍然有效。

如果只能 overfit support rollout，而不能改善 query rollout，说明 `c` 没有学到真正的环境 property。

---

### 17.3 问题三：world model 学到的物理信息能否传到 policy

第三个必须验证的问题是：

> world model 通过 `c` 学到了当前环境的 physical property 之后，policy 是否真的会用这些信息改变动作。

这是之前 FastWAM-TTT 初步实验暴露出的核心风险：TTT 或 world prediction 可能确实改变了 representation，但 action head 仍然主要依赖 BC backbone，导致行为几乎不变。

因此必须验证 policy-world bridge，而不是只验证 prediction。

可能的桥接方式包括：

1. **共享 `c`**

   同一个 `c` 同时输入 world model 和 policy：

   ```text
   ẑ_{t+1} = F(z_t, a_t, c)
   a_t = π(z_t, g, c)
   ```

   然后检查改变 `c` 是否会导致合理的 action change。

2. **world model teacher / planner**

   用 adapted world model 评估 candidate actions，得到更适合当前环境的 action target：

   ```text
   a* = argmax_a Score(F(z_t, a, c), g)
   π(z_t, g, c) ≈ a*
   ```

   这样 policy 被训练成利用 `c` 的快速 controller，而不是独立于 world model 的 BC head。

3. **query trajectory outer loss**

   inner loop 用 support rollout 更新 `c`，outer loop 在 query rollout 上训练 action generation / task success。这样模型会被迫学习：

   ```text
   useful c update → better future action
   ```

成功标准是：相同 observation 和 goal 下，不同 `c` 会产生符合物理直觉的动作变化。例如高摩擦时更大推力，低摩擦时更保守的推力，夹爪弱时更大闭合幅度或更长保持时间，插入偏移时改变接触修正方向。

如果 world model prediction 变好了但 policy success 没变，说明 bridge 失败；这时不能继续堆更复杂的 world model，而要优先修 policy 对 `c` 的使用。

---

### 17.4 问题四：能否抽取任务阶段相关的重要 state

第四个问题是最难的，也最可能成为真正突破：

> 模型能否在一个具体任务阶段中自动抽取当前最重要的 low-dimensional working state。

这个 state 不是固定 object state，也不是通用 object-centric representation。它应该随任务阶段变化：

* pushing 阶段：object position、velocity、rotation、slip trend、目标方向；
* grasping 阶段：夹爪相对位姿、物体微小滑动、接触稳定性；
* insertion 阶段：tip clearance、axis alignment、jamming cue、contact residual；
* collision-risk 阶段：hand-table clearance、safe direction、collision margin。

第一版不应该强行要求完全自动发现 working state。更务实的路径是：

1. 先用 human prior / simulator state / tracking tool 抽取 state，验证 `c` 和 adaptation 机制是否成立；
2. 再尝试让 model 从 image / depth / proprioception 中学习提取这些 state；
3. 最后探索是否可以用 attention、prediction error、goal conditioning 或 VLM prior 自动发现阶段相关 state。

这一问题如果能解决，会是很强的 contribution；但它不应该成为第一阶段系统是否成立的唯一前提。第一阶段最重要的是先验证：

```text
low-level physical response → identifiable c → controllable world model → changed policy behavior
```

---

## 18. 实验设计建议

### 18.1 Task categories

可以选几类 dynamics shift 明确的任务：

1. **Pushing**

   * 高/低摩擦；
   * 不均匀质量分布；
   * object slide distance 变化。

2. **Rope / cable / deformable manipulation**

   * 绳子更硬；
   * 绳子更软；
   * damping / stiffness 变化。

3. **Plug insertion**

   * plug/socket 几何偏移；
   * insertion axis 变化；
   * jamming / contact shift。

4. **Actuator / gripper adaptation**

   * rusty claw；
   * same command, smaller displacement；
   * delay / gain mismatch。

5. **Collision-aware adaptation**

   * 插入过程中手靠近桌面；
   * working state 从 alignment 切换到 collision avoidance。

---

### 18.2 Key ablations

需要比较：

* no `c`；
* random `c`；
* `c` only in world model；
* `c` only in policy；
* `c` in both world model and policy；
* `c` also steers working-state extractor；
* no video head；
* video head as main model；
* state-guided video head；
* fast `c` update only；
* fast `c` + slow habit update；
* slow update without preserving c-control；
* slow update with c-control preservation。

---

### 18.3 Few-trial adaptation curve

横轴：

```text
0 trial / 1 trial / 3 trials / 5 trials / 10 trials
```

纵轴：

* state prediction error；
* policy success rate；
* reward；
* collision rate；
* jamming recovery rate；
* video-state consistency。

---

### 18.4 Latent traversal

固定 `z_t` 和 action，改变 `c`，观察预测是否符合物理直觉：

* friction direction 增大 → slide distance 变短；
* stiffness direction 增大 → deformation 变小；
* actuator damping 增大 → displacement 变小；
* contact offset 变化 → 插入轨迹修正方向变化。

---

### 18.5 Held-out combinations

训练见过：

```text
low friction + normal actuator
high friction + delayed actuator
```

测试：

```text
low friction + delayed actuator
```

如果成功，说明 `c` 不是 memorizing environment ID，而是在组合物理因素。

---

### 18.6 Working-state switching test

测试模型是否能在任务阶段变化时切换 working state：

* alignment → insertion；
* insertion → jamming；
* jamming → retreat / retry；
* normal manipulation → collision avoidance。

---

## 19. 可能的方法名称

可以考虑：

* **AWSM**: Adaptive Working-State Model；
* **AWM**: Adaptive Working-state World Model；
* **PCLA**: Physical Context Latent Adaptation；
* **BELA**: Belief-based Latent Adaptation；
* **State-Guided Video World Model**；
* **Context-Preserving Habit Consolidation**；
* **Adaptive Working-State World Models for Test-Time Robotic Adaptation**。

一个比较完整的标题方向：

> **Adaptive Working-State World Models for Test-Time Robotic Adaptation**

或者：

> **Learning Physical Context Latents for Adaptive Working-State Control**

或者：

> **From Pixels to Working States: Context-Adaptive World Models for Robotic Manipulation**

---

## 20. 方法核心 claim

可以用下面这段作为 proposal 摘要：

> We propose an adaptive working-state world model for robotic test-time adaptation. Instead of predicting future pixels or relying on fixed object-centric states, the model learns to extract a compact task-relevant working state from high-dimensional observations. A test-time optimized physical context latent serves as an implicit belief over the current environment dynamics and steers the working-state extractor, state world model, policy, and auxiliary video head. The policy acts from the current working state, goal, and context, while the state world model predicts the consequences of candidate actions under that context. A state-guided video generation head provides auxiliary visual supervision and interpretable rollouts without becoming the primary substrate of control. During meta-training, the outer loop learns a model initialization that can be rapidly adapted through few-shot context updates. During deployment, fast adaptation updates only the context latent, while optional slow consolidation writes the fast-adapted behavior into environment-specific parameters without destroying the controllability of the context latent.

中文对应：

> 我们提出一个用于机器人测试时适应的自适应 working-state 世界模型。它不预测未来像素，也不依赖固定 object-centric 状态，而是从高维 observation 中提取紧凑的任务相关 working state。测试时优化的 physical context latent 作为当前环境动力学的隐式 belief，同时调节 working-state extractor、state world model、policy 和辅助 video head。policy 根据当前 working state、目标和 context 直接选动作；state world model 根据 action 和 context 预测未来 working state；state-guided video head 提供辅助视觉监督和可解释 rollout，但不成为控制主线。训练时 outer loop 学习一个可以通过少量 context update 快速适应的初始化；部署时 fast loop 只更新 context latent，而可选的 slow consolidation 将 fast-adapted 行为写入当前环境特化参数，同时保留 context latent 对模型的控制能力。

---

## 21. 最终一句话总结

这个课题的核心不是“做一个更小的 VLA”，也不是“加一个 world model head”，而是：

> **让机器人像语言模型拥有 token、像棋类 AI 拥有 board state 一样，自动从高维 embodied observation 中发现当前任务真正需要的 working state；再通过一个 few-trial 学出来的 physical context latent，把这个 working-state world model 和 policy 快速适应到当前测试环境。**

更短地说：

> **Learn what matters now, infer what kind of world this is, and act in that adapted working state.**
