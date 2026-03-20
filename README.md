# FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization

🏠 [Homepage](https://qwen-pilot.notion.site/fipo) | 📝 [Paper PDF](./assets/FIPO_Eliciting_Deep_Reasoning_with_Future_KL_Influenced_Policy_Optimization.pdf) | 🤗 [Hugging Face](https://huggingface.co/Henrymachiyu/FIPO) | 🤖 [ModelScope](https://modelscope.cn/models/Henrymachiyu/FIPO) | 🐱 [GitHub](https://github.com/Henrymachiyu/FIPO)

**Qwen Pilot, Alibaba Group | Published on March 20, 2026**

FIPO is a value-free RL recipe for eliciting deeper reasoning from a clean base model. The central idea is simple: outcome-based GRPO training is effective, but its credit assignment is too coarse. FIPO densifies that signal by reweighting token-level updates with a discounted Future-KL term that reflects how the rest of the trajectory evolves after each token.

## Summary

- We introduce **Future-KL Influenced Policy Optimization (FIPO)**, a reinforcement learning algorithm designed to overcome reasoning bottlenecks in ORM-based GRPO training.
- Standard GRPO assigns the same outcome-level advantage to every token in a trajectory. We argue that this **coarse-grained credit assignment** creates a lower performance ceiling, because the model cannot distinguish critical reasoning pivots from trivial continuation tokens.
- FIPO replaces that uniform treatment with a **dense advantage formulation**: each token is reweighted according to the discounted signed policy shift of its future trajectory.
- On **Qwen2.5-32B-Base**, FIPO breaks the typical reasoning-length plateau of standard baselines, extends average chain-of-thought length from roughly **4,000** to **over 10,000** tokens, and improves **AIME 2024 Pass@1** from **50.0%** to a peak of **58.0%**.
- More broadly, FIPO suggests that improving **advantage density inside the GRPO framework** is a promising direction for eliciting deep reasoning without relying on long-CoT SFT or a separate value model.

## Introduction

![Figure 1 overview](assets/readme/fig1.png)

*Figure 1. FIPO vs. baselines on AIME 2024. FIPO shows that pure RL training alone can outperform reproduced pure-RL baselines such as DAPO and DeepSeek-R1-Zero-32B, while also producing substantially longer responses on average.*

Modern reasoning models increasingly rely on **inference-time scaling**: reinforcement learning drives longer and more deliberate chains of thought. But reproducing that behavior cleanly in the open remains difficult, since many strong pipelines still depend on undisclosed tricks, long-CoT supervision, or critic-based guidance.

This motivates a core question:

> Can we elicit deep reasoning from a clean base model, without relying on long-CoT synthetic data or critic-based token supervision?

Value-free recipes such as DAPO show that GRPO-style training can already improve reasoning. However, they often hit a structural ceiling: response length grows at first, then stalls around the 4k-token regime. We view this as a **coarse credit-assignment** problem. When every token receives the same final-answer-derived advantage, the optimizer cannot distinguish critical reasoning pivots from routine continuation tokens.

FIPO addresses this limitation with a **future-aware token reweighting mechanism**. Instead of only asking whether a sampled response is correct, FIPO also asks whether the updated policy is reinforcing or suppressing the **future trajectory initiated by each token**. This yields a denser training signal while staying inside a critic-free GRPO-style loop.

## Core Change

FIPO keeps the standard PPO/DAPO scaffold, but changes how token-level updates are weighted. The core idea is to reward tokens that lead into futures the updated policy prefers, and down-weight tokens that lead into futures it is already moving away from.

### Probability Shift as the Atomic Signal

FIPO starts from the signed log-probability difference between the current policy and the old policy:

```math
\Delta \log p_t = \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{old}}(y_t \mid x, y_{<t})
```

This quantity is treated as a directional signal of local policy movement:

- If `Δ log p_t > 0`, the updated policy is reinforcing that token.
- If `Δ log p_t < 0`, the updated policy is suppressing that token.

Unlike a standard KL penalty, FIPO uses this drift as a signal of **behavioral adjustment**. But reasoning is sequential, so a local shift alone is not enough.

### Future-KL as Forward-Looking Credit Assignment

To capture the downstream effect of a token on the rest of the sampled reasoning chain, FIPO accumulates discounted signed probability shifts over the future trajectory:

```math
\mathrm{FutureKL}_t = \sum_{k=t}^{T} M_k \cdot \gamma^{k-t} \cdot \Delta \log p_k
```

Here, the decay factor models diminishing causal dependency over longer horizons, while the mask removes extreme negative-advantage outliers that would otherwise destabilize the accumulation.

Functionally:

- Positive `FutureKL_t` means the future trajectory following token `t` is being reinforced.
- Negative `FutureKL_t` means that future trajectory is being suppressed.
- The decay window keeps the signal focused on the effective reasoning horizon instead of letting distant stochasticity dominate.

This is the core intuition behind FIPO: **the value of a token depends on the future it leads into.**

### Future-KL Reweighted Objective

FIPO maps Future-KL into a bounded multiplicative influence weight:

```math
f_t = \operatorname{clip}\left(\exp(\mathrm{FutureKL}_t), 1-\epsilon_{f,\mathrm{low}}, 1+\epsilon_{f,\mathrm{high}}\right), \quad \tilde{A}_t = \hat{A}_t \cdot f_t
```

Operationally:

- If a token leads into a future the updated policy prefers, its update is **amplified**.
- If a token leads into a future the policy is suppressing, its update is **attenuated**.
- Clipping keeps that modulation controlled and numerically stable.

Under the token-level DAPO formulation, the final FIPO loss remains a clipped policy-gradient objective, but the advantage term is now **future-aware** rather than uniformly inherited from the final outcome.

### Why This Matters in Practice

FIPO is appealing because it achieves dense token-level supervision **without introducing a value model**. In other words, it stays close to the efficiency and simplicity of GRPO/DAPO, while addressing the exact place where these methods often struggle most: sustaining coherent long-range reasoning growth.

In this repo, the implementation is mainly localized to:

- `verl/trainer/ppo/core_algos.py`
- `verl/workers/config/actor.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/actor/megatron_actor.py`

The provided 32B launcher enables `future_kl` loss together with the corresponding decay, clipping, and safety controls used in our main experiments.

## Getting Started

FIPO is built on top of the existing VeRL and DAPO training stack in this repository.

- Follow the standard VeRL environment setup and cluster preparation flow.
- Reuse the same Ray runtime pattern as the DAPO recipe.
- Use the new FIPO launcher in `recipe/fipo/` as the default 32B entrypoint.

Useful local references:

- DAPO recipe overview: [`recipe/dapo/README.md`](./recipe/dapo/README.md)
- DAPO baseline launcher: [`recipe/dapo/run_dapo_qwen2.5_32b.sh`](./recipe/dapo/run_dapo_qwen2.5_32b.sh)
- FIPO launcher: [`recipe/fipo/run_fipo_qwen2.5_32b.sh`](./recipe/fipo/run_fipo_qwen2.5_32b.sh)

## Training

The recommended launcher is:

```bash
bash recipe/fipo/run_fipo_qwen2.5_32b.sh
```

A typical submission flow looks like this:

```bash
cd FIPO
bash recipe/fipo/run_fipo_qwen2.5_32b.sh
```

The script keeps the DAPO-style defaults and leaves the major paths env-overridable, including `MODEL_PATH`, `TRAIN_FILE`, `TEST_FILE`, `CKPTS_DIR`, and `NNODES`.

## What changed in the scripts?

Compared with the DAPO 32B launcher, the FIPO launcher preserves the same overall training entrypoint and rollout structure, but changes the optimization behavior in a few important ways:

- `actor_rollout_ref.actor.ppo_mini_batch_size` is increased from `32` to `64` for better stability at 32B scale.
- `actor_rollout_ref.actor.policy_loss.loss_mode` switches from the default PPO-style objective to `future_kl`.
- FIPO-specific knobs are added through `policy_loss`, especially the Future-KL decay horizon, influence-weight clipping range, start mode, averaging behavior, and safety threshold.
- The practical effect is a training loop that stays very close to DAPO operationally, while making the **policy update itself** more future-aware and more robust for long-chain reasoning.

## 📊 Results & Figures

### Training Dynamics

The most important qualitative observation in FIPO is that performance gains are tightly coupled with **continuous response-length scaling**.

Under the DAPO baseline, response length grows at first and then gradually stalls around the 4k-token regime. Under FIPO, the model continues to expand its reasoning budget instead of collapsing into that intermediate plateau. This is not just a tail effect driven by a few unusually long samples. The response-length distribution shifts upward more broadly during training.

More importantly, these extra tokens are not merely verbosity. They increasingly support self-reflection, re-derivation, intermediate checking, and multi-pass verification. In other words, FIPO does not simply make the model speak longer. It helps the model use additional length as **genuine reasoning depth**.

![Training dynamics placeholder](assets/readme/response_length.png)

*Figure 2. Dynamics of response length and performance scaling during training. Compared to the DAPO baseline, FIPO significantly increases response length and maintains a strong positive correlation between longer chain-of-thought and higher accuracy.*

### Main Result

FIPO is designed to lengthen and deepen reasoning under the same DAPO-style training scaffold rather than replacing the whole pipeline. In the paper's 32B setting, the main takeaway is straightforward: the FIPO objective yields longer responses and a stronger AIME 2024 peak than the DAPO baseline.

- DAPO baseline: **50.0%** AIME 2024 Pass@1
- FIPO: **58.0%** peak AIME 2024 Pass@1, converging around **56.0%**
- Response length: roughly **4k** to **10k+** tokens during training

The broader claim of the paper is that this gain comes from a stronger token-level credit-assignment mechanism, not from a separately trained value model or long-CoT warm-up supervision. FIPO shows that pure RL on a clean base model can already push reasoning trajectories much further than standard value-free baselines.

![Main results placeholder](assets/readme/main_results.png)

*Main 32B result figure. FIPO outperforms reproduced pure-RL baselines on AIME 2024 while also producing substantially longer responses on average.*

## 🎈 Citation

If you find this work useful, please cite:

```bibtex
@misc{FIPO,
  title = {FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization},
  url = {https://qwen-pilot.notion.site/fipo},
  author = {Chiyu Ma and Shuo Yang and Kexin Huang and Jinda Lu and Haoming Meng and Shangshang Wang and Bolin Ding and Soroush Vosoughi and Guoyin Wang and Jingren Zhou},
  year = {2026},
  month = {March},
}
```

## 🌻 Acknowledgement

This project builds on top of the **VeRL** training framework and follows the practical recipe structure introduced by **DAPO**.

- VeRL repository: <https://github.com/volcengine/verl>
- DAPO recipe in this repo: [`recipe/dapo`](./recipe/dapo)
