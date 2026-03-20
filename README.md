# FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization

🏠 [Homepage](https://qwen-pilot.notion.site/fipo) | 📝 [Paper PDF](./assets/FIPO_Eliciting_Deep_Reasoning_with_Future_KL_Influenced_Policy_Optimization.pdf) | 🤗 [Hugging Face](https://huggingface.co/Henrymachiyu/FIPO) | 🤖 [ModelScope](https://modelscope.cn/models/Henrymachiyu/FIPO) | 🐱 [GitHub](https://github.com/Henrymachiyu/FIPO)

**Qwen Pilot, Alibaba Group | Published on March 20, 2026**

FIPO is a value-free RL recipe for eliciting deeper reasoning from a clean base model. The central idea is simple: GRPO-style training works, but its token credit assignment is too coarse. FIPO densifies that signal with a discounted Future-KL term that reflects how the rest of the trajectory evolves after each token.

## Overview

![Figure 1 overview](assets/readme/fig1.png)

*Figure 1. FIPO vs. baselines on AIME 2024. FIPO shows that pure RL training alone can outperform reproduced pure-RL baselines such as DAPO and DeepSeek-R1-Zero-32B, while also producing substantially longer responses on average.*

- FIPO is a **value-free RL algorithm** designed to overcome reasoning bottlenecks in ORM-based GRPO training.
- Standard GRPO assigns the same outcome-level advantage to every token in a trajectory. We argue that this **coarse credit assignment** creates a lower ceiling, because the model cannot distinguish critical reasoning steps from routine continuation tokens.
- FIPO replaces that uniform treatment with a **dense advantage formulation**: each token is reweighted by the discounted signed policy shift of its future trajectory.
- On **Qwen2.5-32B-Base**, FIPO breaks the typical reasoning-length plateau, extending average chain-of-thought length from roughly **4,000** to **10,000+** tokens and improving **AIME 2024 Pass@1** from **50.0%** to a peak of **58.0%**.
- More broadly, FIPO suggests that improving **advantage density inside the GRPO framework** is a promising path for eliciting deep reasoning without relying on long-CoT SFT or a separate value model.

## Core Change

FIPO keeps the standard PPO/DAPO scaffold, but changes how token-level updates are weighted. The local signal is the signed log-probability shift between the current and old policy:

$$
\Delta \log p_t = \log \pi_\theta(y_t \mid x, y_{1:t-1}) - \log \pi_{old}(y_t \mid x, y_{1:t-1})
$$

Positive values mean the token is being reinforced, while negative values mean it is being suppressed. Since reasoning is sequential, FIPO then accumulates this signal over the future trajectory:

$$
FutureKL_t = \sum_{k=t}^{T} M_k \cdot \gamma^{k-t} \cdot \Delta \log p_k
$$

Positive `FutureKL_t` means the future following token `t` is being reinforced; negative `FutureKL_t` means it is being suppressed. The decay window keeps the signal local enough to stay stable, while the mask removes extreme-ratio outliers.

FIPO maps this future signal into a bounded influence weight:

$$
f_t = clip(\exp(FutureKL_t), 1-\epsilon_{f,low}, 1+\epsilon_{f,high}), \quad \tilde{A}_t = \hat{A}_t \cdot f_t
$$

Tokens that lead into preferred futures are **amplified**, while tokens that lead into suppressed futures are **attenuated**. Clipping keeps this modulation stable. The final DAPO-style loss therefore stays clipped and simple, but the advantage term becomes **future-aware** rather than uniformly inherited from the final outcome.

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
