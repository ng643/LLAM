#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_live_mujoco.py — online policy-gradient fine-tuning of the LoopLM robot
policy **inside the live MuJoCo bridge** (A2C: REINFORCE + a learned value
baseline), on the Radeon 880M through ROCm/HIP.

Why this file exists
--------------------
`train_loop_robot.py` fits the adapter by imitating a hand-written expert on
synthetic arrays.  That leaves the policy flat whenever the synthetic expert is
uninformative (`--trainable adapters` on a random trunk ⇒ the head collapses to
one token, e.g. a permanent `-y`).  This script closes the loop with the real
physics engine: the reward is the *measured* distance to the instructed sphere
— the same `goal` telemetry that `mujoco_env_bridge.py` prints — so a policy
that does not move toward the target earns nothing.

It runs in two stages, because the measurement below says the *first* problem is
not the reward but the distribution:

  1. `LiveImitator` — DAgger round 0 (`--imitate-*`).  The learner drives the
     live scene, the same expert that labelled the synthetic data labels the
     states the learner actually reaches, and the adapter is fitted by
     cross-entropy (+ τ·ponder).  This is what repairs a policy whose forward
     pass is *constant* on live states — A2C cannot, because the action such a
     state needs is never sampled, so its advantage is never observed.
  2. `A2CUpdater` — the rollout-buffer policy gradient described below, which
     now starts from a policy that already tracks and can therefore improve the
     parts the expert is bad at (the expert is a rule, not an oracle).

Data flow (one control frame; nothing here resets the latent carry)
-------------------------------------------------------------------
  MuJoCo (CPU, 500 Hz)                        PyTorch (GPU, ROCm/HIP)
  --------------------                        ----------------------
  mj_step × substeps  ──► qpos/qvel/site ──► observe() → 28-D frame
        ▲                                       │  (telemetry + expert labels)
        │                                 render_state() → ONE sentence
        │                                       │
        │                             StateWindow (S sentences, oldest first)
        │                                       │  encode → sids (1,S·L)
        │                                       │  answer markers → apos (1,S,Q)
        │                                       ▼   .to(device) — int64, no staging
        │               model(instr_ids, instr_mask, sids, apos, carry)
        │                          │  gated sensor injection + ACT loop (≤ max_loops)
        │                          ├─ q_logits (Jev labels) ─► sample per question ─► JSON ─► Command
        │                          │     (a_t = json_action(labels), telemetry; logπ = Σ_q log P_q)
        │                          ├─ speech: END at the last state position = silent,
        │                          │     else greedy ≤ --speak-max tokens ─► info['say'] / JSON "say"
        │                          ├─ out.carry ─────────────► Critic ─► V(s_t)
        │                          └─ ponder / cycles  (ACT compute cost)
        │                                       │
        └── data.ctrl ◄── command(a_t) ◄── torque map (trained policy ⇄ same map as
                                            the bridge: task-space PD + bias comp)
                                            │
                                r_t = w_p·(d_t − d_t+1) + w_d·(1 − d/ref)
                                      + reach / hold bonuses − ponder penalty
                                            │
                          ┌─────────────────┴───────────────────┐
                          │  ROLLOUT BUFFER (--rollout frames)  │
                          │ sids · apos · carry · a_t · r_t · done │
                          └─────────────────┬───────────────────┘
                                            │  every --rollout frames:
                          G_t = r_t + γ·G_t+1   (cut at `done`; the tail is
                                    bootstrapped by the value of the frame after)
                          A_t = normalise(G_t − V(s_t))
                          L   = mean[−A_t·logπ(a_t|s_t) − β·H(π)] + ½·(V(s_t) − G_t)²
                                            │
                                 one AdamW step (adapter + critic)

What is and is not back-propagated through time
-----------------------------------------------
* **Inside a frame**: yes — the loss reaches every ACT cycle of the loop, all
  repeated trunk applications and the sensor-injection gate.  That is real BPTT
  through the internal recurrence.
* **Across frames**: the model returns `LoopOutput.carry` already `.detach()`ed
  (the inference-threading contract), so the *latent* path carries no gradient.
  Cross-frame credit assignment is therefore the discounted return `G_t` over a
  rollout, with the value baseline supplying the horizon beyond it.
* **Why the graph is batched, not per-frame**: a per-frame draft that
  bootstrapped each transition from the *next* frame's value leaves frame t's
  graph alive while the optimizer writes the parameters in place — autograd then
  raises "one of the variables needed for gradient computation has been modified
  by an inplace operation".  Collecting the rollout under `no_grad` and building
  *and consuming* the graph inside a single `update()` call removes the hazard:
  no graph ever outlives the call that created it.  It is also faster — one
  kernel-launch set per rollout instead of one per frame.

Is it learning? (why the yardstick is a protocol, not the training stream)
-------------------------------------------------------------------------
The spheres are *force-driven* along a Lissajous trajectory, so the distance in
the training stream is dominated by where the target wandered, not by what the
policy did — unchanged weights send it up and down.  Every eval therefore runs
through `EvalProbe`: arm back in the ready pose, spheres back at their MJCF start
poses, `sim_time = 0` (identical trajectory phase), the instruction currently in
play, `--eval-frames` **greedy** frames with an empty carry and a fresh window.
`d0` is the determinism check — it must come out identical in every eval — and
`d_mean` / `best` / `reach%` / `t_reach` are then attributable to the policy.
Two evals only compare when the *goal* matched, though: the protocol restarts
the arm in the same place, but a different instructed sphere is a different
task, so the summary groups the series into same-goal runs and reports the
trend inside each one.
The probe snapshots MuJoCo's state, the latent carry and the observation window
and restores all three, so it cannot perturb training (MuJoCo's internal caches
are merely recomputed).

Measured on this box (Radeon 880M · torch 2.12.0+rocm7.14.1 · gfx1150 · 8 GiB carve-out)
---------------------------------------------------------------------------------------
Cost — tiny trunk (0.038M adapter + 0.034M critic, bf16, rollout 32 / batch 8):
  9.2 Hz (76–85 ms/f) · peak 0.08 GiB of the cap · OOM 0.  Qwen2.5-0.5B (6.87M
  adapter + 0.46M critic, rollout 16 / batch 4): peak 3.8 GiB of the cap —
  `--update-batch` is the VRAM knob.
  Both figures are *after* the `--gc-every` fix: the loop used to call a full
  `gc.collect()` every frame (215–260 ms, 78 % of the frame) and ran at 3.0–3.3 Hz
  — see `train_loop_robot.freeze_gc()` for the measurement.  Per-frame kernel
  launches now dominate (one forward costs the same at S=8 and S=24), so the
  trunk, not the window length, is what costs time.

Does it learn?  Warm start from a `train_loop_robot.py` adapter, ONE fixed goal
(`--retarget-every 0 --goal red`), tiny trunk on the repaired bridge, 3 DAgger
rounds (~140 s of GPU) followed by A2C frames, eval every 250 frames (40 greedy
frames).  Three runs, same recipe and seed, different warm starts:
                                                   baseline  imitation  final   best
  `_sup3.pt` → 1500 A2C frames                     0.519m    0.052m     0.080m  0.070m
                                                            72.5 %     40.0 %
  `_sup4.pt` → `--frames 1` (imitation only)       0.519m    0.076m     0.075m  0.076m
                                                            27.5 %     27.5 %
  `_sup4.pt` → 1500 A2C frames                     0.519m    0.095m     0.127m  0.073m
                                                                       0.0 %
  ceiling (expert rule, same protocol)             0.062m · 97.5 % · t_reach 2

  The *current* best artifacts, on the fixed trunk (see the pitfall below):
                                                   baseline  imitation  final
  `robot_json_red.pt` (tiny, red reach, 10 rounds) 0.160m    0.103m     0.103m
                                                            0 %        0 %
                                                   60-frame bridge replay: 90 % in reach
  `robot_qwen_live.pt` (Qwen2.5-0.5B, red reach)   0.519m    0.034m     0.036m
                                                            95.0 %     95.0 %
  ceiling (expert rule, same protocol)             0.062m · 97.5 % · t_reach 2
  The Qwen run beats the expert's own rule (0.036m vs 0.062m) at 95 % of the
  expert's reach rate.  (That run also produced words off the LM head through
  the old instruction-region narration; speech is now a separate sentence
  generated AFTER the state — see "Tasks and speech" below — and
  `--untie-lm-head` still gives the speech loss a trainable head.)
  PITFALL (fixed): `--model tiny` used to build a DIFFERENT random trunk in every
  process — `build_tiny_trunk` drew from the global torch RNG and nothing seeded
  it — so the same checkpoint scored 80 %, 25 % or 1.7 % on identical commands and
  two "identical" runs generated different first tokens.  `train_loop_robot.py`
  now pins it (`TINY_TRUNK_SEED` + `fork_rng` around `AutoModel.from_config`);
  three identical CPU replays then agree exactly.  Any tiny-trunk checkpoint
  trained before that fix has adapters fitted to a trunk it will never see again
  and must be retrained (hub weights such as Qwen were never affected).
  The *imitation* is the stage that learns: three rounds of expert-labelled live
  states turn a constant `-x` policy (0.519m · 0 % reach · 0 % expert agreement)
  into a tracker at 0.052–0.095m, the best of which beats the expert's own rule
  (0.052m vs 0.062m under the same protocol).  The A2C stage as configured is
  net-negative or neutral after it — the entropy bonus pulls the policy back
  toward uniform and lr=3e-4 is coarse for a policy that already tracks — and it
  never improves on the imitation result, although its *best snapshot* does
  (0.073m); that is why every save also keeps `<out>_best.pt` and the summary
  prints which file to replay.  So `--frames 1 --imitate-rounds 3` is the recipe
  that matters.
  The live stream (moving spheres) reads better than the protocol eval — reach
  38–50 % and d 0.06–0.08m in the last 50 frames of the third run, against 0 %
  from the ready pose with the spheres parked at their MJCF start poses — the
  protocol is the harder, repeatable number.
  Gate note: the two `_sup4.pt` runs are the first ever to train with a *live*
  gated cross-attention path (`fusion.gate` 0.171 → 0.204 across the stages, so
  the gate does learn).  In every earlier checkpoint it sat at exactly 0.0
  because of a zero-init deadlock in `train_loop_robot.py` (fixed — see the
  `GatedSensorFusion` note there).  It does not move the tracking number much
  (0.052m dead-gate vs 0.076m live-gate imitation, run-to-run spread), but the
  module is now training instead of being silently inert.

JSON decision interface (inherited from `train_loop_robot.py`)
-------------------------------------------------------------
  `TrainablePolicy` builds the typed-question bank straight from the checkpoint
  (`questions` is adopted by `meb.load_checkpoint`) and REQUIRES one that can
  decode a motor token (`tlr.has_json_action`); the DAgger stage trains the
  answers with `joint_loss` = question CE (labels from `derive_question_labels`
  on the expert action) + tau·ponder (+ task, + speech), and `--json-out` writes
  the live decision document each frame — typed answers, the derived action,
  the top-level `say` (null when silent) and the executed command.  The answer
  options are Jev labels (A/B/C…, 0/1/2…) read off the LM head; the document
  reports the original option names.

JSON control (always — the action head was removed)
---------------------------------------------------
  The arm follows the *questionnaire*: each question is sampled from its own
  categorical, the document is rendered with those labels,
  `meb.command_from_json()` binds it to a `Command` (direction / speed scale /
  jaw / hold / brake) and the same torque map is applied.  Two consequences for
  the RL math:
    · logπ(a|s) = Σ_q log P_q(label_q) — a product over the bank, so both the
      policy term and the entropy bonus are sums over the questions (see
      `A2CUpdater._json_logprob`); the buffer stores the sampled label vector,
      so the pair (state, labels) the update differentiates is exactly the one
      the rollout acted on.
    · the gradient flows into the answer read-out (LM-head label rows, the
      `task_proj` instruction shortcut added to every h_ans, fusion, LoRA).
      The 6-way token is `tlr.json_action(labels)` — telemetry and the expert
      comparison, never a separate head.  `drive()` is the single place the
      policy's decision reaches the actuators, so the bridge and this loop
      cannot diverge; the expert and ceiling paths call `env.command()`
      directly, which keeps the eval protocol comparable.
  `--control action` is refused with a clear error (there is no token head
  left to drive the arm).

Tasks and speech (`--task`, `--speak-max`)
------------------------------------------
  `--task {any,reach,grasp,throw,push}` rotates the *instruction* across DAgger
  rounds (`meb.MujocoArm.new_instruction`), and the expert that labels the live
  states is task-conditioned (`expert_action(obs, target, task)`): `reach` brakes
  on the sphere, `grasp` holds it, `throw` carries it off, `push` drives straight
  through and never closes the jaw.  Nothing in the architecture changes — the
  verb is text, which is exactly the property that makes the JSON interface
  portable.  Speech follows the state: the LM head at the last position of the
  last state step predicts the first speech token from the SAME forward that
  produced the answers; END means silent (zero extra cost), anything else is
  greedy-decoded up to `--speak-max` tokens (default 8, 0 mutes) and lands in
  `--json-out` as the top-level `say`.  With `--text-weight > 0` the imitation
  pool is labelled by the same event detector as the synthetic generator
  (`tlr.detect_speech_event`: grasped / released / reached / pushing the
  <colour> block, silent otherwise) and the fit adds the teacher-forced speech CE.
  Measured (tiny trunk, per-task bridge replay, 40 frames, goal=red): the
  supervised-only `_mt2.pt` sits at 0.0 % in reach / last 0.602 m under *every*
  verb — supervision binds the verb in the synthetic world, not on live states.
  Five DAgger rounds over `--task any --goal any --retarget-every 100` (2000 live
  frames, 5 × 250 fits, `--imitate-lr 1e-4 --q-weight 2.0`) move it to
  reach 5.0 % / 0.105 m · grasp 5.0 % / 0.112 m · throw 17.5 % / 0.084 m ·
  push 5.0 % / 0.104 m, and the protocol eval from 0.486 m (baseline) to 0.146 m
  (ceiling on that instruction: 0.074 m / 80 %).

The harness guard (`--unstuck`, measured)
-----------------------------------------
  A live-only failure mode: both axis answers come back `stay`, the binding
  brakes, and a *stationary* arm makes `stay` more likely again (`vel ≈ 0` and
  `prev_action = brake` both vote for it) — a self-reinforcing deadlock that
  never occurs in the synthetic world, where `vel = 0.85·vel + 0.30·ACTION_VECS[a]`
  never reaches exactly zero.  Measured on `robot_live_json.pt`, 60 frames:
  `x0·y0·spd0.51` from frame 10 to 60 with `goal` frozen at 0.077 m → 1.7 % of
  frames in reach; the same weights under the (since removed) token control steered normally
  (36.7 %).  `meb.unstuck_guard()` therefore counts frames in which the answers
  brake with no progress and, after `--unstuck` (default 8) of them, issues a
  harness-steered `Command` for `--unstuck-rescue` (default 6) frames: 1.7 % →
  21.7–23.3 % in reach, goal trace alive (0.160 → 0.062 → 0.010 → 0.064 m), 4 of
  60 frames steered.  It only guards the JSON control, and it is exactly the
  Jev-as-Policy split: the model classifies, the harness owns the geometry.

A2C alone (measured *before* the diagnosis below, kept as the contrast case):
      d_mean  0.653m (baseline) → 0.436 → 0.436 → 0.436 → 0.278 → 0.278 →
              0.278m (final)                              Δ −0.376m (−58 %)
      live stream agrees: return per 32-frame rollout +1.97 → +2.16, and the
      action mix walks from −x:0.23/+x:0.19 to −x:0.12/+x:0.30
  Two plateaus, three evals each — the quantisation of a policy that is still
  effectively constant: six discrete actions mean the greedy trajectory stops
  changing until a whole action flips.  With the *earlier* (open-loop torque)
  bridge the same recipe read 0.301 → 0.215 m, but that number was meaningless:
  on that actuator map a constant brake scored 0.426 m and the expert itself
  only 0.314 m.

Why 1500 frames of A2C plateaued (measured — the reason `LiveImitator` exists)
-----------------------------------------------------------------------------
Every eval also reports the fraction of its greedy decisions that match the
expert rule, which turns "the policy is bad" into a diagnosis:
      `_sup2.pt`    (600 supervised steps)          d_mean 0.519m · `-x` on all
                    40 frames · 0 switches · expert agreement  0.0 %
      `_sup3.pt`    (2400 steps, 0.67 synthetic acc) 0.519m · `-x` · 0 · 0.0 %
      `_rl_park.pt` (1200 RL frames, fixed reward)   0.436m · `+y` · 0 · 2.5 %
      untrained tiny adapter (random init)           0.257m · `+x` 0.42 /
                    `+y` 0.50 · 36 switches · 7.5 %
  Chance is 16.7 % (one of six actions).  The trained checkpoints score *below*
  chance because they emit a single constant action, while those same weights
  reach 0.67 action accuracy on the synthetic eval — the weights are intact and
  the failure is pure distribution shift between `SensorWorld` windows and live
  states (live values sit in a corner of the training ranges: |pos| ≤ 0.75 vs
  ±1.2, |rel| ≤ 0.6 vs ±2.4).  A constant policy cannot be repaired by A2C — the
  action a live state needs is never sampled, so its advantage is never seen —
  which is why the expert labels live states first (`--imitate-*`) and A2C only
  starts once the policy already tracks.

Only the `requires_grad` adapter (+ the small critic) is updated — the 0.5B trunk
stays frozen and resident, and every VRAM guardrail from `train_loop_robot.py`
(allocator cap, 8 GiB budget, `MemoryGuard`, OOM skip) is inherited untouched.

Usage
-----
  # the measured pipeline: synthetic warm start → DAgger round 0 → A2C
  python train_loop_robot.py --model tiny --steps 2400 --batch 4 --window 8 \
      --trainable adapters --eval-every 800 --eval-batches 4 --out robot_act.pt
  python train_live_mujoco.py --model tiny --device cuda --checkpoint robot_act.pt \
      --trainable adapters --goal red --retarget-every 0 --window 8 \
      --imitate-rounds 3 --frames 1500 --eval-every 250 --rollout 32 \
      --update-batch 8 --lr 3e-4 --out robot_live.pt

  # skip the imitation stage entirely (A2C only — the behaviour measured above)
  python train_live_mujoco.py --model tiny --imitate-rounds 0 --frames 1500

  # watch a fresh adapter learn to chase the spheres (cold start, tiny trunk)
  python train_live_mujoco.py --model tiny --frames 1500 --trainable recurrent

  # real policy: warm-start from the supervised adapter, 6.87M trainable weights
  python train_live_mujoco.py --trainable recurrent \
      --checkpoint robot_act.pt --out robot_live.pt --frames 4000

  # headless, snapshot the learning progress every 200 frames
  python train_live_mujoco.py --model tiny --headless \
      --render-out rl_frame.png --render-every 200 --frames 1200

  # verify the RL math (reward signs, γ-returns, PG direction, critic fit,
  # gradient flow, carry contract, eval protocol, payload round-trip)
  python train_live_mujoco.py --model tiny --self-test

The artifact written by `--out` is the same payload `train_loop_robot.py` emits
(`trainable_state` + architecture metadata) plus the critic, so
`mujoco_env_bridge.py --checkpoint robot_live.pt` plays it back unchanged.

Read the policy against its ceiling, not against zero
-----------------------------------------------------
The same file measures what the interface can do at all: `EvalProbe.expert_ceiling`
runs the rule that *labelled the supervised data*
(`train_loop_robot.SensorWorld.episode`) through the identical protocol
(`--goal` pins the colour).  This is not decoration — it is how the real
blocker was found.  The bridge originally wrote one torque and held it across
all 100 physics substeps, so a saturated command accelerated the arm for the
whole 0.2 s decision window: 34.8 cm of tool travel for an 8 cm velocity pulse,
|v| = 2.71 m/s, 100 % saturation, and on that interface the expert scored
*d_worse than doing nothing* (0.314 m vs 0.426 m for a constant brake).  Every
number the RL loop produced was being read against a broken actuator map.
After the per-substep servo, the expert reaches 0.057-0.062 m / 87-97 % on red,
which is the bar the policy now has to close (printed as `vs ceiling` at the end
of every run).

One more trap the same measurement exposed: the policy and the critic are
optimised with different objectives (a log-prob and a squared return), so their
raw gradient norms differ by ~10× — 46 vs ~1 at the start of a run.  A single
`clip_grad_norm_` over both parameter groups therefore rescales both by the
larger norm, and the one number in the log says nothing about which half of the
loss is moving.  The two groups are now normed, clipped and printed separately
(`gn_pol`/`gn_val`) — honest instrumentation of a shared optimiser, not a fix
for learning: with AdamW the *magnitude* of a uniform rescale cancels in
`m̂/(√v̂+ε)`, so a joint clip was already close to a no-op.  It is the number
that had to be split, not the step.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# IMPORT ORDER IS LOAD-BEARING: `train_loop_robot` installs the allocator /
# MIOpen / HSA guards *before* torch is imported, and `mujoco_env_bridge`
# supplies the physics scene, the 28-D contract and the policy plumbing we train.
# ---------------------------------------------------------------------------
import train_loop_robot as tlr                                       # noqa: E402

import torch                                                         # noqa: E402
import torch.nn as nn                                               # noqa: E402
import torch.nn.functional as F                                      # noqa: E402

import mujoco                                                        # noqa: E402
import mujoco_env_bridge as meb                                      # noqa: E402

try:                                    # a viewer needs a real GL context
    import mujoco.viewer                                            # noqa: E402
    _VIEWER_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:                                            # pragma: no cover
    _VIEWER_IMPORT_ERROR = exc


# =============================================================================
# 1 · REWARD CONTRACT  (what "the arm is tracking the spheres" means, in joules)
# =============================================================================
@dataclass(frozen=True)
class RewardSpec:
    """
    Per-frame reward.  All distances are in **metres**, straight from the
    bridge's `goal` telemetry = ‖object_xy[target] − tool_site_xy‖.

        r = w_progress · (d_prev − d_now)            ← the actual task signal
          + w_dist     · (d_prev − γ·d_now)          ← NG-1999 potential shaping
          + reach_bonus· 1[d_now < reach]            ← crossed the "graspable" line
          + hold_bonus · 1[was in reach ∧ still in reach]
          − ponder_penalty · ponder                  ← ACT compute cost (as in
                                                       train_loop_robot's τ·ponder)
        (clipped to ±reward_clip so a physics glitch cannot blow up a γ-return)

    The shaping term is deliberately **not** `(1 − d/dist_ref)`.  Any term that
    pays a positive amount per frame just for existing in a state is an income
    stream, and a finite reach bonus can lose to it: with `1 − d/0.5m` a policy
    parked at 0.28 m collected +0.44 every frame forever, so the *greedy*
    optimum was "stop at a comfortable distance" — which is exactly the static
    action this file used to produce.  Shaping as a γ-consistent potential
    difference F = γΦ(s') − Φ(s) with Φ(s) = −d(s) telescopes over any path to
    (d_start − d_end), so it cannot change which policy is optimal (Ng, Harada
    & Russell 1999); it only makes the gradient dense.  Standing still then pays
    (1 − γ)·d ≈ 0.3 % of d, and arrival — the bonus plus the hold — dominates.
    """
    w_progress: float
    w_dist: float
    reach_bonus: float
    hold_bonus: float
    ponder_penalty: float
    gamma: float
    clip: float

    @classmethod
    def from_cfg(cls, cfg: argparse.Namespace) -> "RewardSpec":
        return cls(w_progress=cfg.w_progress, w_dist=cfg.w_dist,
                   reach_bonus=cfg.reach_bonus, hold_bonus=cfg.hold_bonus,
                   ponder_penalty=cfg.ponder_penalty, gamma=cfg.gamma,
                   clip=cfg.reward_clip)

    def table(self) -> str:
        return (f"  reward      : w_progress={self.w_progress:g}/m · "
                f"w_dist={self.w_dist:g}·(d_prev−γ·d_now) · "
                f"reach_bonus={self.reach_bonus:g} · hold={self.hold_bonus:g} · "
                f"ponder_penalty={self.ponder_penalty:g} · clip=±{self.clip:g}")


def compute_reward(spec: RewardSpec, d_prev: float, d_now: float, reach: float,
                   was_in_reach: bool, ponder: float
                   ) -> Tuple[float, Dict[str, float]]:
    """One reward from the pre-action distance and the post-action distance."""
    progress = spec.w_progress * (d_prev - d_now)
    shaping = spec.w_dist * (d_prev - spec.gamma * d_now)
    bonus = spec.reach_bonus if d_now < reach else 0.0
    hold = spec.hold_bonus if (was_in_reach and d_now < reach) else 0.0
    cost = spec.ponder_penalty * ponder
    raw = progress + shaping + bonus + hold - cost
    r = float(np.clip(raw, -spec.clip, spec.clip))
    return r, {"progress": progress, "shaping": shaping, "bonus": bonus,
               "hold": hold, "cost": cost, "raw": raw}


def a2c_loss(logprob: torch.Tensor, entropy: torch.Tensor, value: torch.Tensor,
             advantage: torch.Tensor, ret: torch.Tensor,
             ent_coef: float, value_coef: float
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Advantage actor-critic loss over a batch of decisions:

        A     = G_t − V(s_t)                       (normalised by the caller)
        L_pg  = −(A · logπ(a_t|s_t) + β·H(π))      (REINFORCE + entropy)
        L_v   = value_coef · (V(s_t) − G_t)²       (baseline regression)
        L     = mean(L_pg) + L_v

    `A · logπ` is the policy gradient: with A > 0 the log-prob of the sampled
    action is *increased*, with A < 0 it is decreased.  The entropy term is what
    keeps the head from collapsing onto a single token (`-y` forever).

    `advantage` and `ret` are separate arguments on purpose: the *normalised*
    advantage multiplies logπ (variance reduction), while the critic regresses
    the **raw** discounted return — tying the two would silently retarget the
    baseline.  Everything is flattened to (N,), and the value runs in fp32 (the
    critic's own contract), so nothing broadcasts by accident.
    """
    v = value.reshape(-1).float()                       # critic precision = fp32
    a = advantage.reshape(-1).float()
    g = ret.reshape(-1).float()
    pg = -(a.detach() * logprob.reshape(-1) + ent_coef * entropy.reshape(-1))
    vf = value_coef * F.mse_loss(v, g)
    return (pg + vf).mean(), pg.detach().mean(), vf, float(a.detach().mean())


def _num(v) -> float:
    """`stats`/`cycles` come back as tensors OR floats depending on the model
    build — one accessor so telemetry never drags the autograd graph along."""
    return float(v.detach()) if isinstance(v, torch.Tensor) else float(v)


# =============================================================================
# 2 · CRITIC  (V(s): the loop's mixed latent → a scalar baseline)
# =============================================================================
class Critic(nn.Module):
    """
    `LoopOutput.carry` is the model's own state summary — the weighted mix of
    every ACT cycle, already `.detach()`ed, so this head trains on its own and
    can never disturb the trunk graph.  LayerNorm makes it scale-free across
    Qwen-0.5B (H=896) and the tiny trunk (H=64); the last layer starts at zero,
    so V ≡ 0 at step 0 and early advantages are pure returns.
    """

    def __init__(self, hidden: int, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width),
                                 nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


# =============================================================================
# 3 · TRAINABLE POLICY  (the bridge's PolicyRunner, with gradients + a critic)
# =============================================================================
@dataclass
class ActionOut:
    """One live decision.  Plain numbers on purpose: the rollout runs under
    `no_grad`, and the graph is built later, inside `A2CUpdater.update()`.

    The arm is always driven by the typed answers: `action` is the index
    `tlr.json_action` decodes from the (sampled) labels — telemetry and the
    expert-agreement metric, never a separate head — `labels` holds the sampled
    question labels (what `update()` recomputes logπ from), `cmd` is the torque
    Command the harness derived from them and `say` the frame's utterance
    (None = silent).
    """
    action: int
    logprob: float                   # logπ(a_t|s_t) at sampling time (drift check)
    value: float                     # V(s_t), fp32, detached
    probs: np.ndarray                # (A,) softmax, or the per-question concat
    cycles: float
    ponder: float
    halted_frac: float
    mass_err: float
    latent_norm: float
    labels: Dict[str, int] = field(default_factory=dict)   # sampled answers
    cmd: Optional[Any] = None                              # bound motor Command
    say: Optional[str] = None                              # utterance, None = silent


def drive(env, ao: ActionOut) -> None:
    """
    Send one *policy* decision to the actuators: the `Command` the
    questionnaire was bound to (`ao.cmd`).  The expert / ceiling paths keep
    calling `env.command()` with the 6-way token directly, so the comparison
    protocol is unaffected; a decision without a Command (never produced by the
    policy) falls back to that same token table.
    """
    if ao.cmd is not None:
        env.set_command(ao.cmd)
    else:
        env.command(ao.action)
    env.note_action(ao.action)


# A policy window is a PAIR of token tensors now, not a float frame buffer:
#   sids (1, S·L) long   — the last S state sentences, flattened (`render_state`)
#   apos (1, S, Q) long  — the answer-marker position of each question, per step
# `meb.StateWindow` builds both from the harness frame; nothing here is numeric
# state any more — the model reads the sentence.
Window = Tuple[torch.Tensor, torch.Tensor]


class TrainablePolicy(meb.PolicyRunner):
    """
    Same model, same text-state contract, same never-reset carry as the
    inference runner — plus a value head, and an `act()` that samples without a
    graph.  The trunk stays frozen: only the `requires_grad` adapters are
    optimised.
    """

    def __init__(self, cfg: argparse.Namespace, accel, tok, trunk,
                 payload: Optional[Dict[str, Any]] = None):
        super().__init__(cfg, accel, tok, trunk, payload)
        self.model.train()                     # the parent leaves it in eval()
        # The bridge builds the loop with grad_checkpoint=False (inference never
        # needs it); RL does — it is the knob that keeps the batched backward
        # inside the 8 GiB carve-out.
        self.model.grad_checkpoint = bool(cfg.grad_checkpoint)
        self.hidden = int(getattr(trunk.config, "hidden_size"))
        self.critic = Critic(self.hidden, cfg.critic_hidden).to(accel.device).to(torch.float32)
        # --freeze-lm-head: the untied LM head (~136M rows on Qwen2.5-0.5B) is
        # the answer read-out; fitting it on a small single-task live pool
        # memorises that pool (CE falls while rollout agreement drops), so the
        # live stage can keep it at its supervised values.
        if bool(getattr(cfg, "freeze_lm_head", False)) and \
                getattr(self.model, "lm_head", None) is not None:
            self.model.lm_head.requires_grad_(False)
        self.trainable: List[nn.Parameter] = [p for p in self.model.parameters()
                                              if p.requires_grad]
        self.n_trainable = sum(p.numel() for p in self.trainable)
        self.n_critic = sum(p.numel() for p in self.critic.parameters())
        # Control is ALWAYS the questionnaire (the action head was removed);
        # the parent already refused a bank without json_action questions.
        ctrl = str(getattr(cfg, "control", "json"))
        if ctrl not in ("auto", "json"):
            raise SystemExit(f"--control {ctrl} is no longer supported: the action "
                             "head was removed and the arm is always driven by "
                             "the JSON answers (drop the flag or pass --control json)")
        self.control = "json"
        self.qids: List[str] = [s.qid for s in self.bank.specs]
        self.last_entropy = 0.0
        self.last_json: Optional[Dict[str, Any]] = None

    # ── one autocast forward of the whole loop ──────────────────────────────
    def forward_out(self, sids: torch.Tensor, apos: torch.Tensor,
                    carry: Optional[torch.Tensor],
                    instr_ids: Optional[torch.Tensor] = None,
                    instr_mask: Optional[torch.Tensor] = None,
                    speech_ids: Optional[torch.Tensor] = None,
                    speech_mask: Optional[torch.Tensor] = None):
        """`sids`/`apos` are (B,S·L)/(B,S,Q) — B=1 live, B=chunk for an update.

        The instruction is one prompt for the whole rollout, so it is broadcast
        with `expand` (no copy) instead of being re-encoded per frame; a
        retarget flushes the rollout, so the goal can never differ inside it.
        `instr_ids`/`instr_mask` override it with a (B,Ti) batch — the imitation
        pool stores the prompt that was live when each frame was collected, so a
        retarget inside a round cannot relabel earlier frames with the wrong
        task (the verb is part of the label: `throw` holds, `push` never does).
        `speech_ids`/`speech_mask` append teacher-forced speech AFTER the state
        (imitation with --text-weight); causal, so no answer can see them.
        """
        b = sids.shape[0]
        if instr_ids is None:
            instr_ids = self.instr_ids.expand(b, -1)
            instr_mask = self.instr_mask.expand(b, -1)
        with torch.amp.autocast(device_type=self.accel.amp_device_type,
                                dtype=self.accel.dtype, enabled=self.accel.amp):
            return self.model(instr_ids, instr_mask, sids, apos, carry,
                              speech_ids=speech_ids, speech_mask=speech_mask)

    def _json_sample(self, out, sample: bool) -> Tuple[Dict[str, int], float, float]:
        """
        Draw one label per question.  The JSON policy is a *product* over the
        bank, so logπ = Σ_q log P_q(label_q) and the entropy is the sum of the
        per-question entropies — that is what the A2C update differentiates.
        A `noul` question has one logit, so it is a Bernoulli, not a degenerate
        Categorical.
        """
        labels: Dict[str, int] = {}
        lp, ent = 0.0, 0.0
        for spec in self.bank.specs:
            lg = out.q_logits[spec.qid][0, -1].float()
            if spec.n_out == 1:
                p = torch.sigmoid(lg[0])
                k = int(torch.bernoulli(p)) if sample else int(float(p) >= 0.5)
                lp += float(torch.log(p if k else 1.0 - p).clamp_min(-30.0))
                ent += float(-(p * torch.log(p.clamp_min(1e-9)) +
                               (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-9))))
            else:
                dist = torch.distributions.Categorical(logits=lg)
                k = int(dist.sample()) if sample else int(torch.argmax(lg))
                lp += float(dist.log_prob(torch.tensor(k, device=lg.device)))
                ent += float(dist.entropy())
            labels[spec.qid] = k
        return labels, lp, ent

    def _json_out(self, out, sample: bool, sids: torch.Tensor, apos: torch.Tensor,
                  carry_in: Optional[torch.Tensor]) -> ActionOut:
        """
        The Jev-style decision: sample the typed answers, render the document
        from the SAME forward, bind it to a Command, and act on that.  The
        document is written with `q_labels=` so it reports the labels that were
        actually executed (not the arg-maxes), which keeps the stored
        logπ/labels pair exactly on-policy for the update.  Speech is STREAMED:
        the forward carried `self.spoken` after the state, its last position
        gives this frame's first new token (END first = silent, no extra
        cost), `--speak-tokens 2` adds one extra forward; the finished
        sentence lands in the document's `say` on its completing frame.
        """
        labels, lp, ent = self._json_sample(out, sample)
        say, _partial, _n = self.speak_after(out, sids, apos, carry_in)
        if say is not None:
            self.say = say
        doc = self.model.answers_json(out, q_labels=labels, say=say)
        cmd = meb.command_from_json(doc, self.cfg)
        doc["executed"] = {"action": tlr.ACTION_NAMES[cmd.action],
                           "answers": cmd.note}
        self.last_json = doc
        self.last_entropy = ent
        probs = np.concatenate([
            torch.softmax(out.q_logits[s.qid][0, -1].float(), -1).cpu().numpy()
            if s.n_out > 1 else np.array(
                [1.0 - float(torch.sigmoid(out.q_logits[s.qid][0, -1, 0])),
                 float(torch.sigmoid(out.q_logits[s.qid][0, -1, 0]))], np.float32)
            for s in self.bank.specs]).astype(np.float32, copy=False)
        return ActionOut(
            action=tlr.json_action_one(labels), logprob=lp,
            value=float(self.critic(out.carry[0].float())),
            probs=probs,
            cycles=_num(out.cycles[0, -1]), ponder=_num(out.ponder[0, -1]),
            halted_frac=_num(out.stats["halted_frac"]),
            mass_err=_num(out.stats["mass_err"]),
            latent_norm=_num(out.stats["latent_norm"]),
            labels=labels, cmd=cmd, say=say)

    @torch.no_grad()
    def act(self, sids: torch.Tensor,
            apos: torch.Tensor) -> Tuple[ActionOut, Optional[torch.Tensor]]:
        """
        Sample the typed answers **without** building a graph; returns
        `(decision, carry_in)` where `carry_in` is the latent this frame was
        fed.  The caller stores it next to the window, so the batched update
        reproduces this exact forward.
        """
        self.frames += 1
        carry_in = self.carry if self.cfg.carry else None
        sp, spm = self.speech_input()              # utterance streamed so far
        out = self.forward_out(sids, apos, carry_in, speech_ids=sp, speech_mask=spm)
        self.carry = out.carry if self.cfg.carry else None       # never reset
        return self._json_out(out, True, sids, apos, carry_in), carry_in

    @torch.no_grad()
    def greedy(self, sids: torch.Tensor, apos: torch.Tensor) -> ActionOut:
        """arg-max answers — the baseline / best-policy eval path (no exploration)."""
        self.frames += 1
        carry_in = self.carry if self.cfg.carry else None
        sp, spm = self.speech_input()
        out = self.forward_out(sids, apos, carry_in, speech_ids=sp, speech_mask=spm)
        self.carry = out.carry if self.cfg.carry else None
        return self._json_out(out, False, sids, apos, carry_in)

    def reset_carry(self) -> None:
        """Only for the `--reset-carry-on-episode` ablation (off by default)."""
        self.carry = None


# =============================================================================
# 4 · ROLLOUT BUFFER  (γ-returns → ONE batched, single-shot A2C step)
# =============================================================================
class RolloutBuffer:
    """
    Preallocated per-frame storage for one rollout (`--rollout` frames).

    Every row is written with `copy_` from device tensors already in hand, so the
    hot loop allocates nothing; the whole buffer is a handful of small device
    tensors ((F,S·L) state tokens + (F,S,Q) marker positions + (F,H) carry +
    six (F,) vectors).

    Stored per frame t: the state-token window that was fed in, the *input* carry
    (frame t was forwarded with the latent produced by frame t−1), the sampled
    action, the reward, the episode-end flag, and the rollout's own logπ/V(s_t).
    `present[t]` records whether the rollout actually had a carry at frame t —
    the very first frame of a run does not, and the update has to reproduce that
    (feeding it a zero latent would insert a memory token and shift the readout).
    """

    def __init__(self, size: int, steps: int, tokens: int, q: int, hidden: int,
                 device: torch.device, qids: Tuple[str, ...] = ()):
        self.size = int(size)
        self.qids = tuple(qids)
        self.n_q = len(self.qids)
        self.sids = torch.zeros((self.size, steps * tokens), dtype=torch.long,
                                device=device)
        self.apos = torch.zeros((self.size, steps, max(1, q)), dtype=torch.long,
                                device=device)
        self.carry = torch.zeros((self.size, hidden), device=device)
        self.action = torch.zeros((self.size,), dtype=torch.long, device=device)
        # JSON control: the sampled label per question — logπ is recomputed from
        # these in the update, so the pair (state, labels) is exactly on-policy
        self.labels = torch.zeros((self.size, self.n_q), dtype=torch.long,
                                  device=device)
        self.reward = torch.zeros((self.size,), device=device)
        self.done = torch.zeros((self.size,), dtype=torch.bool, device=device)
        self.logprob = torch.zeros((self.size,), device=device)
        self.value = torch.zeros((self.size,), device=device)
        self.present = torch.zeros((self.size,), dtype=torch.bool, device=device)
        self.n = 0

    @property
    def full(self) -> bool:
        return self.n >= self.size

    def reset(self) -> None:
        self.n = 0

    def add(self, sids: torch.Tensor, apos: torch.Tensor,
            carry: Optional[torch.Tensor], action: int,
            reward: float, done: bool, logprob: float, value: float,
            labels: Optional[Dict[str, int]] = None) -> None:
        i = self.n
        self.sids[i].copy_(sids[0])                        # (S·L) ← (1,S·L)
        self.apos[i].copy_(apos[0])                        # (S,Q) ← (1,S,Q)
        # a frame can legitimately have no carry yet (the first frame of a run,
        # or the first after a reset in the ablation): record that, so the update
        # can forward it with exactly the context the rollout used
        self.present[i] = carry is not None
        if carry is None:
            self.carry[i].zero_()
        else:
            self.carry[i].copy_(carry[0])
        self.action[i] = int(action)
        if labels:
            for j, qid in enumerate(self.qids):            # element-wise: no alloc
                self.labels[i, j] = int(labels[qid])
        self.reward[i] = float(reward)
        self.done[i] = bool(done)
        self.logprob[i] = float(logprob)
        self.value[i] = float(value)
        self.n = i + 1

    def returns(self, gamma: float, boot: float) -> torch.Tensor:
        """
        G_t = r_t + γ·G_{t+1}, cut at every episode end and seeded with `boot`
        (= V of the frame that follows the rollout).

        `done[t]` means the *next* state of transition t belongs to a fresh
        episode (the arm was reset), so no value may leak across it; a retarget
        is not an episode end — the state keeps flowing.  Done on the host over
        ≤ `--rollout` floats: the recurrence stays explicit and hand-checkable
        (the self-test does exactly that).
        """
        r = self.reward[:self.n].tolist()
        d = self.done[:self.n].tolist()
        out = [0.0] * len(r)
        acc = 0.0 if (d and d[-1]) else float(boot)
        for t in range(len(r) - 1, -1, -1):
            acc = r[t] + (0.0 if d[t] else gamma * acc)
            out[t] = acc
        return torch.tensor(out, device=self.sids.device)


class A2CUpdater:
    """
    Collects a rollout, then runs exactly ONE batched forward+backward and ONE
    AdamW step per rollout.

    The graph is created and consumed inside `update()` — nothing survives the
    call — so the optimizer can never invalidate a live graph (that in-place
    version clash is what killed the per-frame draft: the step for frames 1..k
    ran while frame k+1's graph was still alive).  `--update-batch` chunks the
    backward so peak VRAM is set by the chunk, not by the rollout length;
    gradients accumulate across chunks, are clipped once and stepped once, which
    for equal-size chunks is exactly a single big batch (mean of means =
    overall mean).
    """

    def __init__(self, cfg: argparse.Namespace, policy: TrainablePolicy, accel,
                 win: "meb.StateWindow"):
        self.cfg, self.policy, self.accel = cfg, policy, accel
        # the window's token geometry (S·L per frame, Q markers) comes from the
        # live StateWindow, so the buffer is shaped by the same renderer the
        # policy reads
        self.buf = RolloutBuffer(cfg.rollout, cfg.window, win.L, win.q,
                                 policy.hidden, accel.device,
                                 qids=tuple(policy.qids))
        self.opt = torch.optim.AdamW(
            [{"params": policy.trainable, "lr": cfg.lr},
             {"params": list(policy.critic.parameters()), "lr": cfg.value_lr}],
            betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
        self.scaler = None                          # fp16 only — bf16 needs none
        if accel.amp and accel.scaler:
            try:
                self.scaler = torch.amp.GradScaler(accel.amp_device_type)
            except (TypeError, ValueError):          # older torch
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.guard = tlr.MemoryGuard(accel, gc_every=cfg.gc_every)
        self.due = False                            # buffer full → update next frame
        self.updates = 0
        self.oom_events = 0
        self.last: Dict[str, float] = {}

    # ── the one update ──────────────────────────────────────────────────────
    def _json_logprob(self, out, sl: slice, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logπ and H of the JSON policy: a product over the bank, so both are sums
        over the questions of the label that was actually executed.  `noul`
        questions are Bernoulli (one logit), the rest Categorical.
        """
        dev = out.carry.device
        lp = torch.zeros(m, device=dev, dtype=torch.float32)
        ent = torch.zeros(m, device=dev, dtype=torch.float32)
        for j, spec in enumerate(self.policy.bank.specs):
            lg = out.q_logits[spec.qid][:, -1, :].float()
            lab = self.buf.labels[sl, j]
            if spec.n_out == 1:
                lp = lp + F.logsigmoid(lg[:, 0] * (2.0 * lab.float() - 1.0))
                ent = ent + torch.distributions.Bernoulli(logits=lg[:, 0]).entropy()
            else:
                dist = torch.distributions.Categorical(logits=lg)
                lp = lp + dist.log_prob(lab)
                ent = ent + dist.entropy()
        return lp, ent

    def update(self, boot: float = 0.0) -> Optional[Dict[str, float]]:
        """Forward the whole rollout, backprop the A2C loss, take one step."""
        n = self.buf.n
        if n < 1:
            return None
        cfg = self.cfg
        ret = self.buf.returns(cfg.gamma, boot)                 # (n,)
        adv = ret - self.buf.value[:n]                          # G_t − V(s_t)
        if cfg.adv_norm and n > 1:                              # PG term only
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        present = self.buf.present[:n].tolist()
        chunk = max(1, min(cfg.update_batch, n))
        # group consecutive frames by whether their rollout had a carry; a frame
        # without one must be forwarded *without* the memory token, or the
        # readout positions shift and the stored logπ would not match the update
        groups: List[Tuple[int, int, bool]] = []
        for i in range(n):
            if groups and groups[-1][2] == present[i]:
                groups[-1] = (groups[-1][0], i + 1, present[i])
            else:
                groups.append((i, i + 1, present[i]))
        self.opt.zero_grad(set_to_none=True)
        stats: Dict[str, float] = {}
        drift_sum, cyc_sum, pov_sum = 0.0, 0.0, 0.0
        for lo, hi, has_carry in groups:
            for clo in range(lo, hi, chunk):
                chi = min(clo + chunk, hi)
                sl = slice(clo, chi)
                m = chi - clo                                   # frames here
                with torch.amp.autocast(device_type=self.accel.amp_device_type,
                                        dtype=self.accel.dtype,
                                        enabled=self.accel.amp):
                    out = self.policy.forward_out(
                        self.buf.sids[sl], self.buf.apos[sl],
                        self.buf.carry[sl] if has_carry else None)
                    # logπ = Σ_q log P_q(sampled label_q) — the only policy
                    logprob, entropy = self._json_logprob(out, sl, m)
                    value = self.policy.critic(out.carry.float())     # (m,) fp32
                    loss, pg, vf, adv_m = a2c_loss(logprob, entropy, value,
                                                   adv[sl], ret[sl],
                                                   cfg.entropy_coef, cfg.value_coef)
                try:
                    # m/n: each group contributes its own share of the rollout mean
                    if self.scaler is None:
                        (loss * (m / n)).backward()
                    else:
                        self.scaler.scale(loss * (m / n)).backward()
                except RuntimeError as exc:
                    if not tlr.is_oom(exc):                          # a real bug
                        raise
                    self.oom_events += 1
                    self.guard.oom_events += 1
                    self.opt.zero_grad(set_to_none=True)
                    tlr.free_device_cache(self.accel)
                    print(f"    OOM during the rollout update — skipped "
                          f"({self.oom_events} so far, cache flushed; try a smaller "
                          f"--update-batch)", flush=True)
                    return None
                # the update re-runs the rollout forward: with no step in between
                # the log-probs must match to numerical noise (bf16 tiling)
                drift_sum += float((logprob.detach()
                                    - self.buf.logprob[sl]).abs().mean()) * m
                cyc_sum += _num(out.cycles[:, -1].detach().mean()) * m
                pov_sum += _num(out.ponder[:, -1].detach().mean()) * m
                stats = {"pg": float(pg), "vf": float(vf.detach()), "adv": adv_m,
                         "ret": float(ret[sl].mean()),
                         "entropy": _num(entropy.detach().mean())}
        stats.update({"drift": drift_sum / n, "cycles": cyc_sum / n,
                      "ponder": pov_sum / n})
        return self._step(stats)

    def _step(self, stats: Dict[str, float]) -> Dict[str, float]:
        pol = self.policy.trainable
        val = list(self.policy.critic.parameters())

        def _norm(ps) -> float:
            return math.sqrt(sum(float(p.grad.detach().pow(2).sum())
                                 for p in ps if p.grad is not None))

        # measured BEFORE clipping, and split: the critic regresses returns of
        # order 1-5 while the policy term is a log-prob, so their raw norms
        # differ by ~10× (measured 46 vs ~1 early in a run).  One joint clip
        # would rescale both by the larger and the single logged number would
        # say nothing about which half of the loss is moving.  Clip and report
        # the two groups against the same threshold — note AdamW is invariant to
        # a uniform rescale, so this is instrumentation, not a step-size fix.
        gn_pol, gn_val = _norm(pol), _norm(val)
        if self.scaler is None:
            torch.nn.utils.clip_grad_norm_(pol, self.cfg.clip)
            torch.nn.utils.clip_grad_norm_(val, self.cfg.clip)
            self.opt.step()
        else:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(pol, self.cfg.clip)
            torch.nn.utils.clip_grad_norm_(val, self.cfg.clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.updates += 1
        stats = dict(stats)
        stats["gn"] = max(gn_pol, gn_val)
        stats["gn_pol"] = gn_pol
        stats["gn_val"] = gn_val
        stats["updates"] = float(self.updates)
        self.last = stats
        return stats

    def flush(self, boot: float = 0.0) -> Optional[Dict[str, float]]:
        """Update whatever the buffer holds — used before an eval/checkpoint and
        at the end.  The tail is then bootstrapped on 0: a truncation, not a
        guess, and the only place a partial rollout ever reaches the optimiser."""
        info = self.update(boot) if self.buf.n else None
        self.buf.reset()
        self.due = False
        return info

    # ── first-update autotune, kept out of the timing log ───────────────────
    def warmup(self, sids: torch.Tensor, apos: torch.Tensor) -> float:
        t0 = time.perf_counter()
        ao, carry_in = self.policy.act(sids, apos)      # same forward as a live frame
        self.buf.reset()
        self.buf.add(sids, apos, carry_in, ao.action, 0.0, True, ao.logprob,
                     ao.value, labels=ao.labels)
        self.update(boot=0.0)
        self.buf.reset()
        self.opt.zero_grad(set_to_none=True)
        self.due = False
        self.updates = 0                                # autotune is not a training step
        if self.accel.backend in ("rocm", "cuda"):
            torch.cuda.synchronize(self.accel.device)
        return time.perf_counter() - t0


# =============================================================================
# 5 · EVAL PROBE + EXPERT CEILING  (the yardstick, and its own upper bound)
# =============================================================================
def expert_action(obs: np.ndarray, target: int, task: int = 0) -> int:
    """
    `train_loop_robot.SensorWorld.episode` (the expert that labels the training
    data) evaluated on the bridge's 28-D vector — same units, because the bridge
    scales its state INTO the training units.

        steer = (obj_t − tool)/d_t                              attraction
        steer += 0.35 · Σ_{o≠t} −(obj_o − tool)/d_o³ · [d_o < 0.35]   repulsion
        a      = argmax over the 4 axis moves of ⟨steer, action⟩  best direction
        d_t < 0.10 → grasp · d_t < 0.04 → brake                  terminal

    `task` is the verb from the instruction (`tlr.TASKS`), and it is the only
    thing that differs once the arm has arrived: `reach` brakes on the object,
    `grasp` holds it, `throw` carries it off (+y), `push` never stops at all.
    The jaw state comes from the observation itself (`SL_GRIP`), so the rule
    needs no hidden bookkeeping.

    Not a policy — a measurement.  Run through the SAME protocol as the policy
    it answers what the interface permits, so a policy's number can be read
    against its ceiling instead of against zero.
    """
    rel = obs[meb.SL_REL].reshape(meb.NUM_OBJECTS, 2).astype(np.float64)
    dist = np.maximum(obs[meb.SL_DIST].astype(np.float64), 1e-4)
    d = float(dist[target])
    steer = rel[target] / max(d, 1e-6)
    push = -(rel / dist[:, None]) / (dist[:, None] ** 2) * (dist < 0.35)[:, None]
    push[target] = 0.0
    steer = steer + 0.35 * push.sum(axis=0)
    vecs = np.asarray(tlr.ACTION_VECS, dtype=np.float64)
    a = int(np.argmax(steer @ vecs[:4].T))
    held = float(np.ravel(obs[meb.SL_GRIP])[0]) > 0.5   # jaw already on the object
    if held and task == 2:                         # throw: carry it off
        return 3
    if held and task == 1:                         # grasp: hold it still
        return 5
    if task == 0 and d < 0.04:
        return 5                                   # reach: at the object → brake
    if task != 3 and d < meb.TRAIN_REACH_UNITS:
        return 4                                   # inside reach → close the jaw
    return a                                       # push drives straight through


def expert_height(obs: np.ndarray, target: int, task: int = 0) -> int:
    """
    The third axis, mirroring the height rule in
    `train_loop_robot.SensorWorld.episode`:

        close (d_xy < 0.10) and tool above the target by Z_TOL  → down
        close and tool below the target by Z_TOL (nothing held) → up
        carrying an object for `throw`                          → up
        otherwise                                               → stay

    All three channels are in the observation (`SL_TOOLZ`, `SL_RELZ`), so the
    target's height is recovered exactly as `tool_z + rel_z[target]` — the same
    quantity the synthetic world computed.  This is the label the `height`
    question is trained on; the planar action table cannot express it.
    """
    tool_z = float(np.ravel(obs[meb.SL_TOOLZ])[0])
    rel_z = obs[meb.SL_RELZ].reshape(meb.NUM_OBJECTS).astype(np.float64)
    tgt_z = tool_z + float(rel_z[target])
    d = float(np.maximum(obs[meb.SL_DIST].astype(np.float64), 1e-4)[target])
    held = float(np.ravel(obs[meb.SL_GRIP])[0]) > 0.5
    if held and task == 2:                         # throw: lift what we carry
        return 2
    if d < meb.TRAIN_REACH_UNITS:
        if tool_z > tgt_z + tlr.Z_TOL:
            return 0                               # descend onto the object
        if tool_z < tgt_z - tlr.Z_TOL and not held:
            return 2                               # overshot: climb back
    return 1


# =============================================================================
class EvalProbe:
    """
    Greedy evaluation on a **fixed protocol**, so successive evals are actually
    comparable and the training trajectory is never disturbed.

    Why not just watch the live loop: the spheres are force-driven along a
    Lissajous trajectory, so the distance in the training stream is dominated by
    where the *target* drifted, not by what the policy did — that number goes up
    and down with unchanged weights.  The probe removes the confound: the arm
    returns to its ready pose, the spheres to their MJCF start poses, `sim_time`
    to 0 (identical Lissajous phase), the instruction stays the one in play, and
    `--eval-frames` greedy frames run with an empty carry and a fresh window.
    Same protocol every time ⇒ the only variable left is the policy.

    It is also non-invasive: MuJoCo's state, the policy's latent carry and the
    training window are snapshotted and put back, so running evals more often
    cannot change what the optimiser sees (MuJoCo's internal caches are merely
    recomputed).

        d_first   what the protocol always starts from (sanity: must never move)
        d_mean    mean distance over the protocol   ← the progress number
        d_best    minimum distance reached
        t_reach   first frame inside `--reach`, or `--` if it never got there
        reach%    share of frames inside reach

    The mapping also carries `goal` (the instructed colour).  The arm always
    starts from the same pose, but the *target* differs between instructions, so
    only evals sharing a goal are comparable — the summary groups them into
    runs and reports the trend inside each one.
    """

    def __init__(self, cfg: argparse.Namespace, env: meb.MujocoArm,
                 policy: TrainablePolicy, win: "meb.StateWindow"):
        self.cfg, self.env, self.policy = cfg, env, policy
        self.win = win

    # ── MuJoCo + policy state, so a probe can be undone ─────────────────────
    def _snapshot(self) -> Dict[str, Any]:
        d, env = self.env.d, self.env
        return {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                "sim_time": env.sim_time, "prev_action": env.prev_action,
                "prev_onehot": env.prev_onehot.copy(), "target": env.target,
                "q_hold": env.q_hold.copy(), "carry": self.policy.carry,
                "window": list(self.win.texts), "frames": self.policy.frames}

    def _restore(self, snap: Dict[str, Any]) -> None:
        d, env = self.env.d, self.env
        d.qpos[:] = snap["qpos"]
        d.qvel[:] = snap["qvel"]
        env.sim_time = snap["sim_time"]
        env.prev_action = snap["prev_action"]
        env.prev_onehot[:] = snap["prev_onehot"]
        env.target = snap["target"]
        env.q_hold[:] = snap["q_hold"]
        self.policy.carry = snap["carry"]
        self.win.restore(snap["window"])
        self.policy.frames = snap["frames"]
        mujoco.mj_forward(env.m, d)          # contacts/derived terms recomputed

    # ── the expert ceiling, on the same protocol ────────────────────────────
    def expert_ceiling(self) -> Dict[str, Any]:
        """Same protocol, driven by the rule that generated the supervised data
        (`train_loop_robot.SensorWorld.episode`).  This is the achievable ceiling
        for the CURRENT bridge: if the expert cannot reach, no amount of RL on
        this policy can — which is how the control-loop defect (one open-loop
        torque held across 100 substeps, tool whipping at 2.7 m/s) was found."""
        env = self.env
        snap = self._snapshot()
        try:
            env.reset()
            obs, d0 = env.observe()
            dists: List[float] = []
            for _ in range(max(1, self.cfg.eval_frames)):
                o, _ = env.observe()
                a = expert_action(o, env.target, env.task)
                env.command(a)
                env.note_action(a)
                env.step()
                _, dn = env.observe()
                dists.append(dn)
            hit = [i + 1 for i, x in enumerate(dists) if x < env.reach]
            return {"d_first": float(d0), "dist": float(np.mean(dists)),
                    "best": float(np.min(dists)),
                    "t_reach": float(hit[0]) if hit else float("nan"),
                    "reach": 100.0 * len(hit) / len(dists),
                    "goal": tlr.COLORS[env.target]}
        finally:
            self._restore(snap)

    def __call__(self, tag: str) -> Dict[str, Any]:
        cfg, env, policy = self.cfg, self.env, self.policy
        snap = self._snapshot()
        try:
            env.reset()                      # ready pose · spheres at start · t=0
            policy.carry = None              # same clean slate as the training start
            obs, d0 = env.observe()
            self.win.reset(meb.state_sentence(obs))
            dists: List[float] = []
            hist = [0] * tlr.NUM_ACTIONS     # what the greedy path actually does
            flips = 0
            prev_a = -1
            cyc = 0.0
            agree = 0                        # policy vs the expert, same state
            for _ in range(max(1, cfg.eval_frames)):
                o, _ = env.observe()
                sids, apos = self.win.push(meb.state_sentence(o))
                go = policy.greedy(sids, apos)
                agree += int(go.action == expert_action(o, env.target, env.task))
                env.command(go.action)
                env.note_action(go.action)
                env.step()
                _, dn = env.observe()
                dists.append(dn)
                hist[go.action] += 1
                flips += int(prev_a >= 0 and go.action != prev_a)
                prev_a = go.action
                cyc += go.cycles
            hit = [i + 1 for i, x in enumerate(dists) if x < env.reach]
            m = {"d_first": float(d0), "dist": float(np.mean(dists)),
                 "best": float(np.min(dists)),
                 "t_reach": float(hit[0]) if hit else float("nan"),
                 "reach": 100.0 * len(hit) / len(dists),
                 "cycles": cyc / len(dists),
                 # what the greedy path did: an argmax over six discrete actions
                 # can thrash between two near-tied logits, and a thrashing
                 # trajectory cannot hold a 6 cm band — this makes that visible
                 "acts": tuple(hist), "flips": flips,
                 # agreement with the expert ON THE LIVE STATES.  The supervised
                 # data comes from the synthetic world, whose state distribution
                 # is not the bridge's, so this is the number that says whether
                 # the warm start transferred at all (chance = 1/6 = 16.7 %)
                 "agree": 100.0 * agree / max(1, cfg.eval_frames),
                 # which colour the protocol was run for: two evals with
                 # different colours are NOT comparable, so the caller groups by it
                 "goal": tlr.COLORS[env.target]}
        finally:
            self._restore(snap)
        t_reach = "  --  " if math.isnan(m["t_reach"]) else f"{m['t_reach']:4.0f}"
        print(f"  eval        : {tag:<22} goal={tlr.COLORS[env.target]:<6} "
              f"d0={m['d_first']:.3f}m d_mean={m['dist']:.3f}m "
              f"best={m['best']:.3f}m reach={m['reach']:5.1f}% "
              f"t_reach={t_reach} cyc={m['cycles']:.2f}", flush=True)
        _tot = max(1, sum(m["acts"]))
        print(f"  {'':12s}: greedy path — " + " ".join(
            f"{k}:{c/_tot:.2f}" for k, c in zip(tlr.ACTION_NAMES, m["acts"]))
            + f" · {m['flips']} switch(es) · expert agrees on "
            f"{m['agree']:.1f}% of decisions", flush=True)
        return m


# =============================================================================
# 6 · LIVE IMITATION WARM-UP  (DAgger round 0: the learner walks, the expert labels)
# =============================================================================
class LiveImitator:
    """
    Why this exists, measured rather than assumed: the supervised warm start from
    `train_loop_robot.py` is fitted on `SensorWorld` windows, and on the bridge's
    *live* states both `_sup2.pt` (600 steps) and `_sup3.pt` (2400 steps, 0.67
    accuracy on the synthetic eval) emit ONE constant action — `-x` for all 40
    protocol frames, agreeing with the expert on 0.0 % of decisions while chance
    is 16.7 %.  A policy whose forward pass is constant cannot be repaired by A2C:
    the action a live state needs is never sampled, so its advantage is never
    observed.  The fix is not a better reward, it is a training distribution.

    So: run the *learner* in the live scene and let the expert label the states
    the learner actually reaches (DAgger — Ross, Gordon & Bagnell 2011).  The
    expert is a rule over the same 28-D obs (`expert_action`), so a label costs
    one function call, and the same joint loss as the supervised trainer
    (cross-entropy + `--tau`·ponder) is enough to fit it.  A2C then starts from a
    policy that already tracks instead of from a constant.

    Frames are stored with the *input* carry the learner was fed, so the fit
    reproduces the rollout's forward condition exactly; a frame whose carry was
    absent (the first frame of a round — the model was fed no memory token) is
    dropped rather than stored with a zero latent, because a zero latent inserts
    a memory token and shifts the readout.
    """

    def __init__(self, cfg: argparse.Namespace, policy: TrainablePolicy, accel,
                 env, win: "meb.StateWindow") -> None:
        self.cfg, self.policy, self.env = cfg, policy, env
        self.accel = accel
        self.win = win
        total = max(0, cfg.imitate_rounds) * max(0, cfg.imitate_frames)
        self.sids = torch.zeros((max(1, total), cfg.window * win.L),
                                dtype=torch.long, device=accel.device)
        self.apos = torch.zeros((max(1, total), cfg.window, win.q),
                                dtype=torch.long, device=accel.device)
        self.carry = torch.zeros((max(1, total), policy.hidden), device=accel.device)
        # The prompt is part of the transition, not a global: a round retargets
        # every `--retarget-every` frames, and the *label* depends on the verb
        # (throw holds the object, push never closes the jaw).  Re-forwarding an
        # old frame under the newest instruction would train it against the wrong
        # task, so each stored frame keeps the prompt it was actually fed.
        self.ti = int(policy.instr_ids.shape[1])
        self.instr = torch.zeros((max(1, total), self.ti), dtype=torch.long,
                                 device=accel.device)
        self.imask = torch.zeros((max(1, total), self.ti), dtype=torch.bool,
                                 device=accel.device)
        self.label = torch.zeros((max(1, total),), dtype=torch.long,
                                 device=accel.device)
        self.tasks = torch.zeros((max(1, total),), dtype=torch.long,
                                 device=accel.device)     # verb, for --task-weight
        # The typed answers are labelled by the EXPERT on the state the learner
        # visited, at collect time — the text state alone cannot be re-labelled
        # later (`axis_x`/`speed` are read off the numbers the expert saw).
        self.qids = tuple(policy.qids)
        self.q_lab = torch.zeros((max(1, total), max(1, len(self.qids))),
                                 dtype=torch.long, device=accel.device)
        # Speech targets, labelled at collect time by the SAME event detector as
        # the synthetic generator (`tlr.detect_speech_event`) and spread over the
        # frames by the SAME `tlr.stream_schedule` (K = --speak-tokens): per
        # frame the sentence + END of the ACTIVE utterance (-100 padded; silent
        # = [END]) and `sp_off` = tokens of it already spoken.  Width = the
        # longest template.
        self.end_id = tlr.speech_end_id(policy.tok)
        self.speak_tokens = max(1, int(getattr(cfg, "speak_tokens", 1)))
        self.ls = 1 + max(len(tlr.encode_sentence(policy.tok,
                                                  tlr.speech_sentence(e, c)))
                          for e in tlr.SPEECH_EVENTS for c in tlr.COLORS)
        self.sp = torch.full((max(1, total), self.ls), -100, dtype=torch.long,
                             device=accel.device)
        self.sp_off = torch.zeros((max(1, total),), dtype=torch.long,
                                  device=accel.device)
        self.n = 0
        self.rng = np.random.default_rng(int(cfg.seed) + 7919)
        self.oom_events = 0
        lr = cfg.imitate_lr if cfg.imitate_lr > 0 else cfg.lr
        self.opt = torch.optim.AdamW(policy.trainable, lr=lr, betas=(0.9, 0.95),
                                     weight_decay=0.0, eps=1e-8)
        self.scaler = None                      # fp16 only — bf16 needs none
        if accel.amp and accel.scaler:
            try:
                self.scaler = torch.amp.GradScaler(accel.amp_device_type)
            except (TypeError, ValueError):     # older torch
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    @torch.no_grad()
    def collect(self, frames: int) -> float:
        """Roll the learner from the ready pose, label every visited state.

        The instruction rotates on the `--retarget-every` schedule — the same
        clock the A2C loop uses — so `--goal any` really does cover all four
        colour slots instead of only the one the collector started with.
        """
        cfg, policy, env = self.cfg, self.policy, self.env
        env.reset()
        policy.carry = None
        obs, _ = env.observe()
        self.win.reset(meb.state_sentence(obs))
        agree = 0
        prev_d = 0.0          # mirrors SensorWorld: no `arrive` on the first frame
        # per-frame event sentence tokens (None = no event) and the pool slot
        # the frame was stored in (None = not stored); the stream schedule runs
        # over the WHOLE rollout after it, exactly as `collate` does per episode
        events: List[Optional[List[int]]] = []
        slots: List[Optional[int]] = []
        for i in range(frames):
            # The pool has to see every instruction: otherwise DAgger trains the
            # one goal the collector happened to start with.  Measured — a
            # `--goal any --retarget-every 100` run whose pool stayed red-only
            # scored 95 % in reach on red and 0 % on green/blue/yellow.
            if cfg.retarget_every and i and i % cfg.retarget_every == 0:
                text, _ = env.new_instruction(self.rng, cfg.goal, cfg.task)
                policy.set_instruction(text)
            o, _ = env.observe()
            sids, apos = self.win.push(meb.state_sentence(o))
            a_expert = expert_action(o, env.target, env.task)
            h_expert = expert_height(o, env.target, env.task)
            # the event this frame is narrated with (same rule as SensorWorld)
            d_now = float(np.maximum(o[meb.SL_DIST].astype(np.float64),
                                     1e-4)[env.target])
            held = float(np.ravel(o[meb.SL_GRIP])[0]) > 0.5
            ev = tlr.detect_speech_event(
                int(env.task), closes=(a_expert == 4 and not held),
                opens=(held and a_expert not in (4, 5)),
                arrive=(d_now < meb.TRAIN_REACH_UNITS
                        and prev_d >= meb.TRAIN_REACH_UNITS))
            prev_d = d_now
            sent = tlr.speech_sentence(ev, tlr.COLORS[env.target])
            events.append(tlr.encode_sentence(policy.tok, sent) if sent else None)
            slots.append(None)
            ao, carry_in = policy.act(sids, apos)  # the learner's own trajectory
            agree += int(ao.action == a_expert)
            if cfg.carry and carry_in is None:
                pass                              # no memory token was fed here
            else:
                self.sids[self.n].copy_(sids[0])
                self.apos[self.n].copy_(apos[0])
                if cfg.carry:
                    self.carry[self.n].copy_(carry_in[0])
                self.label[self.n] = a_expert
                slots[-1] = self.n
                if self.qids and policy.model.bank is not None:
                    ot = torch.from_numpy(o).to(self.sids.device).reshape(1, 1, -1)
                    at = torch.tensor([[a_expert]], device=self.sids.device)
                    lab = tlr.derive_question_labels(
                        policy.model.bank, ot, at,
                        extra={"height": torch.tensor([[h_expert]],
                                                      device=self.sids.device)})
                    self.q_lab[self.n] = torch.tensor(
                        [int(lab[q].reshape(-1)[0]) for q in self.qids],
                        device=self.sids.device)
                self.tasks[self.n] = int(env.task)
                ids, msk = policy.instr_ids[0], policy.instr_mask[0]
                if ids.shape[0] > self.ti:        # a longer instruction arrived
                    grow = ids.shape[0] - self.ti
                    self.instr = torch.cat([self.instr, torch.zeros(
                        (self.instr.shape[0], grow), dtype=torch.long,
                        device=self.instr.device)], dim=1)
                    self.imask = torch.cat([self.imask, torch.zeros(
                        (self.imask.shape[0], grow), dtype=torch.bool,
                        device=self.imask.device)], dim=1)
                    self.ti = int(ids.shape[0])
                self.instr[self.n, :ids.shape[0]].copy_(ids)
                self.imask[self.n, :ids.shape[0]].copy_(msk)
                self.n += 1
            drive(env, ao)
            env.step()
        ids, offs = tlr.stream_schedule(events, self.speak_tokens, self.end_id)
        for slot, tgt, off in zip(slots, ids, offs):
            if slot is None:
                continue
            tgt = tgt[:self.ls]
            self.sp[slot].fill_(-100)
            self.sp[slot, :len(tgt)] = torch.tensor(tgt, dtype=torch.long,
                                                    device=self.sp.device)
            self.sp_off[slot] = int(off)
        return 100.0 * agree / max(1, frames)

    def fit(self, steps: int, batch: int) -> Dict[str, float]:
        """Question CE (+ task, + speech) + tau·ponder on the collected
        (state, expert-labelled) frames.  `ce_*` is the question CE of the
        optimiser steps; `acc` is the json_action decoded from the arg-max
        answers vs the expert action, measured AFTER the fit over a fixed
        random sample of up to --imitate-acc-windows pool windows (not one
        batch).  Each optimiser step accumulates --imitate-accum micro-batches
        of `batch` windows, so the effective batch is batch × accum."""
        cfg, policy = self.cfg, self.policy
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        ce_first = ce_last = float("nan")
        tw = float(getattr(cfg, "text_weight", 0.0))
        accum = max(1, int(getattr(cfg, "imitate_accum", 1)))
        done = 0
        policy.model.train()
        for _ in range(steps):
            self.opt.zero_grad(set_to_none=True)
            ce_sum, ok = 0.0, True
            for _m in range(accum):
                k = min(batch, self.n)
                idx = torch.randint(0, self.n, (k,), generator=gen).to(self.sids.device)
                try:
                    loss, ce = self._micro_loss(idx, tw)
                    loss = loss / accum
                    if self.scaler is None:
                        loss.backward()
                    else:
                        self.scaler.scale(loss).backward()
                except RuntimeError as exc:
                    if not tlr.is_oom(exc):          # a real bug
                        raise
                    ok = False
                    break
                ce_sum += float(ce.detach())
            if not ok:
                self.oom_events += 1
                self.opt.zero_grad(set_to_none=True)
                tlr.free_device_cache(self.accel)
                print(f"    OOM during an imitation step — skipped "
                      f"({self.oom_events} so far, cache flushed; try a smaller "
                      f"--imitate-batch)", flush=True)
                continue
            if self.scaler is None:
                torch.nn.utils.clip_grad_norm_(policy.trainable, cfg.clip)
                self.opt.step()
            else:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(policy.trainable, cfg.clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            ce_last = ce_sum / accum
            done += 1
            if done == 1:
                ce_first = ce_last
        acc, n_acc = self.pool_accuracy(batch)
        speak = float(((self.sp[:self.n] >= 0).sum(1) > 1).float().mean()) \
            if self.n else 0.0
        return {"ce_first": ce_first, "ce_last": ce_last, "acc": 100.0 * acc,
                "acc_windows": float(n_acc), "speak_frac": 100.0 * speak,
                "samples": float(self.n), "steps": float(done)}

    def _micro_loss(self, idx: torch.Tensor, tw: float):
        """One forward on pool windows `idx` → (joint loss, question CE)."""
        cfg, policy = self.cfg, self.policy
        sids, apos = self.sids[idx], self.apos[idx]
        carry = self.carry[idx] if cfg.carry else None
        with torch.amp.autocast(device_type=self.accel.amp_device_type,
                                dtype=self.accel.dtype,
                                enabled=self.accel.amp):
            sio = (tlr.speech_io_from_targets(self.sp[idx], self.sp_off[idx],
                                              self.speak_tokens)
                   if tw > 0.0 else None)
            out = policy.forward_out(sids, apos, carry,
                                     instr_ids=self.instr[idx, :self.ti],
                                     instr_mask=self.imask[idx, :self.ti],
                                     speech_ids=None if sio is None else sio[0],
                                     speech_mask=None if sio is None else sio[1])
            # the typed answers were labelled by the expert when the frame was
            # visited — a text state cannot be re-labelled after the fact
            q_lab = ({qid: self.q_lab[idx, j].unsqueeze(1).expand(-1, cfg.window)
                      for j, qid in enumerate(self.qids)} or None)
            sce = (policy.model.speech_ce(out.speech_h, sio[2])
                   if sio is not None else None)
            loss, parts = tlr.joint_loss(out, cfg.tau,
                                         q_labels=q_lab,
                                         q_weight=float(getattr(cfg, "q_weight", 1.0)),
                                         speech_ce=sce, text_weight=tw,
                                         task_ids=self.tasks[idx],
                                         task_weight=float(getattr(
                                             cfg, "task_weight", 0.0)))
        return loss, parts.get("questions", loss)

    @torch.no_grad()
    def pool_accuracy(self, batch: int) -> Tuple[float, int]:
        """json_action accuracy (last step of each window vs the expert label)
        over a FIXED random sample of up to --imitate-acc-windows pool windows —
        the same seed every round, so rounds are comparable and the number is
        not quantised by one tiny batch.  Returns (fraction, windows scored)."""
        cfg, policy = self.cfg, self.policy
        cap = int(getattr(cfg, "imitate_acc_windows", 256))
        if self.n == 0 or cap <= 0:
            return float("nan"), 0
        g = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 1)
        sel = torch.randperm(self.n, generator=g)[:min(cap, self.n)]
        was_training = policy.model.training
        policy.model.eval()
        hit = 0
        try:
            for s in range(0, sel.numel(), max(1, batch)):
                idx = sel[s:s + max(1, batch)].to(self.sids.device)
                carry = self.carry[idx] if cfg.carry else None
                with torch.amp.autocast(device_type=self.accel.amp_device_type,
                                        dtype=self.accel.dtype,
                                        enabled=self.accel.amp):
                    out = policy.forward_out(self.sids[idx], self.apos[idx], carry,
                                             instr_ids=self.instr[idx, :self.ti],
                                             instr_mask=self.imask[idx, :self.ti])
                pred = tlr.json_action({k: v.float() for k, v in
                                        out.q_logits.items()})[:, -1]
                hit += int((pred.to(self.label.device) == self.label[idx]).sum())
        finally:
            policy.model.train(was_training)
        return hit / sel.numel(), int(sel.numel())

    def run(self) -> Optional[Dict[str, float]]:
        cfg = self.cfg
        if min(cfg.imitate_rounds, cfg.imitate_frames, cfg.imitate_steps) <= 0:
            print("  imitate     : off (--imitate-rounds 0) — the learner starts "
                  "from the synthetic warm start alone", flush=True)
            return None
        lr = cfg.imitate_lr if cfg.imitate_lr > 0 else cfg.lr
        accum = max(1, int(getattr(cfg, "imitate_accum", 1)))
        print(f"  imitate     : DAgger round 0 — {cfg.imitate_rounds} round(s) × "
              f"{cfg.imitate_frames} learner frames, expert-labelled · "
              f"{cfg.imitate_steps} CE steps × {cfg.imitate_batch}×{accum} "
              f"windows/step · lr={lr:g} tau={cfg.tau:g} · lm_head "
              f"{'frozen' if getattr(cfg, 'freeze_lm_head', False) else 'trained'}",
              flush=True)
        last: Dict[str, float] = {}
        for r in range(1, cfg.imitate_rounds + 1):
            agreed = self.collect(cfg.imitate_frames)
            last = self.fit(cfg.imitate_steps, cfg.imitate_batch)
            print(f"  [{r}/{cfg.imitate_rounds}] pool={int(last['samples'])} windows · "
                  f"question CE {last['ce_first']:.3f} → {last['ce_last']:.3f} · "
                  f"json_action pool acc {last['acc']:.1f}% "
                  f"(n={int(last.get('acc_windows', 0))}) · speaking "
                  f"{last.get('speak_frac', 0.0):.0f}% of frames · the rollout agreed "
                  f"with the expert on {agreed:.1f}% of its {cfg.imitate_frames} "
                  f"decisions", flush=True)
        return last


# =============================================================================
# 7 · CHECKPOINT  (bridge-compatible payload + the critic)
# =============================================================================
def save_payload(path: str, cfg: argparse.Namespace, policy: TrainablePolicy,
                 updater: A2CUpdater, extra: Dict[str, Any]) -> int:
    """
    Same shape `train_loop_robot.save_checkpoint` writes, so
    `mujoco_env_bridge.py --checkpoint <path>` keeps working — plus the critic
    and the RL bookkeeping.
    """
    sd = {n: p.detach().cpu()
          for n, p in policy.model.named_parameters() if p.requires_grad}
    payload = {
        "model_id": cfg.model,
        "trainable": cfg.trainable,
        "args": vars(cfg),
        "sensor_dim": meb.SENSOR_DIM,
        "state": "text",
        "action_names": list(tlr.ACTION_NAMES),
        # the question bank + Jev label tokens the answers were trained on, so
        # `tlr.load_trainable_state` can warn when a reader binds different ones
        "qbind": policy.qbind.signature(),
        "trainable_state": sd,
        "critic_state": policy.critic.state_dict(),
        "rl": dict(extra),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)                      # never leave a half-written artifact
    return len(sd)


# =============================================================================
# 8 · SELF-TEST  (reward signs · γ-returns · PG direction · critic fit ·
#                 gradient flow · carry contract · eval protocol · payload)
# =============================================================================
def _text_window(cfg: argparse.Namespace, policy: TrainablePolicy, accel,
                 steps: int = 0) -> "meb.StateWindow":
    """A `StateWindow` for the self-test — the real renderer, a zero frame."""
    win = meb.StateWindow(int(steps or cfg.window), policy.tok, policy.qbind,
                          accel.device)
    win.reset(meb.state_sentence(np.zeros(meb.SENSOR_DIM, dtype=np.float32)))
    return win


def _rand_window(win: "meb.StateWindow") -> Window:
    """Push a synthetic-but-valid harness frame through the real renderer."""
    o = np.random.default_rng(7).normal(0.0, 0.3, meb.SENSOR_DIM).astype(np.float32)
    o[meb.SL_GRIP] = 0.0
    o[meb.SL_PREV] = 0.0
    return win.push(meb.state_sentence(o))


def run_self_test(cfg: argparse.Namespace, accel) -> int:
    print("=" * 78)
    print("LIVE-RL SELF-TEST  (the RL math, not the physics luck)")
    print("=" * 78)
    fails = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  —  {detail}" if detail else ""), flush=True)

    spec = RewardSpec.from_cfg(cfg)
    print(spec.table())

    # ── t1 · reward shaping -------------------------------------------------
    r_close, t_close = compute_reward(spec, 0.40, 0.30, 0.0625, False, 2.0)
    r_away, _ = compute_reward(spec, 0.30, 0.40, 0.0625, False, 2.0)
    _, t_flat = compute_reward(spec, 0.35, 0.35, 0.0625, False, 0.0)
    _, t_reach = compute_reward(spec, 0.20, 0.05, 0.0625, False, 0.0)
    r_cheap, _ = compute_reward(spec, 0.30, 0.30, 0.0625, False, 1.0)
    r_costly, _ = compute_reward(spec, 0.30, 0.30, 0.0625, False, 4.0)
    ok("progress term is signed (closer > 0 > away)",
       r_close > 0.0 > r_away, f"closer={r_close:+.3f} away={r_away:+.3f}")
    ok("standing still earns no progress term",
       abs(t_flat["progress"]) < 1e-12, f"progress={t_flat['progress']:.1e}")
    # The parking trap, which is what the static-action policy was doing: a
    # shaping term that pays a positive amount for merely existing in a state is
    # an income stream, and a finite reach bonus can lose to it.  With Φ = −d the
    # standing-still income is (1−γ)·d — three orders below one real step.
    ok("standing still earns no shaping income (parking is not a strategy)",
       abs(t_flat["shaping"]) < 0.01 * abs(t_close["progress"]),
       f"still={t_flat['shaping']:+.4f} vs one 10cm step={t_close['progress']:+.3f}")
    # Potential-based shaping must telescope: at γ=1 the sum over any path is
    # (d_start − d_end), so it cannot retarget the optimum — only densify it.
    _spec1 = replace(spec, gamma=1.0)
    _s1 = compute_reward(_spec1, 0.40, 0.30, 0.0625, False, 0.0)[1]["shaping"]
    _s2 = compute_reward(_spec1, 0.30, 0.20, 0.0625, False, 0.0)[1]["shaping"]
    ok("shaping telescopes to (d_start − d_end) at γ=1",
       abs((_s1 + _s2) - 0.20) < 1e-12, f"Σshaping={_s1 + _s2:+.4f} vs +0.2000")
    ok("crossing the reach line pays exactly the bonus",
       abs(t_reach["bonus"] - spec.reach_bonus) < 1e-12,
       f"bonus={t_reach['bonus']:g} at d=0.05m < reach")
    ok("pondering costs reward (ACT compute is not free)",
       r_cheap > r_costly, f"ponder 1→4: {r_cheap:+.3f} → {r_costly:+.3f}")
    r_clip, _ = compute_reward(spec, 10.0, 0.0, 0.0625, False, 0.0)
    ok("per-frame reward is clipped", abs(r_clip) <= spec.clip + 1e-12,
       f"|r|≤{spec.clip:g} (got {r_clip:+.3f})")

    # ── t2 · discounted returns over a rollout (hand-computed) ---------------
    buf = RolloutBuffer(8, cfg.window, 4, 1, 4, torch.device("cpu"))
    blank_sids = torch.zeros(1, cfg.window * 4, dtype=torch.long)
    blank_apos = torch.zeros(1, cfg.window, 1, dtype=torch.long)
    for r_ in (1.0, 1.0, 1.0, 1.0):
        buf.add(blank_sids, blank_apos, None, 0, r_, False, 0.0, 0.0)
    g = [round(x, 6) for x in buf.returns(0.5, 8.0).tolist()]
    ok("γ-returns discount backwards and bootstrap the tail",
       g == [2.375, 2.75, 3.5, 5.0], f"G={g} (r=1, γ=0.5, boot=8)")
    buf.done[2] = True
    g2 = [round(x, 6) for x in buf.returns(0.5, 8.0).tolist()]
    ok("an episode end cuts the return (no value leaks across a reset)",
       g2 == [1.75, 1.5, 1.0, 5.0], f"G={g2} (done at t=2)")

    # ── t3 · policy gradient direction (the real loss, a toy policy) ---------
    torch.manual_seed(0)
    toy = nn.Linear(4, 6)
    toy_opt = torch.optim.Adam(toy.parameters(), lr=0.1)
    x = torch.randn(1, 4)
    wanted = torch.tensor([3])
    p_before = 0.0
    for i in range(80):
        logits = toy(x)
        dist = torch.distributions.Categorical(logits=logits)
        if i == 0:
            p_before = float(torch.softmax(logits.detach(), -1)[0, 3])
        loss, _, _, _ = a2c_loss(dist.log_prob(wanted), dist.entropy(),
                                 torch.zeros(1), torch.ones(1), torch.ones(1),
                                 0.0, 0.0)
        toy_opt.zero_grad()
        loss.backward()
        toy_opt.step()
    logits = toy(x).detach()
    p_after = float(torch.softmax(logits, -1)[0, 3])
    ok("positive advantage raises P(sampled action) and flips the argmax",
       p_after > p_before and int(logits.argmax()) == 3,
       f"P(a*) {p_before:.3f} → {p_after:.3f}")

    # ── t4 · critic regression -------------------------------------------------
    torch.manual_seed(0)
    critic = Critic(8, 32)
    c_opt = torch.optim.Adam(critic.parameters(), lr=5e-3)
    states = torch.randn(64, 8)
    targets = (states[:, :1] * 1.5 - 0.25).squeeze(-1)   # a learnable scalar map
    first = 0.0
    for i in range(300):
        v = critic(states)
        mse = F.mse_loss(v, targets)
        if i == 0:
            first = float(mse.detach())
        c_opt.zero_grad()
        mse.backward()
        c_opt.step()
    last = float(F.mse_loss(critic(states), targets).detach())
    ok("critic (baseline) regresses the return", last < first * 0.2,
       f"MSE {first:.4f} → {last:.6f}")

    # ── t5 · the real model: rollout → ONE batched update --------------------
    print(f"  · building the real policy for the gradient tests "
          f"({accel.name or accel.device})")
    tok, trunk = tlr.load_base("tiny", accel)
    policy = TrainablePolicy(cfg, accel, tok, trunk, payload=None)
    win = _text_window(cfg, policy, accel)
    updater = A2CUpdater(cfg, policy, accel, win)
    window = _rand_window(win)
    ao, carry_in = policy.act(*window)
    if True:   # control is always JSON (the action head was removed)
        # the JSON policy is a product over the bank: Σp over the concatenated
        # per-question distributions is n_q, logπ is the sum of n_q log-probs,
        # and every question carries a label that the document reports back
        H = float(-np.sum(ao.probs * np.log(np.clip(ao.probs, 1e-12, 1.0))))
        hmax = sum(math.log(max(s.n_out, 2)) for s in policy.bank.specs)
        n_q = len(policy.bank.specs)
        doc = policy.last_json or {}
        ok("act(): json control samples every question, V finite, Σp = n_q",
           ao.logprob <= 1e-6 and 0.0 <= H <= hmax + 1e-6
           and math.isfinite(ao.value)
           and abs(float(ao.probs.sum()) - n_q) < 1e-5
           and len(ao.labels) == n_q
           and ao.cmd is not None
           and doc.get("executed", {}).get("action") == tlr.ACTION_NAMES[ao.cmd.action],
           f"logπ={ao.logprob:+.3f} H={H:.3f} (max {hmax:.3f}) V={ao.value:+.3f} · "
           f"Σp={float(ao.probs.sum()):.3f} of {n_q} · "
           f"labels={ao.labels} · cmd=\"{ao.cmd.note}\"")
        shown = " ".join("%s=%s" % (q, doc.get(q, {}).get("sampled", "abstained"))
                         for q in ao.labels)
        ok("the JSON document reports the labels that were executed",
           all(doc.get(s.qid, {}).get("sampled") == s.labels[ao.labels[s.qid]]
               or doc.get(s.qid, {}).get("abstained")
               for s in policy.bank.specs),
           shown)
    ok("act(): json_action decodes the sampled labels into the executed token",
       ao.action == tlr.json_action_one(ao.labels)
       and 0 <= ao.action < tlr.NUM_ACTIONS,
       f"labels={ao.labels} → {tlr.ACTION_NAMES[ao.action]}")
    doc = policy.last_json or {}
    ok("the JSON document carries a top-level `say` (null when silent)",
       "say" in doc and (doc["say"] is None or isinstance(doc["say"], str))
       and ao.say == doc["say"],
       f"say={doc.get('say')!r}")
    ok("ACT still forms a convex combination on the rollout path",
       ao.mass_err < 1e-4, f"Σα−1 = {ao.mass_err:.2e}")
    # frame 1 carries nothing (nothing has been computed yet); frame 2 must be
    # fed frame 1's mixed latent, and that tensor must be detached — a live
    # graph spanning frames is what made the per-frame update crash.
    window2 = _rand_window(win)
    _, carry2 = policy.act(*window2)
    if cfg.carry:
        ok("the carry threaded between frames is detached (no graph crosses one)",
           carry2 is not None and not carry2.requires_grad and carry_in is None,
           f"frame1 carry={'None (nothing carried yet)' if carry_in is None else 'present'}"
           f" · frame2 carry={'detached' if carry2 is None else ('LIVE' if carry2.requires_grad else 'detached')}"
           + ("" if carry2 is None else f" |Δ|={float(carry2.abs().mean()):.3e}"))
    else:
        ok("--no-carry: the latent is deliberately not threaded", carry2 is None,
           "carry_in=None on every frame")

    # a synthetic 2-frame rollout with hand-set rewards: the single update must
    # move the adapter, leave the trunk frozen, report the analytic return, and
    # reproduce the rollout's own log-probs (no off-policy drift).
    updater.buf.reset()
    for r_ in (1.0, -0.5):
        w_ = _rand_window(win)
        a_, c_ = policy.act(*w_)
        updater.buf.add(*w_, c_, a_.action, r_, False, a_.logprob, a_.value,
                        labels=a_.labels)
    updater.due = False
    # the arm follows the questionnaire: the answers are read out of the LM
    # head's Jev label tokens on h_ans + task_proj(pooled prompt), so the
    # instruction shortcut `task_proj` (always trainable) is what the policy
    # gradient has to move — there is no per-question parameter left
    get_w = lambda: policy.model.task_proj.weight
    head_name = "task_proj"
    w_before = get_w().detach().clone()
    info = updater.update(boot=0.0)
    changed = not torch.equal(w_before, get_w().detach())
    trunk_grad = policy.model.trunk.embed_tokens.weight.grad
    ok(f"one batched A2C update moves the adapter head ({head_name})",
       changed and updater.updates >= 1,
       f"updates={updater.updates} "
       f"|ΔW|={float((get_w().detach()-w_before).abs().sum()):.3e}")
    ok("the frozen trunk receives no gradient", trunk_grad is None,
       f"embed_tokens.grad={'None (frozen)' if trunk_grad is None else 'PRESENT'}")
    ok("grad-norm is measured before clipping and finished finite",
       info is not None and math.isfinite(info.get("gn", float("nan"))),
       f"gn={info.get('gn', float('nan')):.3f} "
       f"pg={info.get('pg', float('nan')):+.4f} vf={info.get('vf', float('nan')):.4f}")
    ok("policy and critic gradients are normed and clipped as separate groups",
       info is not None and {"gn_pol", "gn_val"} <= set(info)
       and abs(info["gn"] - max(info["gn_pol"], info["gn_val"])) < 1e-9,
       f"gn_pol={info.get('gn_pol', float('nan')):.3f} "
       f"gn_val={info.get('gn_val', float('nan')):.3f} "
       f"(one joint clip would scale both by the larger)")
    exp_ret = (1.0 + cfg.gamma * -0.5 - 0.5) / 2.0
    ok("the reported return is the analytic γ-return of that rollout",
       info is not None and abs(info["ret"] - exp_ret) < 1e-5,
       f"R={info['ret']:+.6f} vs {exp_ret:+.6f}")
    ok("the update pass reproduces the rollout forward (drift ≈ GEMM noise)",
       info is not None and info["drift"] < 1e-2,
       f"|Δlogπ|={info['drift']:.2e} (bf16 batch tiling, not an exactness claim)")

    # ── t6 · payload round-trip (stays playable by the bridge) ---------------
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest_rl.pt")
    try:
        n = save_payload(tmp, cfg, policy, updater, {"selftest": True})
        blob = torch.load(tmp, map_location="cpu", weights_only=False)
        missing, unexpected = tlr.load_trainable_state(policy.model, blob,
                                                       policy.qbind,
                                                       where="selftest")
        ok("checkpoint round-trips with no unexpected keys",
           not unexpected and len(blob["trainable_state"]) == n
           and blob.get("qbind") == policy.qbind.signature(),
           f"{n} tensors · {len(missing)} frozen trunk tensors left at init · "
           f"bank/labels signature saved")
        ok("payload exposes the cls bridge contract",
           blob["sensor_dim"] == meb.SENSOR_DIM
           and tuple(blob["action_names"]) == tuple(tlr.ACTION_NAMES)
           and "critic_state" in blob,
           f"keys={sorted(k for k in blob if k != 'args')}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    # ── t7 · the eval protocol must not perturb training ---------------------
    # The whole point of `EvalProbe` is that a protocol eval is *comparable*
    # across a run and *invisible* to the optimiser.  Both claims are testable:
    # run frames, snapshot, probe twice, compare the training state bit for bit.
    print("  · checking the eval protocol on the real scene "
          f"(MuJoCo {mujoco.__version__} on CPU, no GPU needed)")
    env = meb.MujocoArm(cfg)
    env.reset()
    win = _text_window(cfg, policy, accel)
    obs, _ = env.observe()
    win.reset(meb.state_sentence(obs))
    probe = EvalProbe(cfg, env, policy, win)
    for _ in range(3):                                     # live frames first
        o, _ = env.observe()
        w = win.push(meb.state_sentence(o))
        ao, _ = policy.act(*w)
        drive(env, ao)
        env.step()
    carry_before = None if policy.carry is None else policy.carry.clone()
    qpos_before, qvel_before = env.d.qpos.copy(), env.d.qvel.copy()
    win_before, sim_before = list(win.texts), env.sim_time
    prev_before, tgt_before = env.prev_action, env.target
    m1 = probe("self-test probe #1")
    m2 = probe("self-test probe #2")
    restored = (np.array_equal(qpos_before, env.d.qpos)
                and np.array_equal(qvel_before, env.d.qvel)
                and sim_before == env.sim_time
                and prev_before == env.prev_action
                and tgt_before == env.target
                and (carry_before is None) == (policy.carry is None)
                and (carry_before is None or torch.equal(carry_before, policy.carry))
                and win_before == list(win.texts))
    ok("the eval protocol restores MuJoCo, the carry and the window",
       restored,
       f"qpos/qvel/sim_time/prev_action/target/carry/window identical={restored}")
    ok("the eval protocol is deterministic (d0 and the trajectory repeat)",
       abs(m1["d_first"] - m2["d_first"]) < 1e-12
       and abs(m1["dist"] - m2["dist"]) < 1e-12,
       f"d0={m1['d_first']:.4f}m twice · d_mean {m1['dist']:.4f}m twice · "
       f"reach={m1['reach']:.0f}% · cycles={m1['cycles']:.2f}")

    # ── the ceiling rule itself: units, thresholds and precedence -----------
    def _obs(rel_xy) -> np.ndarray:
        r = np.asarray(rel_xy, dtype=np.float64).reshape(-1, 2)
        pad = np.full((meb.NUM_OBJECTS, 2), 9.0)        # unused slots far away
        pad[:r.shape[0]] = r
        o = np.zeros(meb.SENSOR_DIM, dtype=np.float32)
        o[meb.SL_REL] = pad.reshape(-1)
        o[meb.SL_DIST] = np.linalg.norm(pad, axis=1).clip(1e-4)
        return o

    ok("expert: attraction picks the axis move toward the target",
       expert_action(_obs([(0.5, 0.0)]), 0) == 1
       and expert_action(_obs([(0.0, -0.5)]), 0) == 2,
       "target +x → +x (1) · target −y → −y (2)")
    ok("expert: inside reach → grasp, inside 0.04 → brake (brake wins)",
       expert_action(_obs([(0.05, 0.0)]), 0) == 4
       and expert_action(_obs([(0.03, 0.0)]), 0) == 5,
       f"d=0.05 → 4 · d=0.03 → 5 (reach={meb.TRAIN_REACH_UNITS})")
    ok("expert: a neighbour inside 0.35 repels harder than the goal attracts",
       expert_action(_obs([(0.5, 0.0), (0.06, 0.0)]), 0) == 0
       and expert_action(_obs([(0.5, 0.0), (0.60, 0.0)]), 0) == 1,
       "goal +x, neighbour at d=0.06 → −x (0) · neighbour at d=0.60 → +x (1)")
    _ceil = probe.expert_ceiling()
    ok("expert ceiling runs on the real scene through the eval protocol",
       bool(np.isfinite(_ceil["dist"]) and np.isfinite(_ceil["best"])),
       f"d_mean={_ceil['dist']:.3f}m best={_ceil['best']:.3f}m "
       f"reach={_ceil['reach']:.1f}% on goal={_ceil['goal']}")

    # ── t8 · the DAgger warm-up must store the live distribution faithfully --
    # The stage exists because the supervised checkpoint is constant on live
    # states; what it must guarantee is (a) the learner's own states get in,
    # (b) the expert labels them, (c) the input carry is stored (the fit has to
    # reproduce the rollout's forward condition), and (d) one fit step moves the
    # adapter while the trunk stays frozen.
    print("  · checking the live imitation stage (collect on live states → fit)")
    imitator = LiveImitator(cfg, policy, accel, env, win)
    agreed = imitator.collect(6)
    expect = 6 - (1 if cfg.carry else 0)      # the first frame is fed no carry
    ok("imitation: the learner's live states are stored with expert labels",
       imitator.n == expect
       and int(imitator.label[:imitator.n].min()) >= 0
       and int(imitator.label[:imitator.n].max()) < tlr.NUM_ACTIONS
       and not imitator.carry[:imitator.n].requires_grad
       and torch.unique(imitator.sids[:imitator.n], dim=0).shape[0] > 1,
       f"stored {imitator.n}/{6} frames (the first carries no memory token) · "
       f"labels in [0,{tlr.NUM_ACTIONS}) · windows vary · the rollout agreed with "
       f"the expert on {agreed:.0f}% of its 6 decisions")
    # the imitation loss is question CE (+ task, + speech): the answers read the
    # LM head's label rows on h_ans + task_proj, so task_proj must move
    # (and the untied lm_head, when there is one)
    heads = [("task_proj", policy.model.task_proj.weight)]
    if policy.model.lm_head is not None and policy.model.lm_head.weight.requires_grad:
        heads.append(("lm_head", policy.model.lm_head.weight))
    w_before = [w.detach().clone() for _, w in heads]
    st = imitator.fit(3, min(4, imitator.n))
    _dws = [float((w.detach() - b).abs().max()) for (_, w), b in zip(heads, w_before)]
    _dw = min(_dws)
    ok("imitation: one question-CE+ponder fit moves "
       + "/".join(n for n, _ in heads) + ", trunk stays frozen",
       _dw > 0 and np.isfinite(st["ce_last"])
       and policy.model.trunk.embed_tokens.weight.grad is None,
       f"CE {st['ce_first']:.3f} → {st['ce_last']:.3f} · json_action acc "
       f"{st['acc']:.1f}% · speaking {st['speak_frac']:.0f}% · "
       + " ".join(f"|Δ{n}|={d:.3e}" for (n, _), d in zip(heads, _dws))
       + " · embed_tokens.grad=None (frozen)")
    sp_rows = imitator.sp[:imitator.n]
    ok("imitation: every stored frame has a speech target ending in END",
       bool(all(int(r[(r >= 0)].numel()) >= 1
                and int(r[(r >= 0)][-1]) == imitator.end_id for r in sp_rows)),
       f"{int(((sp_rows >= 0).sum(1) > 1).sum())}/{imitator.n} speaking frame(s)")
    # --imitate-accum: micro-batches accumulate into one step (the VRAM-safe
    # way to a larger effective batch); the accuracy is scored over the pool
    # sample, not the last micro-batch, so it is NOT quantised to 1/batch.
    cfg.imitate_accum = 2
    st2 = imitator.fit(2, min(2, imitator.n))
    cfg.imitate_accum = 1
    cap = int(getattr(cfg, "imitate_acc_windows", 256))
    n_acc = int(st2["acc_windows"])
    ok("imitation: --imitate-accum 2 steps, accuracy scored over the pool sample",
       np.isfinite(st2["ce_last"]) and int(st2["steps"]) == 2
       and n_acc == min(cap, imitator.n)
       and abs(st2["acc"] / 100.0 * n_acc - round(st2["acc"] / 100.0 * n_acc)) < 1e-6,
       f"CE {st2['ce_first']:.3f} → {st2['ce_last']:.3f} · pool acc "
       f"{st2['acc']:.1f}% over {n_acc}/{imitator.n} windows")
    # --freeze-lm-head keeps the (untied) answer read-out out of the optimiser
    cfg_f = argparse.Namespace(**vars(cfg))
    cfg_f.freeze_lm_head = True
    tok_f, trunk_f = tlr.load_base("tiny", accel)
    pol_f = TrainablePolicy(cfg_f, accel, tok_f, trunk_f, payload=None)
    if pol_f.model.lm_head is None:
        ok("--freeze-lm-head: tied head (embed_tokens) is frozen anyway",
           not pol_f.model.trunk.embed_tokens.weight.requires_grad,
           "no untied lm_head on this trunk")
    else:
        ids_f = {id(p) for p in pol_f.trainable}
        ok("--freeze-lm-head: lm_head is excluded from the trainable set",
           not pol_f.model.lm_head.weight.requires_grad
           and id(pol_f.model.lm_head.weight) not in ids_f
           and pol_f.n_trainable < policy.n_trainable,
           f"trainable {pol_f.n_trainable/1e6:.3f}M frozen vs "
           f"{policy.n_trainable/1e6:.3f}M with the head")
    del pol_f, trunk_f, tok_f

    # The verb has to reach the LIVE expert, not just the synthetic generator:
    # `push` never closes the jaw and never brakes, `throw` never grasps.  If the
    # task never made it into `expert_action`, the pool would be labelled with
    # the default verb and the multi-task interface would be decoration.
    task_seen: Dict[str, List[int]] = {}
    for task_name in ("push", "throw"):
        cfg.task = task_name
        imitator.n = 0
        imitator.collect(6)
        task_seen[task_name] = sorted(set(imitator.label[:imitator.n].tolist()))
    cfg.task = "any"
    ok("imitation: the verb reaches the live expert's labels",
       all(a <= 3 for a in task_seen["push"]) and 4 not in task_seen["throw"],
       f"push labels {task_seen['push']} (steering only — never grasp/brake) · "
       f"throw labels {task_seen['throw']} (never grasp)")

    print("-" * 78)
    print(f"  {'SELF-TEST PASSED' if not fails else f'{fails} CHECK(S) FAILED'}"
          f" · frames/updates exercised: {policy.frames}/{updater.updates}")
    return 0 if not fails else 1


# =============================================================================
# 9 · THE LIVE RL LOOP
# =============================================================================
def run_rl(cfg: argparse.Namespace, accel, tok, trunk,
           payload: Optional[Dict[str, Any]]) -> int:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    spec = RewardSpec.from_cfg(cfg)

    env = meb.MujocoArm(cfg)
    env.reset()
    policy = TrainablePolicy(cfg, accel, tok, trunk, payload)
    win = meb.StateWindow(cfg.window, tok, policy.qbind, accel.device)
    updater = A2CUpdater(cfg, policy, accel, win)
    guard = updater.guard

    frames = int(cfg.duration * cfg.control_hz) if cfg.duration > 0 else int(cfg.frames)
    if frames <= 0:
        raise SystemExit("nothing to do: set --frames or --duration")

    print("=" * 78)
    print("LIVE MUJOCO RL  (A2C on the physical bridge)")
    print("=" * 78)
    print(meb.sensor_layout_table())
    print(f"  scene       : {meb.NUM_OBJECTS} force-driven spheres + 3-DoF arm + tendon "
          f"gripper · {env.m.nv} dofs · {env.m.nu} actuators")
    print(f"  control     : {cfg.control_hz:g} Hz policy · {cfg.physics_hz:g} Hz "
          f"physics ({env.substeps} substeps/step) · {frames} training frames "
          f"= {frames/cfg.control_hz:.1f}s sim time")
    print(f"  state       : {win.note()}")
    print(f"  policy      : window S={cfg.window} · max_loops={cfg.loops} · "
          f"carry={'on (never reset)' if cfg.carry else 'OFF (ablation)'} · "
          f"{accel.name or accel.device}")
    print(f"  trainable   : {policy.n_trainable/1e6:.3f}M params mode='{cfg.trainable}' "
          f"+ critic {policy.n_critic/1e6:.3f}M")
    print(f"  optimiser   : AdamW lr={cfg.lr:g} (critic {cfg.value_lr:g}) · "
          f"γ={cfg.gamma:g} · β_ent={cfg.entropy_coef:g} · v_coef={cfg.value_coef:g} · "
          f"clip={cfg.clip:g} (policy and critic clipped separately)")
    print(f"  update      : {cfg.rollout} frames/rollout → one A2C step · "
          f"≤{cfg.update_batch} frame(s) per backward (the VRAM knob) · "
          f"adv_norm={'on' if cfg.adv_norm else 'off'}"
          + (" · grad-checkpoint=on" if cfg.grad_checkpoint else ""))
    print(f"  control     : json — the typed answers drive the arm "
          f"({len(policy.bank.specs)} question(s), logπ = Σ_q log P_q) · the 6-way "
          f"token is json_action(answers), telemetry only", flush=True)
    print(f"  episode     : {cfg.episode_frames or '∞'} frames · retarget every "
          f"{cfg.retarget_every or '∞'} frames · eval every "
          f"{cfg.eval_every or '∞'} ({cfg.eval_frames} greedy frames on the fixed "
          f"protocol: ready pose, spheres at start, t=0, fresh carry)")
    print(f"  task        : {cfg.task} (the verb lives in the instruction text; the "
          f"expert that labels live states is task-conditioned) · speech "
          + (f"streamed {int(getattr(cfg, 'speak_tokens', 1))} token(s)/frame, "
             f"≤{cfg.speak_max} per utterance incl. END"
             if int(getattr(cfg, "speak_max", 8)) > 0 else "muted")
          + f" · task_weight={getattr(cfg, 'task_weight', 0.0)}")
    print(spec.table())
    print(f"  transfer    : none — the state is text, so there is no pinned "
          f"frame buffer to stage")
    print(f"  memory      : {accel.note or 'uncapped'} · amp={accel.amp} "
          f"({str(accel.dtype).split('.')[-1]}) · gc-every={cfg.gc_every or 'off'}")

    # ── optional visualisation (same idioms as the bridge) -------------------
    viewer = None
    if not cfg.headless:
        if _VIEWER_IMPORT_ERROR is not None:
            print(f"  viewer      : unavailable ({_VIEWER_IMPORT_ERROR}) — headless")
        else:
            try:
                viewer = mujoco.viewer.launch_passive(env.m, env.d)
                with viewer.lock():
                    viewer.cam.lookat[:] = (0.0, 0.0, 0.30)
                    viewer.cam.distance, viewer.cam.azimuth = 1.75, 135.0
                    viewer.cam.elevation = -20.0
                print("  viewer      : passive window open — close it to stop")
            except Exception as exc:                          # pragma: no cover
                print(f"  viewer      : launch failed ({exc}) — headless")
                viewer = None
    renderer = cam = None
    if cfg.render_out:
        try:
            renderer = mujoco.Renderer(env.m, cfg.render_height, cfg.render_width)
            cam = meb.free_camera(cfg)
            print(f"  render      : {cfg.render_out} "
                  f"({cfg.render_width}x{cfg.render_height}"
                  + (f", every {cfg.render_every} frames" if cfg.render_every
                     else ", final frame only") + ")")
        except Exception as exc:                              # pragma: no cover
            print(f"  render      : offscreen GL unavailable ({exc}) — continuing")
            renderer = None

    # ── first frame + kernel autotune ---------------------------------------
    obs, goal_m = env.observe()
    win.reset(meb.state_sentence(obs))
    sids, apos = win.push(meb.state_sentence(obs))
    dt_warm = policy.warmup(sids, apos)
    dt_bwd = updater.warmup(sids, apos)
    if accel.backend in ("rocm", "cuda"):
        torch.cuda.synchronize(accel.device)
    print(f"  warmup      : forward {dt_warm*1e3:.0f} ms · "
          f"forward+backward {dt_bwd*1e3:.0f} ms (kernel autotune once)")
    text, slot = env.new_instruction(rng, cfg.goal, cfg.task)
    policy.set_instruction(text)
    print(f"  instruction : \"{policy.instruction}\" (colour slot {slot})")

    frozen = tlr.freeze_gc()             # model/scene graph out of the GC scan
    print(f"  gc          : {frozen} long-lived objects frozen · explicit "
          f"collect {'every ' + str(cfg.gc_every) + ' step(s)' if cfg.gc_every else 'off'}"
          f" (a full pass costs ~215-260 ms here)")

    # ── the yardstick: a deterministic, non-invasive protocol eval -----------
    # (the live distance is dominated by the force-driven target's own drift, so
    #  it cannot tell "the policy improved" from "the sphere wandered away")
    probe = EvalProbe(cfg, env, policy, win)
    probe_hist: List[Tuple[str, str, float]] = []      # (when, goal, d_mean)

    # the carry is empty at this point, so the baseline is the untrained policy
    baseline = probe("baseline (before RL)")
    probe_hist.append(("baseline", baseline["goal"], baseline["dist"]))
    # The starting checkpoint is a candidate too: llmm-live v1 went 0.086m →
    # 0.851m through imitation and then saved the 0.851m policy as "best".
    best_eval = baseline["dist"]
    best_is_input = True        # flips once any trained snapshot beats it

    # what the *interface* permits, measured through the same protocol: the
    # expert that labelled the supervised data.  A policy number is meaningless
    # without it — the bridge used to hold one open-loop torque across 100
    # substeps, an interface where "do nothing" scored better than the expert.
    ceiling = probe.expert_ceiling()
    _tr = ceiling["t_reach"]
    print(f"  ceiling     : expert, same protocol · d_mean={ceiling['dist']:.3f}m "
          f"best={ceiling['best']:.3f}m reach={ceiling['reach']:.1f}% "
          f"t_reach={'--' if not np.isfinite(_tr) else f'{_tr:.0f}'}")

    # ── DAgger round 0: fit the live distribution before any policy gradient --
    # Measured: the synthetic warm start scores 0.67 on its own eval and emits a
    # CONSTANT action on the live states (0-2.5 % expert agreement, chance 16.7 %).
    # A2C cannot recover from that — it never samples the action a live state
    # needs — so the expert labels the learner's own live states first.
    imitator = LiveImitator(cfg, policy, accel, env, win)
    # CPU copy of every trainable tensor, so a harmful imitation can be undone
    snap = {n: p.detach().to("cpu", copy=True)
            for n, p in policy.model.named_parameters() if p.requires_grad}
    # buffers too (floating ones only): anything updated in train() mode
    snap_buf = {n: b.detach().to("cpu", copy=True)
                for n, b in policy.model.named_buffers()
                if b is not None and b.is_floating_point()}
    imitator.run()
    if imitator.n > 0:
        after_imit = probe("after imitation")
        probe_hist.append(("imitation", after_imit["goal"], after_imit["dist"]))
        # The imitation is frequently the best policy of the whole run (measured:
        # 0.052m after 3 DAgger rounds vs 0.070-0.132m once A2C refines it), so it
        # has to enter the best-snapshot bookkeeping — otherwise `_best.pt` can end
        # up holding a *worse* policy than the one the imitation produced.
        if after_imit["dist"] < best_eval:
            best_eval, best_is_input = after_imit["dist"], False
            bp = best_path(cfg)
            if bp:
                save_payload(bp, cfg, policy, updater,
                             {"frames": imitator.n, "updates": 0,
                              "eval_dist": after_imit["dist"], "best": True,
                              "stage": "imitation"})
                print(f"  saved       : best eval d={after_imit['dist']:.3f}m "
                      f"(after imitation) → {bp}")
        elif getattr(cfg, "revert_worse_imitation", True):
            # RL from a broken policy never recovered in v1 (d stayed 0.83-0.86m)
            max_diff = 0.0
            with torch.no_grad():
                for n, p in policy.model.named_parameters():
                    if n in snap:
                        p.copy_(snap[n].to(p.device, p.dtype))
                        max_diff = max(max_diff, float(
                            (p.detach().float().cpu() - snap[n].float()).abs().max()))
                buf_changed = 0
                for n, b in policy.model.named_buffers():
                    if n in snap_buf:
                        if not torch.equal(b.detach().cpu(), snap_buf[n]):
                            buf_changed += 1
                        b.copy_(snap_buf[n].to(b.device, b.dtype))
            n_snap = len(snap)
            n_now = sum(1 for _, p in policy.model.named_parameters() if p.requires_grad)
            print(f"  reverted    : imitation made the eval worse "
                  f"({baseline['dist']:.3f}m → {after_imit['dist']:.3f}m); restored "
                  f"{n_snap} tensors (now trainable {n_now}) + {buf_changed} changed "
                  f"buffers · max |Δ| after "
                  f"restore {max_diff:.2e}", flush=True)
            # llmm-live v2: the first RL eval after the revert was still 0.842m.
            # Measure here so "the revert missed state" and "RL broke it again"
            # can be told apart.
            after_rev = probe("after revert")
            probe_hist.append(("revert", after_rev["goal"], after_rev["dist"]))
            if abs(after_rev["dist"] - baseline["dist"]) > 1e-3:
                print(f"  WARNING     : after-revert eval {after_rev['dist']:.3f}m != "
                      f"baseline {baseline['dist']:.3f}m — the revert does not restore "
                      f"everything imitation changed", flush=True)
    del snap, snap_buf

    # ── training loop -------------------------------------------------------
    rlog: deque = deque(maxlen=cfg.log_window)     # per-frame reward/distance/ACT
    alog: deque = deque(maxlen=cfg.log_window)     # per-update A2C diagnostics
    done_log: Counter = Counter()                  # action histogram (whole run)
    t_start = time.time()
    frames_done = ep_frames = 0
    in_reach = False
    peak_gb = 0.0
    next_eval = cfg.eval_every if cfg.eval_every else None
    next_save = cfg.save_every if cfg.save_every else None

    try:
        while frames_done < frames:
            if viewer is not None and not viewer.is_running():
                print("  viewer      : window closed by user")
                break
            t_loop = time.perf_counter()

            # (a) episode boundary: arm reset, new instruction.  The *latent*
            #     carry survives unless the ablation flag says otherwise.
            boundary = False
            if cfg.episode_frames and ep_frames >= cfg.episode_frames:
                env.reset()
                if cfg.reset_carry_on_episode:
                    policy.reset_carry()
                o0, _ = env.observe()
                win.reset(meb.state_sentence(o0))
                ep_frames = 0
                boundary = True

            # (b) retarget: only the instruction changes — the state keeps
            #     flowing and the latent carry is *not* reset.  The rollout is
            #     flushed first: a return spanning a goal change is meaningless,
            #     and the update re-forwards every stored frame with one prompt.
            if cfg.retarget_every and frames_done and frames_done % cfg.retarget_every == 0:
                updater.flush()
                t2, s2 = env.new_instruction(rng, cfg.goal, cfg.task)
                policy.set_instruction(t2)
                print(f"  [{frames_done:5d}] retarget → \"{t2}\" (slot {s2})", flush=True)

            # (c) observe → window → sample an action.  No graph is built here:
            #     the rollout runs under no_grad and the graph is created in
            #     update(), so nothing is alive while AdamW writes parameters.
            obs, goal_m = env.observe()
            sids, apos = win.push(meb.state_sentence(obs))
            ao, carry_in = policy.act(sids, apos)
            if ao.mass_err > 1e-3:                    # ACT invariant guard
                print(f"    WARNING: Σα−1 = {ao.mass_err:.2e} — check the loop")
            if cfg.json_out and frames_done % max(cfg.json_every, 1) == 0 \
                    and getattr(policy, "last_json", None) is not None:
                # Same decision document the bridge writes: the model's own heads
                # plus the harness telemetry.  Written from the LIVE frame (before
                # the update), so it shows the policy that actually acted.
                meb.write_decision(
                    cfg.json_out, policy,
                    {"json": policy.last_json, "cycles": ao.cycles,
                     "ponder": ao.ponder, "latent_norm": ao.latent_norm},
                    goal_m, frames_done, env)
            # The rollout that just filled is finalised here, one frame late:
            # *this* frame's V(s_t) is the bootstrap of the state that follows
            # the rollout's last transition, so no value is ever invented.
            if updater.due:
                info = updater.update(boot=ao.value)
                if info is not None and "gn" in info:
                    alog.append(info)
                updater.buf.reset()
                updater.due = False

            # (d) decision → torque → physics  (τ map shared with the bridge)
            drive(env, ao)
            env.step()

            # (e) consequence: reward from the bridge's own `goal` telemetry
            _, d_next = env.observe()
            r, terms = compute_reward(spec, goal_m, d_next, env.reach,
                                      in_reach, ao.ponder)
            in_reach = d_next < env.reach
            # the segment ends iff the *next* frame starts a fresh episode
            done = bool(cfg.episode_frames and (ep_frames + 1) >= cfg.episode_frames)
            updater.buf.add(sids, apos, carry_in, ao.action, r, done,
                            ao.logprob, ao.value, labels=ao.labels)
            updater.due = updater.buf.full

            frames_done += 1
            ep_frames += 1
            rlog.append({"r": r, "d": d_next, "reach": in_reach, "cyc": ao.cycles,
                         "pov": ao.ponder, "probs": ao.probs, "terms": terms})
            done_log[tlr.ACTION_NAMES[ao.action]] += 1
            guard.flush_if_low()
            guard.reclaim(frames_done)

            # (f) visualisation
            if viewer is not None and viewer.is_running():
                with viewer.lock():
                    viewer.sync()
            if renderer is not None:
                renderer.update_scene(env.d, camera=cam)
                img = renderer.render()
                if cfg.render_every and frames_done % cfg.render_every == 0:
                    stem, ext = os.path.splitext(cfg.render_out)
                    meb.write_png(f"{stem}_{frames_done:05d}{ext or '.png'}", img)
                meb.write_png(cfg.render_out, img)

            # (g) pacing (optional) + logging
            if cfg.realtime:
                slack = (1.0 / cfg.control_hz) - (time.perf_counter() - t_loop)
                if slack > 0:
                    time.sleep(slack)
            if cfg.log_every and (frames_done % cfg.log_every == 0 or frames_done == 1):
                if accel.backend in ("rocm", "cuda"):
                    peak_gb = max(peak_gb, torch.cuda.max_memory_allocated(
                        accel.device) / 2**30)
                    tail = f" peak={peak_gb:.2f}G"
                else:
                    tail = ""
                wr = float(np.mean([f["r"] for f in rlog]))
                wd = float(np.mean([f["d"] for f in rlog]))
                wr_reach = 100.0 * float(np.mean([1.0 if f["reach"] else 0.0
                                                  for f in rlog]))
                wcyc = float(np.mean([f["cyc"] for f in rlog]))
                H = float(np.mean([-(np.log(np.clip(f["probs"], 1e-9, 1))
                                     * f["probs"]).sum() for f in rlog]))
                dec = f" dec={ao.cmd.note}" if ao.cmd is not None else ""
                print(f"  [{frames_done:5d}]{' *' if boundary else '  '}"
                      f"g={goal_m:5.3f}m r={r:+.3f} "
                      f"|{cfg.log_window}f: R={wr:+.3f} d={wd:.3f}m "
                      f"reach={wr_reach:4.1f}% cyc={wcyc:.2f} H={H:.2f} "
                      f"pov={ao.ponder:.2f} m_err={ao.mass_err:.0e}"
                      f"{dec} "
                      f"{(time.perf_counter()-t_loop)*1e3:6.0f}ms/f{tail}", flush=True)
                if alog:
                    m = {k: float(np.mean([a[k] for a in alog]))
                         for k in ("pg", "vf", "adv", "ret", "gn",
                                   "gn_pol", "gn_val")}
                    if ao.labels:                    # json control: show the answers
                        acts = "ans " + " ".join(f"{k}={v}" for k, v in ao.labels.items())
                    else:
                        acts = " ".join(
                            f"{k}:{done_log[k]/max(1,sum(done_log.values())):.2f}"
                            for k in tlr.ACTION_NAMES)
                    print(f"  {'':8s}upd {updater.updates:5d} "
                          f"pg={m['pg']:+.4f} vf={m['vf']:.4f} A={m['adv']:+.3f} "
                          f"R={m['ret']:+.3f} gn={m['gn_pol']:.2f}(pol)/"
                          f"{m['gn_val']:.2f}(val) | {acts}", flush=True)

            # (h) periodic eval + checkpoint
            #     `frames_done < frames`: an eval on the very last frame would
            #     just duplicate the final probe (which runs after the flush, so
            #     it is the more accurate reading)
            if next_eval and frames_done >= next_eval and frames_done < frames:
                next_eval += cfg.eval_every
                # no flush: the probe snapshots/restores the sim, the carry and
                # the window, so the rollout can keep accumulating across it
                m = probe(f"@ {frames_done} frames")
                probe_hist.append((f"{frames_done}f", m["goal"], m["dist"]))
                if m["dist"] < best_eval:
                    best_eval, best_is_input = m["dist"], False
                    bp = best_path(cfg)
                    if bp:
                        save_payload(bp, cfg, policy, updater,
                                     {"frames": frames_done, "updates": updater.updates,
                                      "eval_dist": m["dist"], "eval_reach": m["reach"],
                                      "best": True})
                        print(f"  saved       : best eval d={m['dist']:.3f}m → {bp}",
                              flush=True)
            if next_save and frames_done >= next_save:
                next_save += cfg.save_every
                updater.flush()
                n = save_payload(cfg.out, cfg, policy, updater,
                                 {"frames": frames_done, "updates": updater.updates})
                print(f"  saved       : {n} adapter tensors → {cfg.out}", flush=True)

            if cfg.max_seconds and time.time() - t_start > cfg.max_seconds:
                print(f"  wall-clock budget --max-seconds={cfg.max_seconds:g} reached")
                break
    except KeyboardInterrupt:
        print("\n  interrupted (Ctrl-C)")
    finally:
        updater.flush()                        # never leave a partial rollout
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    # ── final eval + summary ------------------------------------------------
    print("-" * 78)
    final = probe("final (after RL)")
    probe_hist.append(("final", final["goal"], final["dist"]))
    if final["dist"] < best_eval:
        best_eval, best_is_input = final["dist"], False
        bp = best_path(cfg)
        if bp and bp != cfg.out:
            save_payload(bp, cfg, policy, updater,
                         {"frames": frames_done, "updates": updater.updates,
                          "eval_dist": final["dist"], "eval_reach": final["reach"],
                          "best": True})

    wall = time.time() - t_start
    tot = max(1, sum(done_log.values()))
    wr_all = float(np.mean([f["r"] for f in rlog])) if rlog else float("nan")
    wcyc_all = float(np.mean([f["cyc"] for f in rlog])) if rlog else float("nan")
    wpov_all = float(np.mean([f["pov"] for f in rlog])) if rlog else float("nan")
    print("=" * 78)
    print(f"  frames      : {frames_done} training + evals "
          f"({frames_done/cfg.control_hz:.1f}s sim, {wall:.1f}s wall → "
          f"{frames_done/max(wall,1e-9):.2f} Hz, RTF "
          f"{frames_done/cfg.control_hz/max(wall,1e-9):.2f})")
    print(f"  updates     : {updater.updates} AdamW steps "
          f"({cfg.rollout}-frame rollouts, ≤{cfg.update_batch} frame(s)/backward, "
          f"{frames_done} transitions consumed)")
    if alog:
        m = {k: float(np.mean([a[k] for a in alog]))
             for k in ("pg", "vf", "adv", "ret", "gn", "gn_pol", "gn_val")}
        print(f"  A2C         : last {len(alog)} updates · pg={m['pg']:+.4f} "
              f"vf={m['vf']:.4f} |A|={abs(m['adv']):.3f} R={m['ret']:+.3f} · "
              f"grad-norm p/v={m['gn_pol']:.2f}/{m['gn_val']:.2f} (pre-clip)")
    print(f"  distance    : fixed-protocol eval — {cfg.eval_frames} greedy frames "
          f"from the ready pose, spheres at start, t=0 · d0="
          f"{baseline['d_first']:.3f}m at every eval (determinism check)")
    for _i, (_when, _goal, _dm) in enumerate(probe_hist):
        print(f"      {_when:>9} · goal={_goal:<6} d_mean={_dm:.3f}m"
              + ("   ← final" if _i == len(probe_hist) - 1 else ""))
    # only evals run for the SAME goal are comparable: group consecutive runs
    _goal, _vals = None, []
    for _, _g, _d in probe_hist + [(None, None, 0.0)]:      # sentinel flushes
        if _g != _goal:
            if _goal is not None and len(_vals) > 1:
                print(f"      same-goal trend · {_goal:<6} d_mean "
                      f"{_vals[0]:.3f}m → {_vals[-1]:.3f}m "
                      f"(Δ {_vals[-1] - _vals[0]:+.3f} over {len(_vals)} evals)")
            _goal, _vals = _g, []
        if _g is not None:
            _vals.append(_d)
    print(f"  reach       : {baseline['reach']:.1f}% → {final['reach']:.1f}% "
          f"(threshold {env.reach*100:.1f} cm) · best d_mean {best_eval:.3f}m")
    print(f"  vs ceiling  : expert {ceiling['dist']:.3f}m / {ceiling['reach']:.1f}% "
          f"· policy {final['dist']:.3f}m / {final['reach']:.1f}%  "
          f"(gap {final['dist'] - ceiling['dist']:+.3f}m — what is left to learn)")
    print(f"  reward      : last {len(rlog)} frames R={wr_all:+.3f} "
          f"(progress {spec.w_progress:g}·Δd · shaping "
          f"{spec.w_dist:g}·(d_prev−γ·d_now) "
          f"· reach +{spec.reach_bonus:g} · hold +{spec.hold_bonus:g} "
          f"· ponder −{spec.ponder_penalty:g}·p)")
    print(f"  action mix  : " + " ".join(
        f"{k}:{done_log[k]/tot:.2f}" for k in tlr.ACTION_NAMES))
    print(f"  ACT         : mean cycles {wcyc_all:.2f} · ponder {wpov_all:.2f} "
          f"(loop budget={cfg.loops}, act_eps={1e-2})")
    print(f"  memory      : peak {peak_gb:.2f} GiB of the cap · "
          f"OOM events={guard.oom_events + updater.oom_events + imitator.oom_events} · "
          f"allocator flushes={guard.reclaims}")
    n = save_payload(cfg.out, cfg, policy, updater,
                     {"frames": frames_done, "updates": updater.updates,
                      "eval_dist": final["dist"], "eval_reach": final["reach"]})
    print(f"  saved       : {n} adapter tensors ({cfg.trainable}) → {cfg.out}")
    bp = best_path(cfg)
    if bp:
        if best_is_input:
            print(f"  best        : NOTHING beat the input checkpoint "
                  f"(baseline d_mean {best_eval:.3f}m vs final {final['dist']:.3f}m) "
                  f"— keep using {cfg.checkpoint}")
        elif best_eval < final["dist"] - 1e-9:
            print(f"  best        : d_mean {best_eval:.3f}m — a better snapshot than the "
                  f"final policy ({final['dist']:.3f}m) is on disk at {bp}")
        else:
            print(f"  best        : {bp} holds this same final policy")
    else:
        print("  best        : snapshot disabled (--best-out none)")
    replay = (cfg.checkpoint if best_is_input else
              bp if (bp and best_eval < final['dist'] - 1e-9) else cfg.out)
    print(f"  replay it   : python mujoco_env_bridge.py --model {cfg.model} "
          f"--trainable {cfg.trainable} --checkpoint {replay}")
    return 0


def best_path(cfg: argparse.Namespace) -> str:
    """Where the best-eval snapshot goes — "" disables it entirely."""
    if not cfg.best_out or str(cfg.best_out).lower() in ("none", "off", "0"):
        return ""
    return cfg.best_out


# =============================================================================
# 10 · CLI  (the bridge's parser + an RL group, so every HW guard stays in sync)
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = meb.build_parser()
    p.description = ("Live MuJoCo RL: A2C policy-gradient fine-tuning of the "
                     "Looped Transformer's adapter against the physics engine.")
    g = p.add_argument_group("live RL")
    g.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    g.add_argument("--entropy-coef", type=float, default=0.01,
                   help="β: keeps the action head from collapsing to one token")
    g.add_argument("--value-coef", type=float, default=0.5,
                   help="weight of the critic regression term")
    g.add_argument("--value-lr", type=float, default=0.0,
                   help="critic learning rate (0 = same as --lr)")
    g.add_argument("--critic-hidden", type=int, default=512)
    g.add_argument("--rollout", type=int, default=16,
                   help="frames collected per A2C update (one batched loss; no "
                        "graph ever survives the call that created it). "
                        "Measured on the 880M with a 0.5B trunk: 16 frames with "
                        "--update-batch 4 peaks at 3.8 GiB of the 8 GiB budget")
    g.add_argument("--update-batch", type=int, default=4,
                   help="frames per forward/backward inside an update — the VRAM "
                        "knob for a 0.5B trunk (gradients still accumulate over "
                        "the whole rollout before the single AdamW step)")
    g.add_argument("--adv-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="normalise the advantage for the policy term only (the "
                        "critic always regresses the raw return)")
    g.add_argument("--imitate-rounds", type=int, default=3,
                   help="DAgger round-0 rounds before RL (0 = off).  Needed "
                        "because the synthetic warm start emits a CONSTANT action "
                        "on live states: measured expert agreement 0.0 (600 "
                        "supervised steps), 0.0 (2400 steps, 0.67 synthetic "
                        "accuracy) and 2.5 after 1200 RL frames, against a 16.7 "
                        "chance level.  A constant policy cannot be repaired by "
                        "A2C, so the expert labels the learner's own live states")
    g.add_argument("--imitate-frames", type=int, default=300,
                   help="learner-driven live frames collected per imitation round "
                        "(each one labelled by the expert for free)")
    g.add_argument("--imitate-steps", type=int, default=150,
                   help="cross-entropy steps fitted per imitation round")
    g.add_argument("--imitate-batch", type=int, default=16,
                   help="windows per imitation backward (the VRAM knob: batch 16 "
                        "x window 8 on a 0.5B trunk stays well inside the 8 GiB cap)")
    g.add_argument("--imitate-lr", type=float, default=0.0,
                   help="learning rate for the imitation fit (0 = use --lr)")
    g.add_argument("--imitate-accum", type=int, default=1,
                   help="micro-batches of --imitate-batch accumulated per imitation "
                        "step (effective batch = batch x accum, same VRAM as one)")
    g.add_argument("--imitate-acc-windows", type=int, default=256,
                   help="pool windows (fixed random sample) scored for the "
                        "per-round json_action accuracy (0 = skip)")
    g.add_argument("--freeze-lm-head", action="store_true",
                   help="keep the untied LM head (the answer read-out, ~136M on "
                        "0.5B) out of imitation and RL — only the small modules "
                        "(+ LoRA with --trainable lora) move")
    g.add_argument("--keep-worse-imitation", dest="revert_worse_imitation",
                   action="store_false",
                   help="do NOT restore the pre-imitation weights when the "
                        "after-imitation eval is worse than the baseline")
    g.add_argument("--episode-frames", type=int, default=300,
                   help="frames per episode (arm reset; 0 = never)")
    # NOTE: `--goal` is NOT redefined here — it comes from the bridge parser
    # (`meb.build_parser`), which this parser extends.  One definition, so the
    # bridge demo and the RL run can never disagree about what it means.
    g.add_argument("--reset-carry-on-episode", action="store_true",
                   help="ABLATION: wipe the latent carry at each reset "
                        "(the architecture forbids this — off by default)")
    g.add_argument("--eval-frames", type=int, default=40,
                   help="greedy frames per eval — run on the fixed protocol "
                        "(ready pose, spheres at their start poses, t=0, fresh "
                        "carry), then the training state is restored")
    g.add_argument("--save-every", type=int, default=500,
                   help="checkpoint cadence in frames (0 = only at the end)")
    g.add_argument("--best-out", default="",
                   help="path for the best-eval snapshot (default <out>_best.pt; "
                        "'none' disables)")
    g.add_argument("--w-progress", type=float, default=20.0,
                   help="reward per metre closed on the instructed sphere")
    g.add_argument("--w-dist", type=float, default=1.0,
                   help="weight on potential-based shaping, Φ(s) = −d(s): "
                        "w·(d_prev − γ·d_now).  Telescopes to (d0 − d_end) and "
                        "so cannot change the optimal policy — it only makes "
                        "the distance gradient dense (Ng et al. 1999)")
    g.add_argument("--reach-bonus", type=float, default=2.0,
                   help="paid EVERY frame the tool sits inside --reach, which "
                        "is what makes 'keep tracking the sphere' the payoff "
                        "and 'arrive and stop' strictly worse than arriving early")
    g.add_argument("--hold-bonus", type=float, default=0.5,
                   help="extra per frame that stays inside reach")
    g.add_argument("--ponder-penalty", type=float, default=0.02,
                   help="ACT compute cost per ponder unit (as in τ·ponder)")
    g.add_argument("--reward-clip", type=float, default=5.0,
                   help="per-frame reward clamp (keeps a physics glitch from "
                        "blowing up a γ-return)")
    g.add_argument("--log-window", type=int, default=50,
                   help="frames averaged in the rolling log/eval metrics")
    return p


def apply_rl_defaults(cfg: argparse.Namespace) -> None:
    """
    RL-specific defaults for flags that are shared with the supervised trainer.
    Only applied when the user did not type them, so the CLI stays honest.
    """
    typed = meb._flags_present(sys.argv)
    if "--out" not in typed:
        cfg.out = "robot_live.pt"
    if "--clip" not in typed:
        # measured: pre-clip grad-norms on the 6.87M adapter run 12-16, so the
        # supervised clip of 1.0 scaled every step down ~13× (the policy barely
        # moved in 600 frames).  Clip 5.0 leaves the step in control.
        cfg.clip = 5.0
    if "--lr" not in typed:
        # on-policy PG from a random policy needs a larger step than supervised
        # fine-tuning; from a checkpoint stay at the fine-tuning rate.
        cfg.lr = 1e-4 if cfg.checkpoint else 3e-4
        print(f"[rl] --lr defaults to {cfg.lr:g} with --clip {cfg.clip:g} "
              f"({'warm start from a checkpoint' if cfg.checkpoint else 'cold start'}"
              "; pass --lr/--clip to override)")
    if "--trainable" not in typed:
        cfg.trainable = "recurrent"        # 6.87M weights on Qwen2.5-0.5B
        where = "0.11M on the tiny trunk" if cfg.model == "tiny" \
            else "6.87M on a 0.5B trunk"
        print(f"[rl] --trainable defaults to 'recurrent' here ({where}); "
              "pass --trainable adapters for the head-only variant, or "
              "--trainable lora --lora-rank 16 to adapt the whole trunk")
    if not cfg.value_lr:
        cfg.value_lr = cfg.lr
    cfg.rollout = max(1, cfg.rollout)
    cfg.update_batch = max(1, min(cfg.update_batch, cfg.rollout))
    cfg.imitate_rounds = max(0, cfg.imitate_rounds)
    cfg.imitate_frames = max(0, cfg.imitate_frames)
    cfg.imitate_steps = max(0, cfg.imitate_steps)
    cfg.imitate_batch = max(1, cfg.imitate_batch)
    if not cfg.best_out:
        stem, ext = os.path.splitext(cfg.out)
        cfg.best_out = f"{stem}_best{ext or '.pt'}"
    if cfg.trainable == "full":
        print("[rl] WARNING: --trainable full needs ≈8 GB of AdamW state for a "
              "0.5B trunk — prefer 'recurrent' on an 8 GB carve-out")
    if not cfg.headless and cfg.realtime:
        print("[rl] note: --realtime caps the loop at --control-hz; the optimiser "
              "step will not fit — expect the viewer to run in slow motion")


def main() -> int:
    cfg = build_parser().parse_args()
    apply_rl_defaults(cfg)
    payload = meb.load_checkpoint(cfg)          # adopts the checkpoint's arch
    accel = tlr.verify_hardware(cfg)
    if cfg.self_test:
        print("=" * 78)
        print("SELF-TEST MODE — no learning, no checkpoints written")
        return run_self_test(cfg, accel)
    tok, trunk = tlr.load_base(cfg.model, accel)
    return run_rl(cfg, accel, tok, trunk, payload)


if __name__ == "__main__":
    raise SystemExit(main())
