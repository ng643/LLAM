#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loop_robot.py
================================================================================
Continuous-latent **Looped Transformer** + **ACT halting head** + **gated
multi-modal sensor injection**, trained as a discrete-action robot policy on
synthetic proprioceptive/geometric data.

Target: Windows 11 · Ryzen AI 9 365 · Radeon 880M (gfx1150, 8 GB UMA carve-out).

--------------------------------------------------------------------------------
DATA FLOW (single training window = `--window` control steps)
--------------------------------------------------------------------------------
  instruction+schema (B,Ti) ─► trunk.embed_tokens ────────────┐  discrete text
  state sentences + markers ─► trunk.embed_tokens (B,S*L,H) ──┤  state steps
  carry (B,H) prev latent─► carry_norm + tag ─► (B,1,H) ───────┤  memory token
  speech tokens (B,Ts)   ─► trunk.embed_tokens (teacher-forced)┘  AFTER state
                                   │  concat → h (B,T,H), T = 1 + Ti + S*L + Ts
                                   ▼
      ┌─ PREAMBLE: trunk.layers[0 .. L-R-1]  ── runs ONCE ────────────────────┐
      └────────────────────────────┬─────────────────────────────────────────┘
                                   ▼  h_t  (plain, never reset, never re-embedded)
      ┌─ ACT LOOP, cycle t = 1..max_loops ───────────────────────────────────┐
      │   h_t = h_{t-1} + cycle_emb[t]                  step tag (zero-init) │
      │   h_t = recurrent_block(h_{t-1})                R shared trunk layers│
      │   h_t = h_t + tanh(gate)·CrossAttn(h_t, prompt)  STATE region only  │
      │   p_t = σ(halt_head(h_t))                       per-position halt    │
      │   α_t = p_t  (or 1-Σp if this cycle overshoots)  convex weights      │
      └────────────────────────────┬─────────────────────────────────────────┘
                                   ▼  z = Σ_t α_t·h_t + residual·h_T
                       trunk.norm ─► gather marker slots (h_ans)
                                          ├─► + task_proj(pooled prompt)
                                          │     └─► h_ans @ W[Jev label tokens] ─► JSON
                                          │                 └─► json_action ─► motor
                                          └─► LM head at last state pos ─► speech
                                   └──────────► ponder = updates + residual

  * The latent state h is a *continuous trajectory*: no token is emitted inside
    the loop and nothing is reset when an action is produced — the last mixed
    latent is handed to the next control window as the `carry` memory token
    (truncated BPTT across windows in the trainer).
  * Every add-on starts inert (`gate=0`, `cycle_emb=0`, `carry_tag=0`), so at
    step 0 the gated injection contributes *exactly* nothing and the pretrained
    trunk weights are not perturbed.  Only the gate is zeroed, never the
    cross-attention output projection as well: two zero factors in series cancel
    the gradient in both directions and the fusion path can never turn on
    (the `--self-test` asserts the gate's gradient is non-zero at init).
  * The prompt (instruction + SCHEMA) enters through that same gate, as the
    K/V of the cross-attention.  It has to: the question read-out reads the
    STATE region, and with a frozen trunk the only route from the text to that
    region is the trunk's own attention — untrained.  Measured on a trained
    tiny checkpoint: swapping "move to the red block" for "move to the blue
    block" moved the state-slot hidden states by 8.3e-02 but the read-out
    logits by only 6.4e-04, and live every instruction produced the same `-x`
    command.  The gated cross-attention gives the sentence a trainable route
    into exactly the slots the answers read, and it is zero-init, so
    checkpoints written before it existed behave identically.  The
    route is *trained* by `--task-weight`: an auxiliary CE that names the verb of
    the prompt.  It is the only loss that the state path cannot reduce (the
    observation is identical for all four verbs until contact), so its gradient
    has to travel through the fusion — `--self-test` proves the point by giving
    two different instructions the SAME observation and asserting the head
    separates them (CE 1.37 → 0.02, both argmaxes right).  At inference the
    same logits are reported as `task_inferred` in the JSON document, so a
    mismatch with the instruction is visible per frame.
  * There are TWO prompt routes, and both are needed.  `fusion.task_attn` is the
    gated one inside the loop (it shapes the hidden trajectory); `task_proj`
    adds the pooled prompt latent straight onto the latent read at EVERY answer
    marker (h_ans) before the LM-head option scoring, and onto the step read-out
    the verb head sees — so the instruction reaches the JSON answers directly.
    The gated route alone was measured to be too weak to steer the arm: after
    3000 steps its gate sat at tanh ≈ −0.13 and every instruction still produced
    the same command, while the verb head — which only needs a small
    prompt-dependent nudge — had already learned the sentence (CE 0.002).

--------------------------------------------------------------------------------
JSON DECISION INTERFACE (typed questions, answered inside ONE forward)
--------------------------------------------------------------------------------
  The policy's output is not a decoded string: it is a set of typed questions,
  each answered from the latent at its own marker slot.  The motor command is
  DERIVED from those answers (`json_action`, a fixed decoder — there is no
  separate action head).  The JSON is then *rendered* from the distributions —
  free, exact, and impossible to malform, because no answer token is decoded.
  This is
  the mechanism the Jev family of "system-one" classifiers uses (simple-jev:
  "the model does not generate a JSON completion: the server constructs the
  response from the scores"; robo-jev runs a 2B backbone this way at 10 Hz,
  p95 62.7 ms, generating no text at all).

      z (B,S,Q,H) at markers + task_proj(prompt) ─► h_ans @ W[labels]ᵀ ─► per-question logits
      per-question arg-maxes ─► json_action ─► one of -x,+x,-y,+y,grasp,brake
      z at last state pos ─► LM head ─► first speech token (END = silent)

  Question types (`QuestionBank` / `QuestionSpec`, `--questions auto|off|<json>`):
    choice : K options        -> {"choice": ..., "probabilities": {...}, "confidence": p}
    score  : ordered levels   -> {"score": 1.092, "legend": [...]}  (expected value)
    noul   : yes/no           -> {"noul": 0.83}         (probability of yes)
  The schema is INPUT, and the answers are read OUT of the model's own next-token
  distribution: `collate` renders the questionnaire into the prompt and appends
  one marker token per question to every state sentence, and the answer to
  question q is `h[marker_q] @ W[label tokens]ᵀ` — the LM head restricted to
  that question's Jev LABELS: choice options are labelled A, B, C, … and score
  levels 0, 1, 2, … (`- x: Which way along x? options: A=negative B=stay
  C=positive`); noul keeps its own no/yes.  `QuestionBinding` refuses a bank
  whose labels are not single, distinct tokens.  The JSON still reports the
  original option NAMES.  There are no per-question parameters (`W` is the
  tied embedding matrix, or the untied `lm_head`), so sending a different
  questionnaire really does change the answer space.  Add an `unknown` option
  to a question whose answer may not exist in the state — that is the portable
  way to let the model decline a field.  Labels come from the expert
  trajectory (`ACTION_NAMES` actions → `LABEL_SOURCES`, see
  `derive_question_labels`), so the interface needs no new data collection,
  and `json_action` is the exact inverse of those label sources.

  SPEECH (`--text-weight` > 0): the sequence is [carry][instruction + schema]
  [state steps + markers][speech].  The LM head at the last position of the
  last state step predicts the first token the model says this frame; the
  utterance is the sentence followed by END (tokenizer eos), and a silent
  frame's target is END alone.  Speech is STREAMED: `--speak-tokens K` (1|2)
  tokens per control frame, spread over consecutive frames by
  `stream_schedule` (events queue, at most one waiting).  Training feeds the
  already-spoken prefix (+ within-frame teacher forcing) after the state and
  supervises only this frame's K tokens; the mask is causal, so the speech
  can never move an answer or the carry (self-test).  At inference each
  frame's single forward carries the spoken prefix, its last position gives
  the first new token (silence costs nothing), K=2 costs one extra forward,
  and `--speak-max` (incl. END) caps an utterance.  With a TIED head the
  matrix IS the frozen `embed_tokens`, so the token loss can only be steered by
  moving the latent and it plateaus (txt 6.478 vs ln(1024) = 6.93 after 2400
  steps).  `--untie-lm-head` gives the token path its own trainable
  `Linear(hidden, vocab)` (65k params on the tiny trunk, 136M on Qwen2.5-0.5B)
  — that is what makes speech actually trainable under `--trainable adapters`.

  TASK FAMILY (`TASKS` / `TASK_TEMPLATES` / `SPEECH_TEMPLATES`): the same 28-D
  observation and the same 6 low-level actions serve four verbs — reach, grasp,
  throw, push — selected by the instruction text alone.  The expert is
  task-conditioned (`SensorWorld.episode`), and every step carries a speech
  target from its EVENT (`detect_speech_event`): "grasped the red block",
  "released the red block", "reached the red block", "pushing the red block",
  silent otherwise.  A new task is therefore a new sentence, not a new head:
  that is the property the JSON interface exists to buy.

  HEIGHT (`tool_z` + `rel_z` + the `height` answer): the state is 3-D.  The
  sensor vector carries the tool's own height and every object's height offset,
  the world integrates `tool_z` by `HEIGHT_VECS[height]`, and the expert's
  descend/climb decision is supervised as the `height` question — not as a
  seventh action token, because the 6-token table is planar and expanding it
  would invalidate the reward, the expert and every measurement taken on it.
  The harness reads the answer and moves its height servo one step per decision
  (bridge `HEIGHT_STEP_M`), which is what actually closes the vertical gap:
  before this, `goal_m` read 1.6 cm while the tool hovered 26 cm above the ball.


  # 1) drop any CPU/CUDA/DirectML build first (mixing them corrupts the wheel)
  python -m pip uninstall -y torch torchvision torchaudio torch-directml
  python -c "import torch"               # must now raise ModuleNotFoundError

  # 2) the Windows wheel index lives on repo.amd.com.  repo.radeon.com only
  #    serves the Linux ROCm packages -- https://repo.radeon.com/rocm/whl-multi-arch/
  #    answers HTTP 404.  Newest cp311 / win_amd64 wheel on that index:
  #      torch-2.12.0+rocm7.14.1-cp311-cp311-win_amd64.whl
  #    (2.11.0 / 2.10.0 also exist with +rocm7.13.0 / 7.14.0 / 7.14.1 tags)
  #
  #    Install torch ALONE: the extra resolves the ROCm runtime itself and pulls
  #    rocm-sdk-libraries / rocm-sdk-core / rocm-sdk-device-gfx1150 /
  #    amd-torch-device-gfx1150.  Adding torchvision/torchaudio only gives the
  #    resolver a chance to drag torch back down to whatever version their pins
  #    demand -- and this script imports neither.
  python -m pip install --no-cache-dir ^
      --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1150]"
  python -m pip install --upgrade --index-url https://pypi.org/simple/ ^
      transformers safetensors numpy
  #   (transformers/safetensors live on PyPI -- the AMD index has no PyPI mirror)
  #   per-target index (gfx1150-only build, no extra syntax needed):
  #     --index-url https://repo.amd.com/rocm/whl/gfx1150/ torch
  #   nightly TheRock:  https://nightly.repo.amd.com/rocm/whl-next/

  # 3) verify in one line: prints the 880M + a non-None hip version
  python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  VERIFIED ON (this machine): ASUS Vivobook S 15 M5506WA · Ryzen AI 9 365 ·
  Radeon 880M · driver 32.0.31041.1004 · Python 3.11.9 (global install, no venv)
    torch 2.12.0+rocm7.14.1 · hip 7.14.60850 · transformers 5.17.0 · numpy 2.4.6
    torch.cuda.is_available()=True · name='AMD Radeon(TM) 880M Graphics'
    arch=gfx1150 · is_integrated=1 · 6 CUs · warp=32
    --self-test 21/21 PASS · Qwen2.5-0.5B training (recurrent, batch 2): 2.0 s/it
    on the first (kernel autotune) step then ~0.23 s/it, peak 3.10 GiB of the
    8 GiB cap, 0 OOM events · the same trunk with `--untie-lm-head
    --text-weight 0.5 --task-weight 0.5 --q-weight 2.0` (the speech + interface
    stack, 177.67M trainable): 0.52 s/it, peak 5.43 GiB, 0 OOM events, 10.5 min
    for 1200 steps · tiny trunk training ~0.14 s/it
  Two SDPA warnings are expected and harmless on this build ("Mem Efficient
  attention on Current AMD GPU is still experimental" / "Flash Efficient ..."):
  SDPA still selects a working backend.  Do NOT silence them with
  $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 — measured here that makes the
  first forward raise `torch.AcceleratorError: CUDA error: invalid argument`
  (hipErrorInvalidValue): the experimental AOTriton kernels are broken for
  gfx1150 in this wheel, not merely unvalidated.

  PyTorch exposes HIP through the `torch.cuda` API, so `torch.cuda.is_available()`
  is the correct probe on the 880M. `torch.version.hip` is non-None on that build.
  If the probe is False you are running a CPU/CUDA-only wheel: fix the wheel
  (above), or run with `--device cpu` / `--model tiny --self-test`.

  HSA_OVERRIDE_GFX_VERSION is set *only* on request (`--gfx-override 11.0.0`),
  because it lies to the runtime about the ISA: the gfx1150 wheels above already
  carry real gfx1150 code objects, and forcing gfx1100 there costs kernels and
  can produce hard faults instead of fixes.  Use it when the wheel is generic.
  A kernel crash is easiest to localise with $env:HIP_LAUNCH_BLOCKING=1 (slow,
  but the failing launch is then the call that raises).

  MEASURED PITFALL -- the reported pool is NOT the carve-out.  On the 880M
  `torch.cuda.get_device_properties(0).total_memory` reports 16.86 GiB: that is
  the shared system-RAM pool HIP is allowed to map (is_integrated=1), not the
  8 GB BIOS UMA buffer, and mem_get_info() agrees (16.71/16.86 GiB free).  A bare
  `--vram-fraction 0.85` ceiling would therefore permit 14.3 GiB and protect
  nothing, so the allocator guard is now
      min(--vram-fraction x reported pool, --vram-budget-gib)
  and the default --vram-budget-gib 8.0 is the number that actually keeps the
  desktop compositor alive (measured peak for 20 Qwen-0.5B steps: 3.56 GiB).

  MEASURED PITFALL -- in a per-frame loop the bottleneck was the Python GC, not
  the GPU.  `gc.collect()` costs 215-260 ms in this process (~449k tracked
  objects: torch modules, the HF trunk, the MuJoCo model) and MemoryGuard used to
  call it every frame, so the MuJoCo bridge ran at 3.1 Hz with 78% of each frame
  inside the collector.  With the pass gated off (`--gc-every 0`, the default) and
  `freeze_gc()` run once at startup (long-lived objects moved to the permanent
  generation, after which a full pass costs 0.00 ms) the same loop runs at
  13-15 Hz for the tiny trunk and 7.6 Hz for Qwen2.5-0.5B.  Training, where a
  batch is real GPU work, may still collect -- every --gc-every batches, if set.

  MEASURED PITFALL -- `--model tiny` used to be a DIFFERENT trunk every process.
  `AutoModel.from_config` draws its init from the global torch RNG, and nothing
  seeded it before the trunk was built, so the same checkpoint landed on fresh
  random frozen weights on every run.  Two identical CPU replays of one
  checkpoint scored 90.0 % and 0.0 % of frames in reach, and their generated
  narration differed on the first token ("454 454 454" vs "759 759 454") even
  though the sampler is greedy — that was the trunk, not the sampling.  It also
  meant a checkpoint's adapters were fitted to a trunk no later process could
  rebuild.  `build_tiny_trunk` now builds under `TINY_TRUNK_SEED` inside
  `torch.random.fork_rng` (the caller's RNG stream is restored), so `--model
  tiny` names one trunk everywhere: three identical replays now report the same
  80.0 % / 0.050 m and the same narration.  Checkpoints written before this fix
  were fitted against an unseeded trunk; retrain them (`--model tiny`) if you
  want train and replay to agree exactly.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python train_loop_robot.py --model tiny --self-test         # no download, ~30 s
  python train_loop_robot.py --model Qwen/Qwen2.5-0.5B
  python train_loop_robot.py --model HuggingFaceTB/SmolLM2-360M --trainable recurrent
  python train_loop_robot.py --batch 2 --window 16 --grad-checkpoint   # if OOM

  Checkpoints hold only the trainable tensors (frozen trunk weights are never
  written, so the file stays small). Resume into a freshly built model:
      sd = torch.load("robot_act.pt", map_location="cpu")     # dict
      model.load_state_dict(sd["trainable_state"], strict=False)
      # -> unexpected: []   and no *trainable* name reported missing:
      #    everything "missing" is frozen trunk state, by design.
================================================================================
"""
from __future__ import annotations

# =============================================================================
# 0 · ENVIRONMENT GUARDS — MUST be set before `import torch`
# =============================================================================
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Caching-allocator policy.
#   garbage_collection_threshold / max_split_size_mb: portable, prevent the
#     fragmentation spikes that turn a transient peak into a WDDM TDR (frozen
#     desktop) on a UMA iGPU.
#   expandable_segments: the best anti-fragmentation option, but it is NOT
#     accepted on Windows -> only enabled on POSIX. Export it manually if your
#     HIP build supports it on Windows.
_ALLOC = "garbage_collection_threshold:0.9,max_split_size_mb:256"
if os.name != "nt":
    _ALLOC = "expandable_segments:True," + _ALLOC
for _v in ("PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
    os.environ.setdefault(_v, _ALLOC)

# MIOpen: bounded conv autotune (FAST) + persistent kernel DB, so the very first
# training step does not stall for minutes on an iGPU.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.join(_HERE, ".miopen_cache"))
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "0")     # keep async launches
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ── HSA_OVERRIDE_GFX_VERSION: opt-in ISA override, read by the HIP runtime -----
# The ROCm runtime reads this *when the HIP library is loaded*, i.e. before the
# first `import torch`, so it cannot be handled by argparse afterwards.  We peek
# at sys.argv here (stdlib only) to honour `--gfx-override 11.0.0` / `=11.0.0`.
# Default OFF: the gfx1150 wheels for the 880M ship real gfx1150 code objects,
# and overriding them to gfx1100 removes kernels and can hard-fault the GPU.
# Turn it on only when a *generic* wheel refuses the integrated chip.
def _argv_value(flag: str) -> "Optional[str]":
    """Read `--flag value` or `--flag=value` from sys.argv without argparse."""
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_GFX_OVERRIDE = _argv_value("--gfx-override") or ""
if _GFX_OVERRIDE.strip().lower() not in ("", "none", "off", "0", "auto"):
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = _GFX_OVERRIDE.strip()
    print(f"[env] HSA_OVERRIDE_GFX_VERSION={os.environ['HSA_OVERRIDE_GFX_VERSION']} "
          f"(requested via --gfx-override)", flush=True)

import argparse                                                     # noqa: E402
import gc                                                           # noqa: E402
import inspect                                                      # noqa: E402
import json                                                         # noqa: E402
import math                                                         # noqa: E402
import platform                                                     # noqa: E402
import random                                                       # noqa: E402
import time                                                         # noqa: E402
import zlib                                                         # noqa: E402
from dataclasses import dataclass, field                            # noqa: E402
from typing import (Any, Callable, Dict, List, Optional, Sequence,  # noqa: E402
                    Tuple)

import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

# =============================================================================
# 1 · ACTION SPACE / SCENE CONSTANTS
# =============================================================================
NUM_ACTIONS = 6
ACTION_NAMES = ("-x", "+x", "-y", "+y", "grasp", "brake")
#                       0        1       2       3       4        5
ACTION_VECS = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0],
                            [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

COLORS = ("red", "green", "blue", "yellow")            # object slot i == COLORS[i]
INSTRUCTION_TEMPLATES = (
    "reach the {c} block",
    "go to the {c} block",
    "move to the {c} block",
    "grab the {c} block",
)

# ── task family ──────────────────────────────────────────────────────────────
# The low-level interface stays 6 discrete actions; the TASK lives in the
# instruction text.  Same geometry, different action sequence after arrival:
# `reach` brakes on the object, `grasp` closes and holds, `throw` closes and
# then departs, `push` never stops.  The instruction is therefore load-bearing
# — the sensor vector cannot tell the four apart — which is exactly the
# property a JSON-driven policy needs: a new task is a new sentence, and the
# typed answers say what to do about it.
TASKS = ("reach", "grasp", "throw", "push")
TASK_TEMPLATES = (
    ("reach the {c} block", "go to the {c} block", "move to the {c} block"),
    ("pick up the {c} block", "grasp the {c} block", "grab the {c} block"),
    ("throw the {c} block", "toss the {c} block", "hurl the {c} block"),
    ("push the {c} block", "shove the {c} block", "nudge the {c} block"),
)
# ── speech: what the policy SAYS after it has read the state ────────────────
# Event-driven and short.  A control step is SILENT (speech target = the end
# token alone) unless the expert trajectory has an event at that step; then the
# target is one sentence from this table followed by the end token.  Events are
# detected in `SensorWorld.episode` (synthetic) and `detect_speech_event`
# (shared with the MuJoCo live loop), always for the step being answered.
SPEECH_TEMPLATES: Dict[str, str] = {
    "grasped": "grasped the {c} block",     # jaw closes on the target (grasp/throw)
    "released": "released the {c} block",   # jaw opens while holding (grasp/throw)
    "reached": "reached the {c} block",     # enters the reach radius (reach)
    "pushing": "pushing the {c} block",     # enters the reach radius (push)
}
SPEECH_EVENTS = tuple(SPEECH_TEMPLATES)


def detect_speech_event(task: int, closes: bool, opens: bool, arrive: bool) -> str:
    """
    The ONE event a step is narrated with ('' = silent).  `closes`/`opens` are
    the jaw transitions caused by this step's expert action (0->1 / 1->0),
    `arrive` means the goal distance entered the reach radius at this step.
    Priority: released > grasped (grasp/throw only) > reached (reach) / pushing
    (push).  A `reach` expert also closes the jaw at arrival — that is a detail
    of the expert, not what a reach is about, so it is narrated as `reached`.
    """
    if task in (1, 2):
        if opens:
            return "released"
        if closes:
            return "grasped"
        return ""
    if arrive:
        return "reached" if task == 0 else "pushing"
    return ""


def speech_sentence(event: str, colour: str) -> str:
    return SPEECH_TEMPLATES[event].format(c=colour) if event else ""

# Sensor vector layout (kept derived, never hand-counted):
#   proprio    : pos(2) + vel(2) + gripper(1) + tool_z(1)         = 6
#   objects    : n_obj * [rel_xy(2) + rel_z(1) + dist_xy(1)]      = 16
#   prev action: one-hot(NUM_ACTIONS), all-zero at t=0            = 6
# `tool_z` and `rel_z` are the HEIGHT channel: the planar action table cannot
# express "descend onto it", so the third axis travels as its own question
# (`height` in DEFAULT_QUESTIONS) instead of as a seventh action token.
SENSOR_DIM = 2 + 2 + 1 + 1 + len(COLORS) * 4 + NUM_ACTIONS   # 28

HEIGHT_NAMES = ("down", "stay", "up")                # the z answer space
HEIGHT_VECS = torch.tensor([-0.35, 0.0, 0.35])       # tool_z step per control step
OBJ_Z_CHOICES = (0.0, 0.25, 0.5)                     # object heights, world units
Z_TOL = 0.05                                         # "level with it" tolerance

# ── the state, as text ───────────────────────────────────────────────────────
# The model's INPUT is a sentence per control step, not a float vector — the
# Jev shape: a text state + a questionnaire, answered inside one forward.  The
# sentence is rendered with fixed-width fields (`%+.2f`) so one step always
# tokenizes to the SAME number of tokens, and every step ends with
# `STATE_MARKER`: that token is the read-out anchor the typed heads answer
# from (Jev reads logits at the question position; here the questionnaire is
# the head bank, so the marker is where the answer is read).
#
#   tool +0.41 +0.88 +0.77 vel -0.02 +0.01 grip 0.00 obj red +0.12 -0.44 -0.72 green …
#
# `SENSOR_DIM` below is still the numeric layout the synthetic EXPERT and the
# label sources read — it is a bookkeeping/rendering source, never an input to
# the network.  Any robot that can print this sentence can drive the policy.
STATE_MARKER = "<|im_end|>"                          # end of one state block
STATE_KEYS = ("tool", "vel", "grip", "obj")          # the schema, in order


def render_state(tool, vel, grip, rel) -> str:
    """One control step as text (fixed width ⇒ constant token count).

    tool: (x,y,z) · vel: (vx,vy) · grip: scalar 0/1 · rel: n_obj × (dx,dy,dz)
    """
    return ("tool " + " ".join(f"{float(v):+.2f}" for v in tool)
            + " vel " + " ".join(f"{float(v):+.2f}" for v in vel)
            + f" grip {float(grip):.2f} obj "
            + " ".join(f"{c} " + " ".join(f"{float(v):+.2f}" for v in rel[i])
                       for i, c in enumerate(COLORS)))


# =============================================================================
# 2 · HARDWARE VERIFICATION & SAFE DEVICE BINDING
# =============================================================================
@dataclass
class Accelerator:
    """Everything the trainer needs to know about the compute device."""
    device: torch.device
    backend: str            # rocm | cuda | directml | cpu
    name: str
    total_gb: float
    dtype: torch.dtype      # AMP compute dtype
    amp: bool
    scaler: bool            # GradScaler needed (fp16 only)
    arch: str = ""
    note: str = ""
    attn_impl: str = "sdpa"     # attention backend the trunk loads with
    gfx_override: str = ""      # HSA_OVERRIDE_GFX_VERSION actually exported

    @property
    def is_gpu(self) -> bool:
        return self.backend in ("rocm", "cuda", "directml")

    @property
    def amp_device_type(self) -> str:
        # torch.autocast takes a device *type*; privateuseone (DirectML) has none.
        return "cuda" if self.device.type == "cuda" else "cpu"


def _rocm_install_hint() -> str:
    return (
        "  Windows wheels that expose the Radeon 880M (gfx1150) through torch.cuda:\n"
        "    py -3.12 -m venv venv && venv\\Scripts\\activate\n"
        "    python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "
        "\"torch[device-gfx1150]\" torchvision torchaudio\n"
        "  per-target index:  https://repo.amd.com/rocm/whl/gfx1150/   "
        "(win_amd64: torch 2.9.1 / 2.10.0 / 2.11.0, cp311-cp314)\n"
        "  nightly TheRock:   https://nightly.repo.amd.com/rocm/whl-next/\n"
        "  DirectML is maintenance-mode (torch-directml 0.2.5 pins torch 2.4.1) "
        "-> only a last resort."
    )


def _probe_compute_dtype(device: torch.device, want: str) -> Tuple[torch.dtype, str]:
    """
    Pick the AMP compute dtype by *executing* a matmul, not by asking the ISA.

    `torch.cuda.is_bf16_supported()` answers for the hardware, not for the wheel:
    a gfx1150 build can still be missing a kernel.  So try bf16 → fp16 → fp32 and
    keep the first dtype that runs; fp32 means "no autocast" (`amp=False`).
    """
    order = {"bf16": [torch.bfloat16, torch.float16, torch.float32],
             "fp16": [torch.float16, torch.bfloat16, torch.float32],
             "auto": [torch.bfloat16, torch.float16, torch.float32]}[want]
    notes: List[str] = []
    for dt in order:
        try:
            a = torch.randn(64, 64, device=device, dtype=dt)
            _ = a @ a
            torch.cuda.synchronize(device)          # HIP faults are async
            return dt, ("probe: matmul ok on " + str(dt).split(".")[-1]
                        + (f" (tried {', '.join(notes)})" if notes else ""))
        except Exception as exc:                                   # noqa: BLE001
            notes.append(f"{str(dt).split('.')[-1]}->{type(exc).__name__}")
    return torch.float32, f"probe: every dtype failed ({', '.join(notes)}) -> fp32"


def _probe_sdpa(device: torch.device, dtype: torch.dtype) -> bool:
    """
    The trunk uses `scaled_dot_product_attention` with an additive causal mask.
    Flash/mem-efficient kernels are not equally available per arch *and* dtype,
    so verify the exact op once: on failure the trunk loads with eager attention
    instead of dying on the first forward.
    """
    try:
        q = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
        m = torch.zeros(1, 1, 16, 16, device=device, dtype=dtype).masked_fill(
            torch.triu(torch.ones(16, 16, dtype=torch.bool, device=device), 1),
            torch.finfo(dtype).min)
        F.scaled_dot_product_attention(q, q, q, attn_mask=m)
        torch.cuda.synchronize(device)
        return True
    except Exception:                                              # noqa: BLE001
        return False


def verify_hardware(args: argparse.Namespace) -> Accelerator:
    """
    Probe the accelerator and bind it *safely*.

      torch.cuda.is_available() + torch.cuda.get_device_name(0)  on a HIP/ROCm
      wheel reports e.g. "AMD Radeon(TM) 880M Graphics" — PyTorch routes HIP
      through the `torch.cuda` namespace, so this is the correct check.

    Fallback chain: HIP/ROCm -> torch-directml -> CPU.  Nothing here is allowed
    to raise: a missing accelerator degrades, it does not abort.
    """
    print("=" * 78)
    print("HARDWARE")
    print("=" * 78)
    print(f"  platform    : {platform.platform()}")
    print(f"  python      : {platform.python_version()}  ({platform.machine()})")
    print(f"  torch       : {torch.__version__}   hip={torch.version.hip}   "
          f"cuda={torch.version.cuda}")
    print(f"  alloc conf  : {os.environ.get('PYTORCH_HIP_ALLOC_CONF', '-')}")

    forced = args.device
    wants_cpu = forced == "cpu"
    # explicit --attn-impl wins on every backend; the GPU branch probes when "auto"
    impl_forced = args.attn_impl if args.attn_impl != "auto" else "sdpa"

    if not wants_cpu and torch.cuda.is_available():
        backend = "rocm" if torch.version.hip else "cuda"
        name = torch.cuda.get_device_name(args.gpu_index)          # <- required probe
        props = torch.cuda.get_device_properties(args.gpu_index)
        total = props.total_memory / (1024 ** 3)
        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]
        print(f"  backend     : {backend.upper()} "
              f"(torch.cuda API over {'HIP' if backend == 'rocm' else 'CUDA'})")
        print(f"  device      : {name}   arch={arch or 'n/a'}")
        print(f"  memory pool : {total:.2f} GiB total · "
              f"{props.multi_processor_count} CUs · warp={props.warp_size}")
        if hasattr(props, "is_integrated"):
            print(f"  integrated  : {bool(props.is_integrated)} (UMA carve-out, "
                  f"shares system RAM with the desktop)")
        up = name.upper()
        if "AMD" in up or "RADEON" in up:
            print(f"  verified    : torch.cuda.get_device_name({args.gpu_index}) -> AMD "
                  f"{'Radeon 880M / RDNA 3.5 (gfx1150)' if '880M' in up else 'device'}")
        else:
            print(f"  WARNING     : device name is not a Radeon — is this really the "
                  f"880M and not an NVIDIA/other adapter?")
        if arch and arch.lower() not in ("gfx1150", "n/a") and not _GFX_OVERRIDE:
            print(f"  WARNING     : arch={arch} != gfx1150.  If this is a generic "
                  f"wheel, retry with  --gfx-override 11.0.0")
        print(f"  hsa override: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '-')}")

        # Hard allocator ceiling: leave head-room for the WDDM compositor/driver.
        # On a UMA carve-out an unbounded allocation spike can freeze the desktop.
        #
        # Measured on this machine (880M, gfx1150): `props.total_memory` reports
        # 16.86 GiB, i.e. the *shared* system-RAM pool HIP may map (is_integrated=1) —
        # NOT the BIOS carve-out.  A fraction of that pool would be no protection at
        # all (0.85 x 16.86 = 14.3 GiB), so the fraction is only an upper bound and
        # --vram-budget-gib supplies the absolute number for the real carve-out.
        note = ""
        frac_cap = total * args.vram_fraction
        budget = args.vram_budget_gib
        try:
            cap = min(frac_cap, budget) if budget > 0 else frac_cap
            frac = min(max(cap / total, 1e-3), 1.0)
            torch.cuda.set_per_process_memory_fraction(frac, args.gpu_index)
            note = (f"allocator capped at {frac:.0%} of the reported pool "
                    f"(= {cap:.2f} GiB of {total:.2f} GiB)")
        except Exception as exc:                                   # pragma: no cover
            cap, note = 0.0, f"could not cap allocator ({type(exc).__name__}: {exc})"
        print(f"  guard       : {note}")
        if cap >= total - 1e-9:
            print(f"  WARNING     : the cap equals the whole reported pool — no "
                  f"head-room is reserved for the desktop; lower --vram-fraction")
        elif budget > 0 and budget <= frac_cap and bool(
                getattr(props, "is_integrated", False)):
            print(f"  note        : the reported pool is shared system RAM, so the "
                  f"{cap:.2f} GiB budget — not {args.vram_fraction:.0%} of "
                  f"{total:.2f} GiB — is what actually protects the desktop")
        elif budget > 0 and budget > frac_cap:
            print(f"  note        : budget {budget:.2f} GiB exceeds "
                  f"--vram-fraction x pool ({frac_cap:.2f} GiB), so the fraction is "
                  f"the binding limit; raise --vram-fraction to use the extra")

        device = torch.device(f"cuda:{args.gpu_index}")
        dtype, probe = _probe_compute_dtype(device, args.dtype)
        sdpa_ok = _probe_sdpa(device, dtype)
        amp = bool(args.amp) and dtype is not torch.float32
        scaler = bool(amp) and dtype is torch.float16    # bf16 needs no loss scaling
        attn_impl = (args.attn_impl if args.attn_impl != "auto"
                     else ("sdpa" if sdpa_ok else "eager"))
        print(f"  amp         : {amp}   dtype={str(dtype).split('.')[-1]}   "
              f"grad-scaler={'yes' if scaler else 'no'}   ({probe})")
        print(f"  attention   : {attn_impl}"
              + ("  (SDPA kernel unavailable -> eager reference path)"
                 if attn_impl == "eager" else "  (probe: sdpa+additive mask ok)"))
        alloc_backend = getattr(torch.cuda, "get_allocator_backend", None)
        print(f"  allocator   : {alloc_backend() if alloc_backend else 'caching'}"
              f" · max_split_size_mb=256 · garbage_collection_threshold=0.9")
        free_now, _ = torch.cuda.mem_get_info(args.gpu_index)
        print(f"  free now    : {free_now / 2**30:.2f} GiB free · "
              f"{torch.cuda.memory_allocated(args.gpu_index) / 2**20:.1f} MiB allocated")
        return Accelerator(torch.device(f"cuda:{args.gpu_index}"), backend, name,
                           total, dtype, amp, scaler, arch=arch, note=note,
                           attn_impl=attn_impl, gfx_override=_GFX_OVERRIDE)

    if not wants_cpu and not torch.cuda.is_available():
        print("  backend     : -- no torch.cuda device ---------------------------------")
        if platform.system() == "Windows":
            print("  reason      : this wheel has no AMD/HIP backend. "
                  f"torch.version.hip={torch.version.hip}")
            print(_rocm_install_hint())

    # ── fallback 1: torch-directml (DirectX 12, shared system memory, no autocast)
    if not wants_cpu and args.device == "auto":
        try:
            import torch_directml                                # noqa: WPS433
            dev = torch_directml.device(args.gpu_index)
            name = torch_directml.device_name(args.gpu_index)
            print(f"  backend     : DIRECTML fallback -> {name}")
            print("  note        : no autocast/GradScaler on DirectML; fp32 path. "
                  "Utilities and two-pass backward are supported, per-op support varies.")
            return Accelerator(dev, "directml", name, float("nan"),
                               torch.float32, False, False,
                               note="DirectML fallback (maintenance mode)",
                               attn_impl=impl_forced)
        except ImportError:
            pass

    print(f"  backend     : CPU fallback ({platform.processor() or 'unknown cpu'})")
    print("  note        : CPU runs disable autocast (bf16-on-CPU is slower than fp32 "
          "here); use a ROCm wheel for real training.")
    return Accelerator(torch.device("cpu"), "cpu", platform.processor() or "cpu",
                       float("nan"), torch.float32, False, False,
                       note="CPU: correctness/debug only", attn_impl=impl_forced)


# =============================================================================
# 3 · SYNTHETIC SENSORIMOTOR DATA  (stand-in for a MuJoCo arm)
# =============================================================================
@dataclass
class Episode:
    """A batch of `steps` control steps: instruction + sensors + expert actions
    + what the policy should SAY at each step ('' = silent)."""
    texts: List[str]                              # B instruction strings
    sensors: torch.Tensor                         # (B, steps, SENSOR_DIM) float32
    actions: torch.Tensor                         # (B, steps) long  expert labels
    goal_dist: torch.Tensor                       # (B, steps) float32  |goal-pos|
    instr_ids: Optional[torch.Tensor] = None      # (B, Ti) long   — filled by collate()
    instr_mask: Optional[torch.Tensor] = None     # (B, Ti) bool   True = real token
    tasks: Optional[torch.Tensor] = None          # (B,) long  task index (TASKS)
    speech: Optional[List[List[str]]] = None      # B × steps sentences, '' = silent
    heights: Optional[torch.Tensor] = None        # (B, steps) long  expert height answer
    state_texts: Optional[List[List[str]]] = None  # B × steps — the state SENTENCES
    state_ids: Optional[torch.Tensor] = None      # (B, steps*L) long  filled by collate
    ans_pos: Optional[torch.Tensor] = None        # (B, steps, Q) long  marker index
    speech_ids: Optional[torch.Tensor] = None     # (B, steps, Ls) long  sentence + END of
    #                                               the utterance ACTIVE at that step
    #                                               ([END] = silent), -100 padded (collate)
    speech_off: Optional[torch.Tensor] = None     # (B, steps) long  tokens of that utterance
    #                                               already spoken BEFORE the step
    #                                               (0 when silent / first frame)
    # ── the INTERFACE this batch was rendered with (defaults = the default path) ──
    qbind: Optional["QuestionBinding"] = None     # None = the model's own binding
    opt_perm: Optional[Dict[str, torch.Tensor]] = None   # qid -> (B, K) long: canonical
    #                                               option shown at answer tag j (shuffle)
    labels: Optional[Dict[str, torch.Tensor]] = None     # qid -> (B, steps) long given
    #                                               labels (LIBERO); None = derive them
    q_weights: Optional[Dict[str, torch.Tensor]] = None  # qid -> (K,) option CE weights
    source: str = "synthetic"                     # "synthetic" | "libero"
    variant: str = "default"                      # "default" | "train" | "heldout"

    def _extras(self, rows: Optional[Callable] = None, steps: Optional[slice] = None,
                device: Optional[torch.device] = None) -> Dict[str, Any]:
        """The interface fields of a derived Episode: `rows` cuts the batch dim,
        `steps` the time dim of the (B, steps) labels, `device` moves tensors."""
        each = (lambda m, fn: None if m is None else {k: fn(v) for k, v in m.items()})
        perm, lab, qw = self.opt_perm, self.labels, self.q_weights
        if rows is not None:
            perm, lab = each(perm, rows), each(lab, rows)
        if steps is not None:
            lab = each(lab, lambda t: t[:, steps])
        if device is not None:
            mv = (lambda t: t.to(device))
            perm, lab, qw = each(perm, mv), each(lab, mv), each(qw, mv)
        return dict(qbind=self.qbind, opt_perm=perm, labels=lab, q_weights=qw,
                    source=self.source, variant=self.variant)

    def window(self, start: int, width: int) -> "Episode":
        sl = slice(start, start + width)
        return Episode(self.texts, self.sensors[:, sl], self.actions[:, sl],
                       self.goal_dist[:, sl], self.instr_ids, self.instr_mask,
                       self.tasks,
                       None if self.speech is None else [row[sl] for row in self.speech],
                       None if self.heights is None else self.heights[:, sl],
                       None if self.state_texts is None
                       else [row[sl] for row in self.state_texts],
                       None if self.state_ids is None
                       else self.state_ids[:, sl.start * self.step_len:(sl.stop) * self.step_len]
                       if self.step_len else None,
                       # `ans_pos` is state-region-relative (collate writes
                       # `s*L + …`), so a window must re-base it: otherwise every
                       # marker past the first window points past the end of the
                       # sliced `state_ids` (measured: a CUDA index assert on the
                       # second training window; window 0 is unmoved, so the
                       # fault is invisible until the episode advances).
                       None if self.ans_pos is None
                       else self.ans_pos[:, sl] - sl.start * self.step_len,
                       None if self.speech_ids is None else self.speech_ids[:, sl],
                       None if self.speech_off is None else self.speech_off[:, sl],
                       **self._extras(steps=sl))

    def half(self, which: int) -> "Episode":
        """First/second half of the batch (used by the OOM micro-batch fallback)."""
        h = self.sensors.shape[0] // 2
        return self.rows(slice(0, h) if which == 0 else slice(h, None))

    def rows(self, sl: slice) -> "Episode":
        """Batch rows `sl` (a slice), every per-row field cut alike."""
        cut = (lambda t: None if t is None else t[sl])
        return Episode(self.texts[sl], self.sensors[sl], self.actions[sl],
                       self.goal_dist[sl], cut(self.instr_ids), cut(self.instr_mask),
                       cut(self.tasks),
                       None if self.speech is None else self.speech[sl],
                       cut(self.heights),
                       None if self.state_texts is None else self.state_texts[sl],
                       cut(self.state_ids), cut(self.ans_pos), cut(self.speech_ids),
                       cut(self.speech_off), **self._extras(rows=cut))

    def row_windows(self, starts: Sequence[int], width: int) -> "Episode":
        """Row b's window `[starts[b], starts[b] + width)`, rows re-stacked: eval
        probes that must END on a chosen frame (a speaking one) whatever the
        other rows do.  Same field contract as `window` (markers re-based)."""
        parts = [self.rows(slice(b, b + 1)).window(int(s), width)
                 for b, s in enumerate(starts)]
        f = (lambda name: [getattr(p, name) for p in parts])
        cat = (lambda xs: None if xs[0] is None else torch.cat(xs, 0))
        lst = (lambda xs: None if xs[0] is None else [r for x in xs for r in x])
        dic = (lambda ms: None if ms[0] is None
               else {k: torch.cat([m[k] for m in ms], 0) for k in ms[0]})
        return Episode(lst(f("texts")), cat(f("sensors")), cat(f("actions")),
                       cat(f("goal_dist")), cat(f("instr_ids")), cat(f("instr_mask")),
                       cat(f("tasks")), lst(f("speech")), cat(f("heights")),
                       lst(f("state_texts")), cat(f("state_ids")), cat(f("ans_pos")),
                       cat(f("speech_ids")), cat(f("speech_off")),
                       qbind=self.qbind, opt_perm=dic(f("opt_perm")),
                       labels=dic(f("labels")), q_weights=self.q_weights,
                       source=self.source, variant=self.variant)

    def to(self, device: torch.device) -> "Episode":
        cut = (lambda t: None if t is None else t.to(device))
        return Episode(self.texts, self.sensors.to(device), self.actions.to(device),
                       self.goal_dist.to(device), cut(self.instr_ids),
                       cut(self.instr_mask), cut(self.tasks), self.speech,
                       cut(self.heights), self.state_texts,
                       cut(self.state_ids), cut(self.ans_pos), cut(self.speech_ids),
                       cut(self.speech_off), **self._extras(device=device))

    @property
    def step_len(self) -> int:
        """Tokens per control step in `state_ids` (0 when it is not collated)."""
        if self.state_ids is None or self.ans_pos is None or self.ans_pos.shape[1] == 0:
            return 0
        return int(self.state_ids.shape[1] // self.ans_pos.shape[1])

    def speech_io(self, K: int = 1
                  ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        STREAMED teacher-forcing tensors for the speech of the window's LAST
        step: `(inp (B,Ts), inp_mask (B,Ts), tgt (B,Ts+1))`, see
        `speech_io_from_targets` — input = the already-spoken prefix plus the
        within-frame teacher forcing, loss only on the `K` tokens this frame
        emits.  None when the episode carries no speech targets.
        """
        if self.speech_ids is None:
            return None
        return speech_io_from_targets(
            self.speech_ids[:, -1],
            None if self.speech_off is None else self.speech_off[:, -1], K)


def speech_io_from_targets(tgt: Optional[torch.Tensor],
                           off: Optional[torch.Tensor] = None,
                           K: Optional[int] = None
                           ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    (B,Ls) -100-padded `tgt` (sentence + END of the active utterance, [END] =
    silent) + `off (B,)` (tokens already spoken) -> (inp, inp_mask, tgt_out).

    Per row, with n = min(K, len(tgt) - off) (clipped at END):
      * input  = tgt[:off + n - 1]  — spoken prefix + within-frame teacher forcing
      * loss   ONLY at the positions predicting tgt[off:off + n]; the prefix is
        never supervised.  speech_h position j predicts tgt[j] (position 0 = the
        last state position), so `tgt_out[b, j] = tgt[b, j]` there, -100 elsewhere.
    A silent row (tgt=[END], off=0) has no input and target END at position 0.
    Rows with different `off` are padded with mask False / target -100; widths
    are trimmed to the batch max (inp (B,Ts), tgt_out (B,Ts+1)).
    `off=None` means 0 and `K=None` supervises the whole rest of the sentence
    (plain full-sentence teacher forcing).
    """
    if tgt is None:
        return None
    B, L = int(tgt.shape[0]), int(tgt.shape[1])
    dev = tgt.device
    if B == 0 or L == 0:
        return (torch.zeros(B, 0, dtype=torch.long, device=dev),
                torch.zeros(B, 0, dtype=torch.bool, device=dev),
                torch.full((B, 1), -100, dtype=torch.long, device=dev))
    off = (torch.zeros(B, dtype=torch.long, device=dev) if off is None
           else off.to(dev).long())
    k = L if K is None else max(1, int(K))
    real = (tgt >= 0).sum(dim=1)                                  # (B,) incl. END
    n = (real - off).clamp(min=0).clamp(max=k)
    stop = off + n                                                # exclusive
    Ts = max(int((stop - 1).max()), 0)
    tp = tgt[:, :Ts + 1]
    if tp.shape[1] < Ts + 1:
        tp = F.pad(tp, (0, Ts + 1 - tp.shape[1]), value=-100)
    pos = torch.arange(Ts + 1, device=dev)[None]
    sup = (pos >= off[:, None]) & (pos < stop[:, None])
    tgt_out = torch.where(sup, tp, torch.full_like(tp, -100))
    m = torch.arange(Ts, device=dev)[None] < (stop - 1)[:, None]
    inp = torch.where(m, tp[:, :Ts].clamp(min=0), torch.zeros_like(tp[:, :Ts]))
    return inp, m, tgt_out


def stream_schedule(events: List[Optional[List[int]]], K: int, end_id: int
                    ) -> Tuple[List[List[int]], List[int]]:
    """
    Spread utterances over control frames, `K` tokens per frame.

    `events[t]` = the sentence tokens (WITHOUT END) of an event at frame t, or
    None.  Returns `(ids_per_step, off_per_step)`: per frame the full target
    `tgt = sentence + [END]` of the utterance ACTIVE there ([END] when silent)
    and `off` = tokens of it already spoken before that frame (0 when silent).

    Rules:
      1. An utterance starts on its event frame with off = 0.
      2. Its j-th frame has off = j*K and emits tgt[off:off+K], clipped at END;
         the frame that emits END is its last.
      3. The frame after END is silent ([END], off 0) unless a queued
         utterance starts there.
      4. An event arriving while an utterance is streaming (including on the
         frame that emits its END) is QUEUED — at most ONE in the queue; any
         further event while the queue is full is DROPPED.  The queued one
         starts (off 0) on the frame after the current one's END.
      5. An utterance still streaming at the end of `events` is cut off.
    """
    K = max(1, int(K))
    ids: List[List[int]] = []
    offs: List[int] = []
    cur: Optional[List[int]] = None
    queue: Optional[List[int]] = None
    off = 0
    for ev in events:
        if cur is None and queue is not None:
            cur, queue, off = queue, None, 0                      # rule 3/4
        if ev is not None:
            if cur is None:
                cur, off = list(ev) + [end_id], 0                 # rule 1
            elif queue is None:
                queue = list(ev) + [end_id]                       # rule 4
            # else: queue full -> dropped
        if cur is None:
            ids.append([end_id])
            offs.append(0)
            continue
        ids.append(list(cur))
        offs.append(off)
        off += K                                                  # rule 2
        if off >= len(cur):
            cur, off = None, 0
    return ids, offs


class SensorWorld:
    """
    Vectorised 3-D point-mass task, executed on the CPU in fp32.

    Scene per sample: 4 objects at random XY positions and at one of a few
    discrete heights (`OBJ_Z_CHOICES`); the instruction names ONE colour *and*
    one verb (`TASKS`), and the expert steers the gripper to that object while
    repelling from the others.  The sensor vector contains *all four* objects'
    relative XY, their height offsets and their planar distances, so the
    instruction is load-bearing: the policy must bind the colour token in the
    prompt to the matching object slot.

    Height is the third axis and it does NOT live in the 6-token action table —
    there is no "descend" token — so the expert's height decision is emitted as
    the `height` answer (`HEIGHT_NAMES`) and the planar action stays what it
    always was.  That keeps the action table, the reward and every existing
    measurement intact while making the state genuinely 3-D.
    """

    def __init__(self, seed: int = 0, noise: float = 0.15, reach: float = 0.10):
        self.seed = seed
        self.noise = noise
        self.reach = reach
        self.rng = random.Random(seed)

    def instruction(self) -> str:
        return self.rng.choice(INSTRUCTION_TEMPLATES).format(c=self.rng.choice(COLORS))

    def episode(self, batch: int, steps: int) -> Episode:
        rng = self.rng
        B, O = batch, len(COLORS)
        gen = torch.Generator().manual_seed(rng.randrange(1 << 30))

        obj = torch.empty(B, O, 2).uniform_(-1.0, 1.0, generator=gen)
        pos = torch.empty(B, 2).uniform_(-1.2, 1.2, generator=gen)
        vel, grip = torch.zeros(B, 2), torch.zeros(B)
        prev_oh = torch.zeros(B, NUM_ACTIONS)          # "no previous action" at t=0

        target = torch.empty(B, dtype=torch.long)
        tasks = torch.empty(B, dtype=torch.long)
        texts: List[str] = []
        speech: List[List[str]] = [[] for _ in range(B)]              # B × steps sentences
        for b in range(B):
            target[b] = rng.randrange(O)
            tasks[b] = rng.randrange(len(TASKS))
            colour = COLORS[int(target[b])]
            texts.append(rng.choice(TASK_TEMPLATES[int(tasks[b])]).format(c=colour))

        sensors = torch.zeros(B, steps, SENSOR_DIM)
        actions = torch.zeros(B, steps, dtype=torch.long)
        heights = torch.ones(B, steps, dtype=torch.long)             # 1 == "stay"
        goal_dist = torch.zeros(B, steps)

        # Height channel: the tool starts HIGH and the objects sit at one of a few
        # discrete levels, so `tool_z` + `rel_z` carry real information and the
        # expert has something to teach (descend onto the target, lift a throw).
        tool_z = torch.ones(B)
        obj_z = torch.tensor(OBJ_Z_CHOICES, dtype=torch.float32)[
            torch.randint(0, len(OBJ_Z_CHOICES), (B, O), generator=gen)]
        tgt_z = obj_z[torch.arange(B), target]                       # (B,)

        tgt_xy = obj[torch.arange(B), target]                        # (B,2) expert goal
        keep = torch.ones(B, O)
        keep[torch.arange(B), target] = 0.0                          # non-target slots
        state_texts: List[List[str]] = [[] for _ in range(B)]        # B × steps strings
        is_reach, is_grasp = tasks == 0, tasks == 1
        is_throw = tasks == 2
        # previous goal distance, 0 at t=0 so a start inside the radius is not
        # an "arrival" (nothing was reached — the episode began there)
        prev_d = torch.zeros(B)
        for t in range(steps):
            rel = obj - pos.unsqueeze(1)                             # (B,O,2)
            dist = rel.norm(dim=2, keepdim=True).clamp_min(1e-4)      # (B,O,1)
            rel_z = obj_z - tool_z.unsqueeze(1)                      # (B,O) − = tool is above

            sensors[:, t] = torch.cat(
                [pos, vel, grip.unsqueeze(1), tool_z.unsqueeze(1),   # proprioception + height
                 rel.reshape(B, O * 2),                              # ALL object xy slots
                 rel_z.reshape(B, O), dist.reshape(B, O),            # their height + planar gap
                 prev_oh],                                           # last commanded action
                dim=1)                                               # -> (B, SENSOR_DIM)
            # …and the same step as the SENTENCE the model actually reads.
            for b in range(B):
                state_texts[b].append(render_state(
                    (pos[b, 0], pos[b, 1], tool_z[b]),
                    (vel[b, 0], vel[b, 1]), grip[b],
                    [(rel[b, o, 0], rel[b, o, 1], rel_z[b, o]) for o in range(O)]))

            to_goal = tgt_xy - pos
            d = to_goal.norm(dim=1).clamp_min(1e-6)
            goal_dist[:, t] = d
            steer = to_goal / d.unsqueeze(1)                         # attraction to target
            # repulsion from every non-target object inside 0.35 (vectorised)
            push = -(rel / dist) * (1.0 / dist.pow(2)) * (dist < 0.35).float()
            steer = steer + 0.35 * (push * keep.unsqueeze(-1)).sum(dim=1)
            steer = steer + self.noise * torch.randn(B, 2, generator=gen)

            a = (steer @ ACTION_VECS[:4].T).argmax(dim=1)            # best discrete direction
            held = grip > 0.5                                        # jaw already on it
            close = d < self.reach
            a = torch.where(close & (is_reach | is_grasp | is_throw),
                            torch.full_like(a, 4), a)                # -> close the jaw
            a = torch.where(is_reach & (d < 0.04), torch.full_like(a, 5), a)
            a = torch.where(is_grasp & held, torch.full_like(a, 5), a)   # hold it still
            a = torch.where(is_throw & held, torch.full_like(a, 3), a)   # carry and leave
            # `push` keeps the steering action: it never brakes, it drives through

            # Speech events of THIS step, from the transition the expert just
            # chose (the jaw) and the distance it observed (the arrival).
            closes = (a == 4) & ~held                                # jaw 0 -> 1
            opens = held & (a != 4) & (a != 5)                       # jaw 1 -> 0
            arrive = close & (prev_d >= self.reach)                  # entered the radius
            for b in range(B):
                speech[b].append(speech_sentence(
                    detect_speech_event(int(tasks[b]), bool(closes[b]),
                                        bool(opens[b]), bool(arrive[b])),
                    COLORS[int(target[b])]))
            prev_d = d

            # Height expert: descend while aligned in xy and above the target,
            # climb back if we overshot (and are holding nothing), and lift a
            # carried object for `throw`.  This is the third axis the planar
            # action table cannot express — it travels as the `height` answer.
            hz = torch.ones(B, dtype=torch.long)                     # 1 == "stay"
            hz = torch.where(close & (tool_z > tgt_z + Z_TOL),
                             torch.zeros_like(hz), hz)               # 0 == "down"
            hz = torch.where(close & (tool_z < tgt_z - Z_TOL) & ~held,
                             torch.full_like(hz, 2), hz)             # 2 == "up"
            hz = torch.where(is_throw & held, torch.full_like(hz, 2), hz)
            heights[:, t] = hz

            actions[:, t] = a
            vel = 0.85 * vel + 0.30 * ACTION_VECS[a]                 # simple dynamics
            pos = (pos + 0.12 * vel).clamp(-1.4, 1.4)
            tool_z = (tool_z + HEIGHT_VECS[hz]).clamp(0.0, 1.0)      # z servo, bounded
            grip = torch.where(a == 4, torch.ones_like(grip),
                               torch.where(a == 5, grip, torch.zeros_like(grip)))
            prev_oh = F.one_hot(a, NUM_ACTIONS).float()

        return Episode(texts=texts, sensors=sensors, actions=actions,
                       goal_dist=goal_dist, tasks=tasks, speech=speech, heights=heights,
                       state_texts=state_texts)


class EpisodeStream:
    """
    Yields consecutive windows cut out of long episodes, carrying the latent
    `carry` tensor from window to window (truncated BPTT) and dropping it at
    episode boundaries.  Resetting on action emission is *not* a thing here.
    """

    def __init__(self, world: SensorWorld, batch: int, window: int, windows: int,
                 tok, carry: bool = True, qbind: Optional["QuestionBinding"] = None,
                 speak_tokens: int = 1, interface: Optional["InterfaceSampler"] = None):
        self.world, self.batch, self.tok = world, batch, tok
        self.qbind = qbind
        # `interface` (--vary): draws each long episode's rendering (bank subset /
        # order + state format for the batch; paraphrase / option order / schema
        # layout per row).  None = the default interface, the pre-`--vary` path.
        self.interface = interface
        self.speak_tokens = max(1, int(speak_tokens))   # K: speech tokens per frame
        self.window, self.windows, self.use_carry = window, windows, carry
        self.episode: Optional[Episode] = None
        self.offset = 0
        self.carry: Optional[torch.Tensor] = None
        self.episodes = 0

    def next(self, device: torch.device) -> Tuple[Episode, Optional[torch.Tensor]]:
        if self.episode is None or self.offset + self.window > self.episode.sensors.shape[1]:
            long = self.world.episode(self.batch, self.window * self.windows)
            if self.interface is not None:
                self.episode = self.interface.apply(long, self.tok,
                                                    speak_tokens=self.speak_tokens)
            else:
                self.episode = collate(long, self.tok, self.qbind,   # prompt + schema
                                       speak_tokens=self.speak_tokens)   # + state
            self.offset = 0
            self.carry = None                      # episode boundary -> true reset
            self.episodes += 1
        win = self.episode.window(self.offset, self.window).to(device)
        self.offset += self.window
        return win, (self.carry if self.use_carry else None)

    def push(self, carry: Optional[torch.Tensor]) -> None:
        self.carry = None if carry is None else carry.detach()


# =============================================================================
# 4 · MODEL: TEXT STATE · GATED PROMPT FUSION · ACT LOOP
# =============================================================================
class PromptFusion(nn.Module):
    """
    Cross-attention from the STATE region to the PROMPT, through a
    zero-initialised gate:

        h <- h + tanh(gate) · MHA(q=state, kv=prompt)             gate: zeros(1)

    The state is text now, so it enters through the same frozen embedding table
    as the instruction and no sensor projector is needed — but the prompt still
    needs a *trainable* route into the positions the heads answer from.  That is
    what this module is for, and it is not theoretical: measured on a trained
    checkpoint, swapping "move to the red block" for "move to the blue block"
    moved the state-region hidden states by 8.3e-02 while the action logits
    moved by only 6.4e-04 — the prompt was present in `h` and unused by the
    head, and every instruction produced the same live command.

    The ZERO-INITIALISED GATE alone makes this an exact identity at step 0
    (asserted by `--self-test`): the pre-trained trunk sees `h` untouched until
    the model decides the prompt is worth reading, and `tanh(gate)` opens
    smoothly from 0.

    DO NOT zero `attn.out_proj` as well — it looks like extra safety and is
    instead a permanent deadlock, because the two zero factors cancel the
    gradient in both directions:

        dL/dgate  = dL/dh · out_proj(attn_out)  = 0     (out_proj is zero)
        dL/dW_out = dL/dh · tanh(gate)          = 0     (gate is zero)

    (Measured on this file: all three grads exactly 0.0e+00 after a real
    backward, i.e. the cross-attention path could never turn on.  With
    `out_proj` left at its default init the gate's gradient is ~5e-02 — the
    `--self-test` asserts that, so the trap cannot come back.)

    Causality: the prompt precedes every state token, so reading the prompt can
    never leak a future observation.  Nothing here attends *within* the state
    region — that is the trunk's own causal attention, which the additive mask
    already restricts per position (the `no future leak` self-test).
    """

    def __init__(self, hidden: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden % num_heads == 0, "hidden must be divisible by fusion heads"
        self.q_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout,
                                          batch_first=True)
        # out_proj keeps its default init — see the deadlock note above.
        self.gate = nn.Parameter(torch.zeros(1))       # <- the safety gate

    def forward(self, h_state: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h_state: (B,L,H) the state region · context: (B,C,H) prompt/memory."""
        if h_state.numel() == 0 or context is None or context.shape[1] == 0:
            return h_state
        q = self.q_norm(h_state)
        out, _ = self.attn(q, context, context, need_weights=False)
        return h_state + torch.tanh(self.gate) * out


class ACTHaltingHead(nn.Module):
    """
    Scalar halting probability per latent position:  p_t = σ(W·h_t + b).
    Bias is initialised so that E[cycles] ≈ `expect_cycles` at step 0: the
    model starts by spending a sane amount of its budget and the ponder
    penalty then compresses it — the standard ACT warm-up ordering.  The
    EXPECTATION sets the init, never the ceiling: with an unbounded loop the
    ceiling is only a safety net, so asking the head for 65 cycles would be
    nonsense.
    """

    def __init__(self, hidden: int, expect_cycles: float):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.proj.weight.mul_(0.02)
        self.reset_bias(expect_cycles)

    def reset_bias(self, expect_cycles: float) -> None:
        """Bias for E[cycles] ≈ `expect_cycles` (Σ p over the cycles ≈ 1)."""
        p = 1.0 / (float(expect_cycles) + 1.0)
        with torch.no_grad():
            self.proj.bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h)).squeeze(-1)      # (B,T) in (0,1)


# =============================================================================
# 3b · TYPED QUESTION BANK — the JSON decision interface (one forward)
# =============================================================================
# The control output is a *questionnaire*, not generated text.  Every field of
# the emitted JSON is a small softmax over the latent at that question's marker
# slot, read out of ONE forward pass — so the JSON costs one extra matmul per
# question, not one extra decode step per token.  The motor command is then a
# deterministic decode of those answers (`json_action`).  This is the
# mechanism the Jev / System-One family uses (typed state + typed questions ->
# one probability distribution per question from a single forward; the server,
# not the model, assembles the JSON document):
#
#   state -> h_ans (+ task_proj) -> h @ W[labels] -> softmax per question -> JSON
#                                                          +-> json_action -> motor
#
# Measured on this box: one decision is one forward (27 ms tiny / 121 ms Qwen),
# while generating a JSON object autoregressively would be 8-12 forwards per
# decision.  A questionnaire costs ~1 % of that and is exactly reproducible.
#
# Question types mirror the Jev API:
#   choice : K named options   -> {type, choice, probabilities, confidence}
#   score  : K ordered levels  -> {type, score, legend, probabilities, confidence}
#   noul   : one logit, yes/no -> {type, noul}
QUESTION_TYPES = ("choice", "score", "noul")


def _label_list(v: Any) -> Tuple[str, ...]:
    """Labels from a list/tuple, a {label: description} map, or a single string."""
    if not v:
        return ()
    if isinstance(v, dict):
        return tuple(str(k) for k in v)
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


@dataclass(frozen=True)
class QuestionSpec:
    """One field of the emitted JSON.  `labels` is its label set."""
    qid: str
    type: str = "choice"
    options: Tuple[str, ...] = ()
    levels: Tuple[str, ...] = ()
    instructions: str = ""

    @property
    def labels(self) -> Tuple[str, ...]:
        return self.options if self.type == "choice" else self.levels

    @property
    def n_out(self) -> int:
        """Answer width: one logit per label — a `noul` has two (no/yes)."""
        return len(self.labels)

    @classmethod
    def from_json(cls, qid: str, spec: Any) -> "QuestionSpec":
        d = dict(spec or {})
        qtype = str(d.get("type", "choice")).lower()
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"question {qid!r}: type must be one of {QUESTION_TYPES}")
        crit = d.get("criteria")                      # Jev's spelling
        opts = _label_list(d.get("options",
                                 crit if qtype == "choice" else None))
        lvls = _label_list(d.get("levels",
                                 crit if qtype != "choice" else None))
        if qtype == "choice" and len(opts) < 2:
            raise ValueError(f"question {qid!r}: a choice needs >= 2 options")
        if qtype == "score" and len(lvls) < 2:
            raise ValueError(f"question {qid!r}: a score needs >= 2 levels")
        if qtype == "noul" and len(opts) < 2:
            opts = ("no", "yes")          # a noul is read as a 2-option choice
        return cls(qid=str(qid), type=qtype, options=opts, levels=lvls,
                   instructions=str(d.get("instructions", "")))

    def json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        if self.instructions:
            out["instructions"] = self.instructions
        if self.options:
            out["options"] = list(self.options)
        if self.levels:
            out["levels"] = list(self.levels)
        return out


class QuestionBank:
    """
    An ordered set of typed questions, mapped onto one contiguous logit vector:
    `total = sum(n_out)` and `split()` recovers `{qid: (..., n_out)}`.  Built
    from the same JSON document the Jev API accepts, so a robot's questionnaire
    is data, not code — change the file, not the architecture.
    """

    def __init__(self, specs: Sequence[QuestionSpec]):
        self.specs: Tuple[QuestionSpec, ...] = tuple(specs)
        if not self.specs:
            raise ValueError("question bank is empty")
        seen: set = set()
        self.offsets: Dict[str, Tuple[int, int]] = {}
        at = 0
        for s in self.specs:
            if s.qid in seen:
                raise ValueError(f"duplicate question id {s.qid!r}")
            seen.add(s.qid)
            self.offsets[s.qid] = (at, at + s.n_out)
            at += s.n_out
        self.total = at

    @classmethod
    def from_json(cls, obj: Any) -> "QuestionBank":
        """JSON text · a path to one · {'questions': {...}} · a bare id->spec map."""
        if isinstance(obj, str):
            text = obj.strip()
            if text.startswith("{"):
                obj = json.loads(text)
            else:
                with open(text, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
        if isinstance(obj, dict) and "questions" in obj:
            obj = obj["questions"]
        if not isinstance(obj, dict) or not obj:
            raise ValueError("question bank must be a non-empty object of id -> spec")
        return cls([QuestionSpec.from_json(qid, spec) for qid, spec in obj.items()])

    def json(self) -> Dict[str, Any]:
        return {"questions": {s.qid: s.json() for s in self.specs}}

    def split(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(..., total) -> {qid: (..., n_out)} — views, no copy."""
        if logits.shape[-1] != self.total:
            raise ValueError(f"expected {self.total} logits, got {logits.shape[-1]}")
        return {s.qid: logits[..., self.offsets[s.qid][0]:self.offsets[s.qid][1]]
                for s in self.specs}

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, qid: object) -> bool:
        return qid in self.offsets

    def __getitem__(self, qid: str) -> QuestionSpec:
        for s in self.specs:
            if s.qid == qid:
                return s
        raise KeyError(qid)


# The built-in robot questionnaire.  Every question below has a label source in
# `LABEL_SOURCES`, i.e. it is supervised by the expert trajectory (its discrete
# ACTION_NAMES actions) — no extra data collection, no fabricated labels.
DEFAULT_QUESTIONS: Dict[str, Any] = {
    "questions": {
        "axis_x": {"type": "choice", "options": ["negative", "stay", "positive"],
                   "instructions": "Which way should the end effector move along x?"},
        "axis_y": {"type": "choice", "options": ["negative", "stay", "positive"],
                   "instructions": "Which way should the end effector move along y?"},
        "gripper": {"type": "choice", "options": ["open", "close", "stay"],
                    "instructions": "What should the gripper do?"},
        "intent": {"type": "choice", "options": ["approach", "grasp", "hold"],
                   "instructions": "What is the motor intent?"},
        "height": {"type": "choice", "options": ["down", "stay", "up"],
                   "instructions": "Should the end effector descend, hold its "
                                   "height, or climb?"},
        "speed": {"type": "score", "levels": ["slow", "medium", "fast"],
                  "instructions": "How fast should the motion be?"},
    }
}


def resolve_questions(spec: str) -> Optional[QuestionBank]:
    """'off' -> None · 'auto' -> DEFAULT_QUESTIONS · else JSON text or a path."""
    s = (spec or "auto").strip()
    if s.lower() in ("off", "none", "0", ""):
        return None
    if s.lower() in ("auto", "default"):
        return QuestionBank.from_json(DEFAULT_QUESTIONS)
    return QuestionBank.from_json(s)


# ── applicability: when a question is worth asking at all ────────────────────
# Same signature as LABEL_SOURCES.  A question with an entry here gets a learned
# gate (trained against this predicate), so the policy can *choose not to answer*
# fields that carry no information in the current state — e.g. the gripper
# question is meaningless when no object is within reach.  Nothing in the
# read-out path consults it — it is documentation for whoever writes a schema
# and for the live harness, which is free to ignore an answer it cannot use.
SL_VEL = 2                                       # layout: pos(2) · vel(2) · …
SL_GRIP = 4
SL_TOOLZ = 5                                     # the tool's own height
SL_REL = 6                                       # 4 objects × (dx, dy)
SL_RELZ = SL_REL + len(COLORS) * 2               # 4 objects × dz   (= 14)
SL_DIST = SL_RELZ + len(COLORS)                 # 4 objects × |d|  (= 18)
SL_PREV = SL_DIST + len(COLORS)                 # last action one-hot (= 22)


def _first_token(tok, text: str) -> int:
    """The id of the first token of `text` — the token an option is read from."""
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{text!r} tokenizes to nothing")
    return int(ids[0])


JEV_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def answer_tags(spec: "QuestionSpec") -> List[str]:
    """
    The Jev LABELS a question is answered with — the token the model emits at
    its marker.  `choice` options are lettered A, B, C, …; `score` levels are
    numbered 0, 1, 2, …; a `noul` keeps its own two labels (no | yes).  The
    schema line maps each label to the option NAME (`A=negative`), and the JSON
    document reports the name, never the label.
    """
    if spec.type == "choice":
        if len(spec.labels) > len(JEV_LETTERS):
            raise ValueError(f"question {spec.qid!r}: {len(spec.labels)} options but "
                             f"only {len(JEV_LETTERS)} letter labels exist")
        return list(JEV_LETTERS[: len(spec.labels)])
    if spec.type == "score":
        return [str(i) for i in range(len(spec.labels))]
    return list(spec.labels)


class QuestionBinding:
    """
    Binds a `QuestionBank` to a tokenizer: the SCHEMA that goes into the prompt
    and the token ids every answer is read from.

    This is the Jev contract in one object.  Jev takes *state + schema +
    questions* and returns *typed answers with probabilities*; it never
    generates the JSON.  Here the schema is rendered into the prompt — so the
    questionnaire is DATA, and sending a different one changes what the policy
    answers — and every answer is read as the model's OWN next-token
    distribution restricted to that question's LABEL tokens, at a fixed marker
    position after each state.  Nothing is sampled, nothing is decoded, and no
    per-question parameters exist: change the options and the read-out follows.

    Labels (`answer_tags`): choice -> A B C …, score -> 0 1 2 …, noul -> its own
    two labels.  They are single tokens by construction, whatever the option
    names are, so an option called "negative" no longer shares a first sub-word
    with anything else.  The label id is `tok.encode(label)` WITHOUT a leading
    space.  Measured: on Qwen2.5 and SmolLM2 " A".." F" and "A".."F" are both
    single tokens, but " 0".." 5" split into [space, digit] (Qwen [220, 15],
    SmolLM2 [216, 32]) while "0".."5" are single tokens — so the bare form is
    the only spacing that is one token for BOTH choice and score labels.  No
    answer token is ever inserted into the sequence (the answer is read, not
    fed), so there is no collate-side spelling to keep in sync; the binding is
    the single place the rule lives.  `__init__` enforces it: every label must
    be exactly one token and a question's labels must be distinct ids, else a
    ValueError names the question, the label and what it tokenized to.

    The markers are ordinary vocabulary tokens (`x y z g i s …`), one per
    question, appended in bank order to every state sentence; `collate` returns
    their positions as `ans_pos (B,S,Q)`.
    """

    MARKER_POOL = ("x", "y", "z", "g", "i", "s", "k", "j", "q", "w", "v", "b")

    def __init__(self, bank: QuestionBank, tok) -> None:
        self.bank = bank
        if len(bank.specs) > len(self.MARKER_POOL):
            raise ValueError(f"bank has {len(bank.specs)} questions but only "
                             f"{len(self.MARKER_POOL)} markers exist")
        self.markers = list(self.MARKER_POOL[: len(bank.specs)])
        self.marker_ids = [_first_token(tok, m) for m in self.markers]
        self.tags: Dict[str, List[str]] = {s.qid: answer_tags(s) for s in bank.specs}
        self.option_ids: Dict[str, List[int]] = {}
        for s in bank.specs:
            ids: List[int] = []
            for tag, name in zip(self.tags[s.qid], s.labels):
                enc = [int(t) for t in tok.encode(tag, add_special_tokens=False)]
                if len(enc) != 1:
                    raise ValueError(
                        f"question {s.qid!r}: answer label {tag!r} (option {name!r}) "
                        f"tokenizes to {len(enc)} tokens {enc}; every Jev label must "
                        f"be exactly ONE token of this tokenizer")
                ids.append(enc[0])
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"question {s.qid!r}: answer labels {self.tags[s.qid]} map to "
                    f"token ids {ids}, which are not distinct — two options would "
                    f"be read from the same logit")
            self.option_ids[s.qid] = ids
        self.schema = self.render_schema()

    def render_schema(self, layout: str = "lines",
                      texts: Optional[Dict[str, str]] = None,
                      perms: Optional[Dict[str, Sequence[int]]] = None) -> str:
        """The questionnaire as prompt text — what a robot system supplies.

        The defaults are THE schema (bridge, live stack, checkpoint).  `--vary`
        renders the same binding differently per row: `texts` (qid -> reworded
        question), `perms` (qid -> canonical option shown at label j, so `A=`
        may name any option) and `layout` ("lines" | "json" | "compact").  The
        markers and their order are the binding's and never change here."""
        texts, perms = texts or {}, perms or {}
        rows = []
        for m, s in zip(self.markers, self.bank.specs):
            names = list(s.labels)
            if s.qid in perms:
                names = [names[int(j)] for j in perms[s.qid]]
            if s.type == "noul":
                opts = "|".join(names)
            else:
                opts = " ".join(f"{t}={n}" for t, n in zip(self.tags[s.qid], names))
            rows.append((m, s, texts.get(s.qid) or s.instructions or f"What is {s.qid}?",
                         names, opts))
        order = "answer the markers in order (" + " ".join(self.markers) + ")"
        if layout == "json":
            body = {m: {"q": q, "options": (opts if s.type == "noul"
                                            else dict(zip(self.tags[s.qid], names)))}
                    for m, s, q, names, opts in rows}
            return f"schema: {json.dumps(body)}\n{order} right after each state."
        if layout == "compact":
            return (f"questions, {order} after each state: "
                    + "; ".join(f"{m}) {q} {opts}" for m, s, q, names, opts in rows))
        lines = ["schema:"]
        for m, s, q, names, opts in rows:
            lines.append(f"- {m}: {q} options: {opts}")
        lines.append(order + " right after each state.")
        return "\n".join(lines)

    def step_suffix(self) -> str:
        """Appended to every state sentence: the answer markers, in bank order."""
        return " " + " ".join(self.markers)

    def option_weight(self, W: torch.Tensor, qid: str) -> torch.Tensor:
        """The (n_out, H) embedding rows a question's answer is read from."""
        return W[torch.tensor(self.option_ids[qid], device=W.device)]

    def signature(self) -> Dict[str, Any]:
        """What a checkpoint records to detect a different bank / labelling."""
        return {"bank": self.bank.json(), "tags": self.tags,
                "option_ids": self.option_ids}

    def __len__(self) -> int:
        return len(self.bank.specs)


def question_logits(h_ans: torch.Tensor, W: torch.Tensor,
                    qbind: QuestionBinding) -> Dict[str, torch.Tensor]:
    """
    (B,S,Q,H) x (V,H) -> {qid: (B,S,n_out)}: each answer is the model's own
    next-token distribution, restricted to that question's option tokens.  With
    a tied trunk `W` IS `embed_tokens.weight`, so this is literally the LM head
    masked to the options — the mechanism Jev uses — and it needs no parameters
    of its own, so the schema stays pure data.
    """
    out: Dict[str, torch.Tensor] = {}
    for i, spec in enumerate(qbind.bank.specs):
        out[spec.qid] = h_ans[:, :, i, :] @ qbind.option_weight(W, spec.qid).t()
    return out


SPEED_EDGES = (0.25, 0.75)      # |tool velocity| → slow | medium | fast


def _lab_axis(a: torch.Tensor, i_neg: int, i_pos: int) -> torch.Tensor:
    """Expert action → axis answer: 0 = negative, 1 = stay, 2 = positive."""
    return torch.where(a == i_neg, torch.zeros_like(a),
                       torch.where(a == i_pos, torch.full_like(a, 2),
                                   torch.ones_like(a)))


def _lab_speed(sensors: torch.Tensor) -> torch.Tensor:
    """|tool velocity| → speed level, from the same proprioception the action uses."""
    v = sensors[..., SL_VEL:SL_VEL + 2].norm(dim=-1)
    return (v > SPEED_EDGES[0]).long() + (v > SPEED_EDGES[1]).long()


LABEL_SOURCES: Dict[str, Callable[..., torch.Tensor]] = {
    "axis_x": lambda a, s, x: _lab_axis(a, 0, 1),
    "axis_y": lambda a, s, x: _lab_axis(a, 2, 3),
    "gripper": lambda a, s, x: torch.where(a == 4, torch.full_like(a, 1),
                                           torch.where(a == 5, torch.full_like(a, 2),
                                                       torch.zeros_like(a))),
    "intent": lambda a, s, x: torch.where(a == 4, torch.full_like(a, 1),
                                          torch.where(a == 5, torch.full_like(a, 2),
                                                      torch.zeros_like(a))),
    "height": lambda a, s, x: x["height"],
    "speed": lambda a, s, x: _lab_speed(s),
    # the `--vary` distractor: the jaw state the state sentence itself prints
    # (0 = open, 1 = closed) — a non-motor read-out the model must not confuse
    # with `gripper` (what to DO with the jaw).
    "grip_state": lambda a, s, x: (s[..., SL_GRIP] > 0.5).long(),
}


def derive_question_labels(bank: QuestionBank, sensors: torch.Tensor,
                           actions: torch.Tensor,
                           extra: Optional[Dict[str, torch.Tensor]] = None
                           ) -> Dict[str, torch.Tensor]:
    """
    Expert labels for every question, derived from the expert trajectory (the
    ACTION_NAMES actions + sensors).  A bank containing a question with no label source raises
    here — loudly — instead of quietly training that field on noise, and a
    question whose source needs `extra` raises when the caller forgot it.
    """
    extra = extra or {}
    out: Dict[str, torch.Tensor] = {}
    for s in bank.specs:
        fn = LABEL_SOURCES.get(s.qid)
        if fn is None:
            raise ValueError(f"no label source for question {s.qid!r}: add one to "
                             f"LABEL_SOURCES or drop it from the bank")
        lab = fn(actions, sensors, extra).long()
        if int(lab.max()) >= s.n_out:
            raise ValueError(f"label {int(lab.max())} out of range for {s.qid!r} "
                             f"({s.n_out} options: {'|'.join(s.labels)})")
        out[s.qid] = lab
    return out


def episode_labels(ep: "Episode", bank: Optional[QuestionBank]
                   ) -> Optional[Dict[str, torch.Tensor]]:
    """Per-question targets of a batch: the GIVEN labels (LIBERO rows carry
    their converter's answers) or, for synthetic rows, the derived ones."""
    if ep.labels is not None:
        if bank is None:
            return dict(ep.labels)
        return {s.qid: ep.labels[s.qid] for s in bank.specs if s.qid in ep.labels}
    if bank is None:
        return None
    return derive_question_labels(bank, ep.sensors, ep.actions,
                                  extra={"height": ep.heights})


def _inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    """(…, K) permutation (tag j shows canonical option perm[j]) -> its inverse
    (canonical option c is shown at tag inv[c])."""
    return torch.argsort(perm, dim=-1)


def canonical_q_logits(q_logits: Dict[str, torch.Tensor],
                       opt_perm: Optional[Dict[str, torch.Tensor]]
                       ) -> Dict[str, torch.Tensor]:
    """
    Undo a per-row option shuffle.  With `perm[b, j]` = the canonical option
    printed at answer tag j (A, B, C, …) for row b, the logit of canonical
    option c is the logit of tag `inv[b, c]`.  After this the logits are in the
    bank's own option order again, so the CE against canonical targets equals
    the CE against the SHOWN (remapped) targets, class weights stay per-option,
    and `json_action` decodes names, not letters.
    """
    if not opt_perm:
        return q_logits
    out = dict(q_logits)
    for qid, perm in opt_perm.items():
        lg = out.get(qid)
        if lg is None:
            continue
        inv = _inverse_perm(perm.to(lg.device))               # (B, K)
        idx = inv.view(inv.shape[0], *([1] * (lg.dim() - 2)), inv.shape[1])
        out[qid] = lg.gather(-1, idx.expand_as(lg))
    return out


def shown_targets(labels: Dict[str, torch.Tensor],
                  opt_perm: Optional[Dict[str, torch.Tensor]]
                  ) -> Dict[str, torch.Tensor]:
    """Canonical targets -> the TAG index each row must answer (the remapped
    target a shuffled schema asks for): `inv[b, lab[b, s]]`."""
    if not opt_perm:
        return dict(labels)
    out = dict(labels)
    for qid, perm in opt_perm.items():
        lab = out.get(qid)
        if lab is None:
            continue
        inv = _inverse_perm(perm.to(lab.device))              # (B, K)
        out[qid] = inv.gather(1, lab.reshape(lab.shape[0], -1)).reshape(lab.shape)
    return out


# ── JSON -> action: the exact inverse of LABEL_SOURCES ───────────────────────
# The 6-way action is no longer a head: it is DECODED from the typed answers.
# Forward table (what LABEL_SOURCES writes for each expert action):
#
#   action     axis_x     axis_y     gripper   intent
#   -x    (0)  negative   stay       open      approach
#   +x    (1)  positive   stay       open      approach
#   -y    (2)  stay       negative   open      approach
#   +y    (3)  stay       positive   open      approach
#   grasp (4)  stay       stay       close     grasp
#   brake (5)  stay       stay       stay      hold
#
# Decoding order — total over EVERY answer combination and deterministic:
#   1. jaw first (the expert overrides steering with grasp/brake the same way):
#      gripper (intent when the bank has no gripper) == close/grasp -> grasp,
#      == stay/hold -> brake.  gripper wins when both are present.
#   2. motion: one axis off `stay` -> that direction.  BOTH off `stay` (never
#      produced by the labels) -> the axis whose arg-max probability is larger;
#      an exact tie (or labels given without confidences) -> x.
#   3. both axes `stay` with the jaw open (never produced) -> brake, the same
#      "stay/stay brakes" rule `command_from_json` applies.
# Label indices are positional (LABEL_SOURCES semantics), so a bank that renames
# the options still decodes as long as it keeps their order.
JSON_ACTION_QIDS = ("axis_x", "axis_y", "gripper", "intent")
# Questions whose answer must be read off the NUMBERS in the state (not a
# near-constant class): the logged shuffled-vs-plain split tracks these.
NUMERIC_QIDS = ("axis_x", "axis_y", "speed")


def json_action_from_labels(labels: Dict[str, Any],
                            conf: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """Label indices {qid: int | (…) long} -> action index tensor (same shape)."""
    lab = {k: torch.as_tensor(v).long() for k, v in labels.items()
           if k in JSON_ACTION_QIDS and v is not None}
    if not lab:
        raise ValueError(f"json_action needs at least one of {JSON_ACTION_QIDS}")
    ref = next(iter(lab.values()))
    stay = torch.ones_like(ref)
    ax = lab.get("axis_x", stay).to(ref.device)
    ay = lab.get("axis_y", stay).to(ref.device)
    conf = conf or {}
    zc = torch.zeros(ref.shape, dtype=torch.float32, device=ref.device)
    cx = torch.as_tensor(conf.get("axis_x", zc)).float().to(ref.device)
    cy = torch.as_tensor(conf.get("axis_y", zc)).float().to(ref.device)
    mx, my = ax != 1, ay != 1
    act = torch.full_like(ref, 5)                                    # brake
    act = torch.where(my, torch.where(ay == 0, torch.full_like(ref, 2),
                                      torch.full_like(ref, 3)), act)
    use_x = mx & (~my | (cx >= cy))
    act = torch.where(use_x, torch.where(ax == 0, torch.zeros_like(ref),
                                         torch.ones_like(ref)), act)
    jaw = lab.get("gripper", lab.get("intent"))
    if jaw is not None:
        jaw = jaw.to(ref.device)
        act = torch.where(jaw == 1, torch.full_like(ref, 4),
                          torch.where(jaw == 2, torch.full_like(ref, 5), act))
    return act


def json_action(q_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
    """{qid: (B,S,K)} question logits -> (B,S) long action index (ACTION_NAMES)."""
    if not q_logits:
        raise ValueError("json_action needs question logits")
    lab: Dict[str, torch.Tensor] = {}
    conf: Dict[str, torch.Tensor] = {}
    for qid in JSON_ACTION_QIDS:
        lg = q_logits.get(qid)
        if lg is None:
            continue
        c, i = torch.softmax(lg.detach().float(), dim=-1).max(dim=-1)
        lab[qid], conf[qid] = i, c
    return json_action_from_labels(lab, conf)


def json_action_one(labels: Dict[str, int],
                    conf: Optional[Dict[str, float]] = None) -> int:
    """Per-sample variant: {qid: label index} (e.g. SAMPLED labels) -> action index."""
    return int(json_action_from_labels(labels, conf))


def has_json_action(bank: Optional[QuestionBank]) -> bool:
    """True when the bank carries at least one question json_action decodes."""
    return bank is not None and any(q in bank for q in JSON_ACTION_QIDS)


def render_answers(bank: Optional[QuestionBank],
                   q_logits: Optional[Dict[str, torch.Tensor]],
                   digits: int = 4,
                   q_labels: Optional[Dict[str, int]] = None,
                   say: Optional[str] = None,
                   action_names: Sequence[str] = ACTION_NAMES) -> Dict[str, Any]:
    """
    Assemble the Jev-shaped response from ONE forward's read-out:
    probabilities, the arg-max option NAME (never the A/B/0/1 label), and (for
    `score`) the probability-weighted value.  Nothing is decoded token by token
    — the JSON is a rendering of the distributions the network already
    produced, which is why it is free and bit-exact, and why a malformed-JSON
    failure mode does not exist.

    `q_labels` overrides the arg-max with the label that was actually SAMPLED
    (or chosen by the caller).  A policy that *acts* on the JSON must report the
    label it executed — otherwise the document would describe a decision the
    robot did not make — and `score` then reports the executed level instead of
    the expectation.

    `action` is DERIVED (`json_action`, the inverse of LABEL_SOURCES) from the
    reported answers — it is not a head.  `say` is the top-level speech field:
    the sentence the policy spoke this frame, `null` when it stayed silent.

    Only the LAST control step of the window is reported: that is the decision
    being executed now.
    """
    answers: Dict[str, Any] = {}
    if bank is None or not q_logits:
        answers["say"] = say
        return answers
    for spec in bank.specs:
        lg = q_logits.get(spec.qid)
        if lg is None:
            continue
        last = lg.detach().float().reshape(-1, lg.shape[-1])[-1]
        if spec.type == "noul":
            # `noul` is P(yes): the second of the two option tokens (no | yes)
            answers[spec.qid] = {
                "type": "noul",
                "noul": round(float(torch.softmax(last, dim=-1)[-1]), digits),
            }
            continue
        pr = torch.softmax(last, dim=-1)
        taken = (q_labels is not None and spec.qid in q_labels)
        i = int(q_labels[spec.qid]) if taken else int(pr.argmax())
        entry: Dict[str, Any] = {
            "type": spec.type,
            "probabilities": {n: round(float(v), digits)
                              for n, v in zip(spec.labels, pr)},
            "confidence": round(float(pr[i]), digits),
        }
        if taken:
            entry["sampled"] = spec.labels[i] if spec.labels else int(i)
        if spec.type == "score":
            if taken:
                entry["score"] = i                     # the level actually executed
            else:
                idx = torch.arange(len(spec.labels), dtype=pr.dtype, device=pr.device)
                entry["score"] = round(float((pr * idx).sum()), 3)
            entry["legend"] = list(spec.labels)
        else:
            entry["choice"] = spec.labels[i]
        answers[spec.qid] = entry
    if has_json_action(bank):
        lab: Dict[str, int] = {}
        conf: Dict[str, float] = {}
        for qid in JSON_ACTION_QIDS:
            lg = q_logits.get(qid)
            if lg is None:
                continue
            pr = torch.softmax(lg.detach().float().reshape(-1, lg.shape[-1])[-1], dim=-1)
            i = (int(q_labels[qid]) if (q_labels is not None and qid in q_labels)
                 else int(pr.argmax()))
            lab[qid], conf[qid] = i, float(pr[i])
        if lab:
            answers["action"] = {"type": "derived",
                                 "choice": action_names[json_action_one(lab, conf)]}
    answers["say"] = say
    return answers


# ── `--vary`: training-time interface variation ──────────────────────────────
# A deployed robot will not print OUR exact schema and state sentence, so the
# supervised stage can re-render its batches: another question subset and
# order (the markers follow the RENDERED order — they are positional), a
# reworded question, shuffled options (the Jev labels A/B/C name other
# options; the logits are un-shuffled before loss and json_action, see
# `canonical_q_logits`), three schema layouts, an optional distractor question,
# and state sentences with other field names / order / units / dropped fields
# / noise.  Paraphrase set PARAPHRASE_HELDOUT and HELDOUT_STATE_FORMAT are
# NEVER drawn for training: they are the eval-only probe of whether the policy
# reads the interface or memorised one rendering of it.
PARAPHRASES: Dict[str, Tuple[str, ...]] = {
    # [0] documents the DEFAULT_QUESTIONS wording (the bank's own text is what
    # set 0 renders), [1] and [2] are training rewordings, [3] is held out.
    "axis_x": ("Which way should the end effector move along x?",
               "Along the x axis, which direction should the tool go?",
               "Pick the x direction for the next motion.",
               "In what direction along x must the hand travel now?"),
    "axis_y": ("Which way should the end effector move along y?",
               "Along the y axis, which direction should the tool go?",
               "Pick the y direction for the next motion.",
               "In what direction along y must the hand travel now?"),
    "gripper": ("What should the gripper do?",
                "What should happen to the jaw now?",
                "Gripper command for this step?",
                "How should the fingers be actuated at this moment?"),
    "intent": ("What is the motor intent?",
               "What is the robot trying to do right now?",
               "Current phase of the manipulation?",
               "Which stage of the task is the arm in?"),
    "height": ("Should the end effector descend, hold its height, or climb?",
               "Should the tool go down, stay level, or go up?",
               "Vertical motion for this step?",
               "Must the arm lower itself, keep its altitude, or rise?"),
    "speed": ("How fast should the motion be?",
              "How quickly should the tool move?",
              "Motion speed for this step?",
              "At what pace should the arm travel?"),
    "grip_state": ("Is the gripper open or closed right now?",
                   "What jaw state does the state line show?",
                   "Current gripper state?",
                   "Are the fingers currently apart or shut?"),
}
PARAPHRASE_HELDOUT = 3                     # wording set 3: eval only

# The distractor: a non-motor question whose answer is printed in the state
# itself (label source `grip_state` in LABEL_SOURCES; synthetic rows only).
DISTRACTOR_QUESTION = QuestionSpec(qid="grip_state", type="choice",
                                   options=("open", "closed"),
                                   instructions=PARAPHRASES["grip_state"][0])


def paraphrase(spec: QuestionSpec, idx: int) -> str:
    """Wording `idx` of a question: 0 = the bank's own text, 1.. = PARAPHRASES
    (a question without a paraphrase set always keeps its own text)."""
    alts = PARAPHRASES.get(spec.qid)
    if idx <= 0 or not alts or idx >= len(alts):
        return spec.instructions or f"What is {spec.qid}?"
    return alts[idx]


@dataclass(frozen=True)
class StateFormat:
    """How a state sentence is printed.  Fields: 0 tool (x y z) · 1 vel · 2 grip
    · 3 obj.  Every format is FIXED WIDTH (clamped numbers), so a batch keeps
    one constant per-step token count."""
    keys: Tuple[str, ...] = STATE_KEYS          # printed field names
    order: Tuple[int, ...] = (0, 1, 2, 3)       # print order (indices into keys)
    unit: str = "m"                             # "m": +0.12 · "cm": +012
    drop: Tuple[int, ...] = ()                  # omitted fields
    noise: float = 0.0                          # N(0, noise) on printed numbers
    name: str = field(default="default", compare=False)


DEFAULT_STATE_FORMAT = StateFormat()
HELDOUT_STATE_FORMAT = StateFormat(keys=("eef", "velocity", "gripper", "items"),
                                   order=(0, 2, 1, 3), unit="cm", name="heldout")
STATE_NAME_POOLS = (("tool", "hand"), ("vel", "motion"), ("grip", "jaw"),
                    ("obj", "objects"))                   # training names only
STATE_ORDERS = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 2, 1, 3), (3, 0, 1, 2), (2, 0, 1, 3))
STATE_NOISE = (0.0, 0.0, 0.005, 0.01)
_F_VEL = 1                                  # the droppable (non-essential) field


def sample_state_format(rng: random.Random, libero: bool = False) -> StateFormat:
    """A TRAINING state format (never HELDOUT_STATE_FORMAT's names).  Velocity
    is dropped only for synthetic rows (the expert steers on positions; the
    `speed` question leaves the bank with it)."""
    keys = tuple(rng.choice(pool) for pool in STATE_NAME_POOLS)
    order = rng.choice(STATE_ORDERS)
    unit = rng.choice(("m", "m", "cm"))
    drop = (_F_VEL,) if (not libero and rng.random() < 0.15) else ()
    return StateFormat(keys, order, unit, drop, rng.choice(STATE_NOISE), name="train")


def _fmt_num(v: float, unit: str) -> str:
    if unit == "cm":
        return f"{max(-999.0, min(999.0, float(v) * 100.0)):+04.0f}"
    return f"{max(-9.99, min(9.99, float(v))):+.2f}"


def render_state_as(fmt: StateFormat, tool, vel, grip, rel=None, *,
                    grip_len: bool = False,
                    rng: Optional[random.Random] = None) -> str:
    """
    One control step in format `fmt`.  DEFAULT_STATE_FORMAT reproduces
    `render_state` (synthetic: 2-D vel, jaw 0/1, 4 objects) and — with
    `grip_len` and no `rel` — the LIBERO converter's sentence (3-D vel, finger
    width in metres) byte for byte for in-range values (numbers are clamped
    to keep the width fixed).  Noise jitters the printed positions and
    velocities only (labels come from the clean values).
    """
    sd = fmt.noise if (rng is not None and fmt.noise > 0.0) else 0.0
    if sd:
        num = (lambda v: _fmt_num(float(v) + rng.gauss(0.0, sd), fmt.unit))
    else:                                       # untouched value (keeps -0.00)
        num = (lambda v: _fmt_num(v, fmt.unit))
    k_tool, k_vel, k_grip, k_obj = fmt.keys
    if grip_len:                                # finger width (m): 0.078 | 07.8 cm
        g = max(0.0, float(grip))
        gs = f"{min(g * 100.0, 99.9):04.1f}" if fmt.unit == "cm" else f"{min(g, 9.999):.3f}"
    else:                                       # jaw state 0/1: unit-free
        gs = f"{float(grip):.2f}"
    fields = [f"{k_tool} " + " ".join(num(v) for v in tool),
              f"{k_vel} " + " ".join(num(v) for v in vel),
              f"{k_grip} {gs}",
              None if rel is None else
              f"{k_obj} " + " ".join(f"{c} " + " ".join(num(v) for v in rel[i])
                                     for i, c in enumerate(COLORS[:len(rel)]))]
    parts = [fields[i] for i in fmt.order if i not in fmt.drop and fields[i] is not None]
    return ("cm " if fmt.unit == "cm" else "") + " ".join(parts)


def render_episode_states(ep: "Episode", fmt: StateFormat,
                          rng: Optional[random.Random] = None) -> List[List[str]]:
    """Re-render every state sentence of a RAW episode from its sensors."""
    v = ep.sensors.detach().float().cpu()
    out: List[List[str]] = []
    for b in range(v.shape[0]):
        row: List[str] = []
        for x in v[b].tolist():
            if ep.source == "libero":           # x y z · vx vy vz · finger width
                row.append(render_state_as(fmt, x[0:3], x[3:6], x[6], None,
                                           grip_len=True, rng=rng))
            else:
                rel = [(x[SL_REL + 2 * o], x[SL_REL + 2 * o + 1], x[SL_RELZ + o])
                       for o in range(len(COLORS))]
                row.append(render_state_as(fmt, (x[0], x[1], x[SL_TOOLZ]),
                                           (x[SL_VEL], x[SL_VEL + 1]), x[SL_GRIP],
                                           rel, rng=rng))
        out.append(row)
    return out


@dataclass
class Interface:
    """One stream episode's rendering: the binding (bank subset + order ->
    markers), the state format, and per-row schema text + option order."""
    qbind: "QuestionBinding"
    fmt: StateFormat = DEFAULT_STATE_FORMAT
    schemas: Optional[List[str]] = None                  # per row; None = qbind.schema
    perms: Optional[Dict[str, List[List[int]]]] = None   # qid -> B x K shown order
    variant: str = "default"                             # default | train | heldout
    paras: Optional[List[Dict[str, int]]] = None         # per row: qid -> wording set


class InterfaceSampler:
    """
    Draws the interface of each long stream episode (`EpisodeStream(...,
    interface=)`).  PER BATCH (one draw for all rows and all windows of the
    episode, because question_logits reads the markers by position and a batch
    needs one per-step token count; it also keeps the carry's interface
    coherent across windows): question subset (the json_action questions
    JSON_ACTION_QIDS are ALWAYS kept) + order, the distractor, the state
    format.  PER ROW (the prompt is padded per row): wording set, option order
    (`choice` questions only — `score` levels are ordinal), schema layout.

    mode="train"  : with probability `prob` a varied draw, else the default.
    mode="heldout": held-out wording set + HELDOUT_STATE_FORMAT on the default
                    bank / layout / option order (eval only).
    """
    LAYOUTS = ("lines", "json", "compact")
    CORE = JSON_ACTION_QIDS

    def __init__(self, qbind: "QuestionBinding", tok, prob: float = 0.0,
                 seed: int = 0, mode: str = "train", distractor: bool = True):
        if mode not in ("train", "heldout"):
            raise ValueError(f"InterfaceSampler mode must be train|heldout, got {mode!r}")
        self.base, self.tok, self.mode = qbind, tok, mode
        self.prob = min(1.0, max(0.0, float(prob)))
        self.rng = random.Random(seed)
        self.distractor = bool(distractor) and "grip_state" not in qbind.bank
        self._cache: Dict[Tuple[str, ...], "QuestionBinding"] = {
            tuple(s.qid for s in qbind.bank.specs): qbind}
        self.drawn = self.varied = self.fallbacks = 0

    def binding(self, specs: Sequence[QuestionSpec]) -> "QuestionBinding":
        key = tuple(s.qid for s in specs)
        qb = self._cache.get(key)
        if qb is None:
            qb = self._cache[key] = QuestionBinding(QuestionBank(list(specs)), self.tok)
        return qb

    def draw(self, B: int, libero: bool = False) -> Interface:
        rng, base = self.rng, self.base
        if self.mode == "heldout":
            texts = {s.qid: paraphrase(s, PARAPHRASE_HELDOUT) for s in base.bank.specs}
            return Interface(base, HELDOUT_STATE_FORMAT,
                             [base.render_schema("lines", texts=texts)] * B, None,
                             "heldout", [{q: PARAPHRASE_HELDOUT for q in texts}] * B)
        if rng.random() >= self.prob:
            return Interface(base)
        fmt = sample_state_format(rng, libero=libero)
        specs = list(base.bank.specs)
        keep = [s for s in specs if s.qid in self.CORE]
        keep += [s for s in specs if s.qid not in self.CORE
                 and not (s.qid == "speed" and _F_VEL in fmt.drop)
                 and rng.random() < 0.6]
        if not keep:                            # a bank without json_action questions
            keep = [rng.choice(specs)]
        if self.distractor and not libero and rng.random() < 0.25:
            keep.append(DISTRACTOR_QUESTION)
        rng.shuffle(keep)
        qb = self.binding(keep)
        schemas: List[str] = []
        perms: Dict[str, List[List[int]]] = {s.qid: [] for s in keep if s.type == "choice"}
        paras: List[Dict[str, int]] = []
        for _ in range(B):
            idx = {s.qid: rng.randrange(PARAPHRASE_HELDOUT) for s in keep}  # never 3
            row: Dict[str, List[int]] = {}
            for s in keep:
                if s.type != "choice":
                    continue
                p = list(range(s.n_out))
                if rng.random() < 0.7:
                    rng.shuffle(p)
                row[s.qid] = p
                perms[s.qid].append(p)
            schemas.append(qb.render_schema(
                rng.choice(self.LAYOUTS), perms=row,
                texts={s.qid: paraphrase(s, idx[s.qid]) for s in keep}))
            paras.append(idx)
        return Interface(qb, fmt, schemas, perms or None, "train", paras)

    def apply(self, raw: "Episode", tok, speak_tokens: int = 1) -> "Episode":
        """Draw an interface for a RAW episode and collate it under that interface."""
        iface = self.draw(raw.sensors.shape[0], libero=(raw.source == "libero"))
        self.drawn += 1
        src = raw
        if iface.fmt != DEFAULT_STATE_FORMAT:
            src = Episode(**dict(vars(raw), state_texts=render_episode_states(
                raw, iface.fmt, self.rng)))
        try:
            ep = collate(src, tok, iface.qbind, speak_tokens=speak_tokens,
                         schemas=iface.schemas)
        except ValueError as exc:
            # a tokenizer that splits one varied state format into a varying
            # token count: train this episode on the default interface rather
            # than stop the run (held-out eval raises; `evaluate` reports it)
            if iface.variant != "train":
                raise
            self.fallbacks += 1
            if self.fallbacks == 1:
                print(f"  [vary] WARNING: {exc} — falling back to the default "
                      f"interface for such episodes")
            return collate(raw, tok, self.base, speak_tokens=speak_tokens)
        if iface.variant != "default":
            self.varied += 1
            ep.qbind, ep.variant = iface.qbind, iface.variant
            ep.opt_perm = ({q: torch.tensor(p, dtype=torch.long)
                            for q, p in iface.perms.items()} if iface.perms else None)
        return ep


# ── LIBERO demos (`--libero`): converted episodes as a second data source ────
# `libero_convert.py` writes one JSON line per demo — {"episode", "split",
# "prompt", "frames": [{"t", "state": sentence, "answers": {qid: option},
# "say": str|None}]} — and meta.json {"class_weights": {qid: {option: w}}}.
# Rows carry GIVEN labels (no synthetic expert), no task id (task loss masked)
# and their own state sentence `tool x y z vel vx vy vz grip width` (3-D
# velocity, finger width in m, no objects): a different per-step token count
# than the synthetic sentence, so LIBERO windows travel in their OWN batches
# (run_train) and every batch keeps one constant step length.
_LIBERO_FIELDS = {"tool": (0, 3), "vel": (3, 3), "grip": (6, 1)}


def parse_state_sentence(text: str) -> List[float]:
    """A LIBERO state sentence -> [x, y, z, vx, vy, vz, width] (0 when absent)."""
    words = str(text or "").split()
    out = [0.0] * 7
    for i, w in enumerate(words):
        span = _LIBERO_FIELDS.get(w)
        if span is None:
            continue
        at, n = span
        for j in range(n):
            if i + 1 + j >= len(words):
                break
            try:
                out[at + j] = float(words[i + 1 + j])
            except ValueError:
                break
    return out


@dataclass
class LiberoEpisode:
    prompt: str
    vals: torch.Tensor                      # (T, 7) float32  x y z vx vy vz width
    labels: Dict[str, torch.Tensor]         # qid -> (T,) long, BANK option order
    say: List[str]                          # T sentences ('' = silent)
    split: str = "train"

    def __len__(self) -> int:
        return int(self.vals.shape[0])


class LiberoData:
    """
    The converted jsonl, parsed ONCE (line by line).  Answers are mapped by
    option NAME onto the bank's label order (an unknown name raises); demos
    shorter than `min_len` frames are skipped; meta.json next to the file gives
    per-option CE weights (`q_weights`, bank option order; None when absent).
    """

    def __init__(self, path: str, bank: QuestionBank, min_len: int = 1,
                 meta: Optional[str] = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LIBERO episodes not found: {path}")
        self.path = path
        self.qids: List[str] = []
        self.episodes: Dict[str, List[LiberoEpisode]] = {"train": [], "eval": []}
        self.skipped = 0
        index = {s.qid: {n: i for i, n in enumerate(s.labels)} for s in bank.specs}
        with open(path, "r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                frames = rec.get("frames") or []
                if not frames or len(frames) < max(1, int(min_len)):
                    self.skipped += 1
                    continue
                if not self.qids:
                    first = frames[0].get("answers") or {}
                    self.qids = [q for q in index if q in first]
                    if not self.qids:
                        raise ValueError(f"{path}:{ln}: no LIBERO answer matches a bank "
                                         f"question ({', '.join(index)})")
                labels: Dict[str, torch.Tensor] = {}
                for q in self.qids:
                    col: List[int] = []
                    for f in frames:
                        name = (f.get("answers") or {}).get(q)
                        i = index[q].get(name)
                        if i is None:
                            raise ValueError(
                                f"{path}:{ln}: frame {f.get('t')} answers {q}={name!r}, "
                                f"not one of {'|'.join(index[q])}")
                        col.append(i)
                    labels[q] = torch.tensor(col, dtype=torch.long)
                vals = torch.tensor([parse_state_sentence(f.get("state", ""))
                                     for f in frames], dtype=torch.float32)
                split = "eval" if str(rec.get("split", "train")).lower() == "eval" else "train"
                self.episodes[split].append(LiberoEpisode(
                    str(rec.get("prompt") or "").strip(), vals, labels,
                    [str(f.get("say") or "").strip() for f in frames], split))
        if not self.episodes["train"]:
            raise ValueError(f"{path}: no train-split LIBERO episodes "
                             f"(>= {min_len} frames)")
        meta = meta or os.path.join(os.path.dirname(os.path.abspath(path)), "meta.json")
        self.meta_path = meta if os.path.isfile(meta) else None
        self.q_weights: Optional[Dict[str, torch.Tensor]] = None
        if self.meta_path:
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                cw = (json.load(fh) or {}).get("class_weights") or {}
            qw = {s.qid: torch.tensor([float(cw[s.qid].get(n, 1.0)) for n in s.labels],
                                      dtype=torch.float32)
                  for s in bank.specs if s.qid in cw and s.qid in self.qids}
            self.q_weights = qw or None

    def summary(self) -> str:
        eps = self.episodes["train"] + self.episodes["eval"]
        frames = sum(len(e) for e in eps)
        spoken = sum(1 for e in eps for s in e.say if s)
        return (f"{len(self.episodes['train'])} train / {len(self.episodes['eval'])} "
                f"eval demos · {frames} frames ({spoken} with speech) · answers "
                f"{' '.join(self.qids)} · class weights "
                f"{'meta.json' if self.q_weights else 'none (in-batch balance)'}"
                + (f" · skipped {self.skipped} short" if self.skipped else ""))


class LiberoWorld:
    """
    LIBERO demos as an `EpisodeStream` world: `episode(B, steps)` cuts B random
    segments of `steps` frames (whole windows; shortened when no demo is that
    long) out of one split and returns them RAW — DEFAULT-format state
    sentences, the given labels, `say` as speech events (collate runs the same
    `stream_schedule` as for synthetic rows), tasks = -1 (task loss masked).
    `speaking=True` makes every segment contain a `say` frame at position >=
    window-1, for the eval speech probe.
    """

    def __init__(self, data: LiberoData, split: str = "train", seed: int = 0,
                 window: int = 1, speaking: bool = False):
        self.data, self.split = data, split
        self.window = max(1, int(window))
        self.pool = [e for e in data.episodes.get(split, []) if len(e) >= self.window]
        if not self.pool:
            raise ValueError(f"LIBERO split {split!r} has no demo of >= {self.window} frames")
        self.seed, self.rng, self.speaking = seed, random.Random(seed), bool(speaking)
        self._say = [[t for t, s in enumerate(e.say) if s] for e in self.pool]

    def episode(self, batch: int, steps: int) -> Episode:
        rng, W = self.rng, self.window
        longest = max(len(e) for e in self.pool)
        S = min(int(steps), longest)
        S = max(W, S - S % W)
        cands = [i for i, e in enumerate(self.pool) if len(e) >= S]
        spk = ([i for i in cands if any(t >= W - 1 for t in self._say[i])]
               if self.speaking else [])
        picks: List[Tuple[LiberoEpisode, int]] = []
        for _ in range(batch):
            if spk:
                i = rng.choice(spk)
                T = len(self.pool[i])
                ts = rng.choice([t for t in self._say[i] if t >= W - 1])
                lo, hi = max(0, ts - S + 1), min(T - S, ts - W + 1)
                s0 = rng.randint(lo, hi) if lo <= hi else rng.randint(0, T - S)
            else:
                i = rng.choice(cands)
                s0 = rng.randint(0, len(self.pool[i]) - S)
            picks.append((self.pool[i], s0))
        B = len(picks)
        vals = torch.stack([e.vals[s0:s0 + S] for e, s0 in picks])          # (B,S,7)
        labels = {q: torch.stack([e.labels[q][s0:s0 + S] for e, s0 in picks])
                  for q in self.data.qids}
        actions = (json_action_from_labels(labels)
                   if any(q in labels for q in JSON_ACTION_QIDS)
                   else torch.zeros(B, S, dtype=torch.long))
        heights = (labels["height"].clone() if "height" in labels
                   else torch.ones(B, S, dtype=torch.long))
        states = [[render_state_as(DEFAULT_STATE_FORMAT, v[0:3], v[3:6], v[6], None,
                                   grip_len=True) for v in row.tolist()] for row in vals]
        return Episode(texts=[e.prompt for e, _ in picks], sensors=vals, actions=actions,
                       goal_dist=torch.zeros(B, S),
                       tasks=torch.full((B,), -1, dtype=torch.long),
                       speech=[e.say[s0:s0 + S] for e, s0 in picks], heights=heights,
                       state_texts=states, labels=labels, q_weights=self.data.q_weights,
                       source="libero")


def speech_probe(ep: "Episode", width: int, rng: random.Random) -> Optional["Episode"]:
    """
    Per-row windows of `width` steps that END on a speaking frame (a random
    frame for a row that never speaks), cut from a COLLATED long episode: the
    speech metric is read at a window's last frame, and natural window ends
    almost never land inside an utterance (eval measured 'speaking 0/32').
    """
    if ep.speech_ids is None or ep.speech_ids.shape[1] < width:
        return None
    S = ep.speech_ids.shape[1]
    speaking = (ep.speech_ids >= 0).sum(-1) > 1                       # (B, S)
    starts: List[int] = []
    for b in range(speaking.shape[0]):
        ts = [t for t in range(width - 1, S) if bool(speaking[b, t])]
        t = rng.choice(ts) if ts else rng.randrange(width - 1, S)
        starts.append(t - width + 1)
    return ep.row_windows(starts, width)


@dataclass
class LoopOutput:
    ponder: torch.Tensor             # (B,S)  ACT compute cost  N(t) + R(t)
    cycles: torch.Tensor             # (B,S)  hard loop cycles consumed
    carry: torch.Tensor              # (B,H)  final mixed latent -> next window
    stats: Dict[str, Any] = field(default_factory=dict)
    q_logits: Optional[Dict[str, torch.Tensor]] = None   # {qid: (B,S,n_out)}
    task_logits: Optional[torch.Tensor] = None           # (B,S,len(TASKS)) verb guess
    speech_h: Optional[torch.Tensor] = None              # (B,1+Ts,H) latents that
    #   predict the speech: the last state position (-> first speech token) and
    #   every appended speech input (-> the next one)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA — adapt the WHOLE trunk without touching a single frozen weight
# ─────────────────────────────────────────────────────────────────────────────
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
"""Every projection a Qwen2 / SmolLM2 / Llama decoder layer stacks."""


class LoRALinear(nn.Module):
    """
        y = W x + (alpha/r) * B (A x)        W frozen, B initialised to ZERO

    Same contract as the gated fusion module: the adapter is *identity* at step
    0, so wrapping every projection in the trunk cannot perturb the pretrained
    forward pass — only the projections the loss actually wants to move get
    moved.  Rank 16 over the 7 projections of all 24 Qwen2.5-0.5B layers is
    ~8.8M trainable parameters: the whole model adapts, for a tenth of the
    memory full fine-tuning needs for its AdamW state, and the pretrained
    language ability (which the narration depends on) is never overwritten.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)              # frozen: that is the whole point
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        h = F.linear(self.drop(x), self.lora_a)                 # (..., r)
        return y + self.scale * F.linear(h, self.lora_b)        # B=0 -> y exactly


def inject_lora(trunk: nn.Module, rank: int, alpha: float = 16.0,
                dropout: float = 0.0) -> int:
    """Wrap every trunk projection in a `LoRALinear`; returns the wrap count."""
    n = 0
    for layer in getattr(trunk, "layers", []):
        for parent in (getattr(layer, "self_attn", None),
                       getattr(layer, "mlp", None)):
            if parent is None:
                continue
            for name in LORA_TARGETS:
                mod = getattr(parent, name, None)
                if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
                    setattr(parent, name, LoRALinear(mod, rank, alpha, dropout))
                    n += 1
    return n


class LoopedACTTransformer(nn.Module):
    """
    Trunk adapter: the pretrained decoder becomes a *looped* transformer.

      layers[0 .. L-R-1] : preamble, executed once
      layers[L-R .. L-1] : the recurrent block, re-applied up to `max_loops`
                           times through the ACT loop (weight-tied recurrence)

    Only `embed_tokens`, `layers`, `norm`, `rotary_emb` of the pretrained trunk
    are used; the LM head is not part of this policy.
    """

    def __init__(self, trunk: nn.Module, *, num_looped_layers: int = 2,
                 max_loops: int = 4, act_eps: float = 1e-2, pond_tau: float = 1e-2,
                 fusion_heads: int = 8,
                 dropout: float = 0.0, grad_checkpoint: bool = False,
                 qbind: Optional[QuestionBinding] = None,
                 untie_lm_head: bool = False,
                 lora_rank: int = 0, lora_alpha: float = 16.0,
                 lora_dropout: float = 0.0,
                 cycle_tag: str = "learned"):
        super().__init__()
        self.trunk = trunk
        # ── LoRA: injected BEFORE the layer lists below, so `loop_layers`
        #    (the SAME layer objects as the trunk's tail) is wrapped too — the
        #    recurrent block must not be the one part of the trunk that cannot
        #    adapt.
        self.lora_rank = int(lora_rank)
        self.n_lora = (inject_lora(trunk, self.lora_rank, lora_alpha, lora_dropout)
                       if self.lora_rank > 0 else 0)
        self.hidden: int = trunk.config.hidden_size
        self.R, self.max_loops = num_looped_layers, max_loops
        assert cycle_tag in ("learned", "sinusoidal"), f"bad cycle_tag {cycle_tag!r}"
        self.cycle_tag = str(cycle_tag)
        self.act_eps, self.pond_tau = act_eps, pond_tau
        self.grad_checkpoint = grad_checkpoint

        n_layers = len(trunk.layers)
        assert 0 < num_looped_layers <= n_layers, "not enough trunk layers to loop"
        self._preamble = list(trunk.layers[: n_layers - num_looped_layers])
        self.loop_layers = nn.ModuleList(list(trunk.layers[n_layers - num_looped_layers:]))

        # ── add-on modules (the only trainable parts by default) ─────────────
        self.fusion = PromptFusion(self.hidden, fusion_heads, dropout)
        self.halt_head = ACTHaltingHead(self.hidden, min(max_loops, 8))
        self.carry_norm = nn.LayerNorm(self.hidden)
        self.carry_tag = nn.Parameter(torch.zeros(self.hidden))
        # ── the per-cycle step tag: HOW the loop remembers how long it has
        #    been looping.  `learned` is a zero-init row per cycle — the table
        #    IS the structural cap, because `cycle_emb[cycle]` cannot be read
        #    past its last row.  `sinusoidal` is a parameter-free encoding of
        #    the cycle INDEX behind a zero-init gate: it is defined for every
        #    cycle, so `max_loops` stops being a structural bound and becomes a
        #    pure safety ceiling — the halting head alone decides how long the
        #    model thinks (the ponder penalty only *rewards* finishing early).
        self.cycle_gate = nn.Parameter(torch.zeros(1))
        if self.cycle_tag == "learned":
            self.cycle_emb: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(max_loops, self.hidden))
        else:
            self.cycle_emb = None
            half = (self.hidden + 1) // 2
            self.register_buffer(
                "cycle_freq",
                10000.0 ** (-torch.arange(half, dtype=torch.float32)
                            * 2.0 / self.hidden),
                persistent=False)               # derived: never in the payload
        # (no action head: the 6-way action is DECODED from the typed answers —
        #  `json_action`, the exact inverse of LABEL_SOURCES)
        # JSON decision interface: the questionnaire is BOUND to the tokenizer
        # (schema text + the token ids of every option) and each answer is read
        # out of the model's OWN next-token distribution at that question's
        # marker — so the schema is data and there are no per-question
        # parameters at all.  See `QuestionBinding` / `question_logits`.
        self.qbind = qbind
        self.bank = qbind.bank if qbind is not None else None

        # ── auxiliary: name the task the prompt asked for ────────────────────
        # WHY THIS EXISTS (measured): the control objective alone gives the
        # prompt a *sparse* signal — the four verbs steer IDENTICALLY until the
        # tool is near the target, so almost every frame is uninformative about
        # which sentence was issued.  On a trained adapter checkpoint, swapping
        # "move to the red block" for "move to the blue block" moved the sensor
        # slots' hidden states by 8.3e-02 but the decision by only 6.4e-04, and
        # live every instruction produced the same `-x` command.  This head turns
        # "read the prompt into the read-out" into its own objective: the prompt
        # is the ONLY channel that carries the verb, so the CE is small and its
        # gradient flows through `fusion.task_attn` and `task_proj` into exactly
        # the latents the answers are read from.  At inference the same logits
        # are a free self-check (`task_inferred` in the JSON document).
        self.task_head = nn.Sequential(
            nn.LayerNorm(self.hidden), nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(), nn.Linear(self.hidden // 2, len(TASKS)),
        )
        # The gated `task_attn` route alone is too WEAK to steer the arm: it is
        # one residual term inside the loop, and after 3000 steps its gate sat at
        # tanh = -0.13, i.e. ~13 % of an attention output — enough for the verb
        # head (CE 0.002) but not enough to flip a decision.  This is the direct
        # route: the pooled prompt latent is projected and ADDED to the latent
        # at EVERY answer marker (`h_ans`) before the LM-head option scoring, so
        # the instruction reaches each JSON answer at full strength; `task_head`
        # reads the same enriched latent (at the last marker of each step).  It
        # is deliberately not zero-init — it feeds only the READ-OUT (never the
        # loop), so nothing inside the trunk is perturbed, and checkpoints
        # without `task_proj` load with `strict=False` and simply start it fresh.
        self.task_proj = nn.Linear(self.hidden, self.hidden)

        # ── token generation path ────────────────────────────────────────────
        # `load_base` loads the trunk WITHOUT its LM head (a policy does not need
        # one).  Generation is restored here, so the same latent can emit text
        # (a plan, a tool name, a JSON document for debugging) as well as the
        # action and the typed answers.  For tied trunks — Qwen2.5, SmolLM2,
        # most small decoders — the head IS `embed_tokens.weight`, so this costs
        # no parameters at all; untied trunks get a real (trainable) head.
        # WHY `untie_lm_head` EXISTS: the tied matrix is a *frozen* trunk tensor
        # under `--trainable adapters`, so a tied head can only be steered by
        # moving the latent — the token loss plateaus (measured: txt 6.478 vs
        # ln(1024)=6.93 after 2400 steps).  Untying gives the latent a real
        # decoder to learn into (a linear map, ~136M params on Qwen2.5-0.5B,
        # 65k on the tiny trunk) and is in ADAPTER_MODULES, hence trainable.
        emb = getattr(trunk, "embed_tokens", None)
        tied = (bool(getattr(trunk.config, "tie_word_embeddings", True))
                and not untie_lm_head)
        self.lm_weight = emb.weight if (emb is not None and tied) else None
        self.lm_head = (None if (emb is None or tied) else
                        nn.Linear(self.hidden, int(trunk.config.vocab_size), bias=False))
        if self.lm_head is not None:
            # Start the untied head AS the pretrained head — a copy of the frozen
            # embedding matrix — not as a random Linear.  Measured on the probe:
            # a random head on this latent opens at txt=22.6 against
            # ln(151936)=11.9, so the first steps are spent relearning language
            # the trunk already had.  The copy keeps the head's weights
            # trainable (the decoder still moves away from the embedding, which
            # is the point of untying) — it just does not start from noise.
            with torch.no_grad():
                self.lm_head.weight.copy_(emb.weight.detach())

        # (the per-cycle "answer now?" head and the early-exit read-out are gone:
        #  the loop halts through ACT alone)

        # ── decoder-layer ABI (transformers 4.44+ vs 5.x differ, probe once) ──
        sig = inspect.signature(type(trunk.layers[0]).forward).parameters
        self._kw = {n for n in ("attention_mask", "position_ids", "past_key_values",
                                "past_key_value", "use_cache", "cache_position",
                                "output_attentions") if n in sig}
        self._use_pe = "position_embeddings" in sig
        rope = getattr(trunk, "rotary_emb", None)
        if rope is None and self._use_pe:                     # older 4.4x layout
            rope = trunk.layers[-1].self_attn.rotary_emb
        self._rope = rope
        assert (not self._use_pe) or (self._rope is not None), "no rotary embedding found"

    # ── decoder plumbing ────────────────────────────────────────────────────
    def _layer(self, layer: nn.Module, h: torch.Tensor,
               pe: Optional[Tuple[torch.Tensor, torch.Tensor]],
               mask: torch.Tensor, cache_pos: torch.Tensor) -> torch.Tensor:
        kw: Dict[str, Any] = {"attention_mask": mask}
        if self._use_pe:
            kw["position_embeddings"] = pe
        if "position_ids" in self._kw:
            kw["position_ids"] = (None if self._use_pe
                                  else cache_pos.unsqueeze(0).expand(h.shape[0], -1))
        if "past_key_values" in self._kw:
            kw["past_key_values"] = None
        if "past_key_value" in self._kw:
            kw["past_key_value"] = None
        if "use_cache" in self._kw:
            kw["use_cache"] = False
        if "cache_position" in self._kw:
            kw["cache_position"] = cache_pos
        if "output_attentions" in self._kw:
            kw["output_attentions"] = False
        out = layer(h, **kw)
        return out[0] if isinstance(out, tuple) else out      # 4.x: tuple, 5.x: tensor

    @staticmethod
    def _additive_mask(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        (B,1,T,T) float mask = causal ∧ key-validity, built in the *attention*
        dtype so it adds to the scores without a type promotion (which would
        knock autocast out of bf16/fp16), and in `finfo(dtype).min` rather than
        -inf: fp16's min is -65504, and a masked score must not become NaN when
        a padded row is fully masked.
        """
        B, T = valid.shape
        neg = torch.finfo(dtype).min
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=valid.device), 1)
        mask = causal.to(dtype).mul_(neg).expand(B, 1, T, T).clone()
        return mask.masked_fill((~valid)[:, None, None, :], neg)

    def _recurrent(self, h: torch.Tensor,
                   pe: Optional[Tuple[torch.Tensor, torch.Tensor]],
                   mask: torch.Tensor, cache_pos: torch.Tensor) -> torch.Tensor:
        """One pass through the weight-tied recurrent block (R trunk layers)."""
        for layer in self.loop_layers:
            h = self._layer(layer, h, pe, mask, cache_pos)
        return h

    # ── forward ─────────────────────────────────────────────────────────────
    def _cycle_tag(self, cycle: int, ref: torch.Tensor) -> torch.Tensor:
        """The step tag added at cycle `cycle` — a function of the INDEX for
        `sinusoidal` (no table, so no bound), a table row for `learned`."""
        if self.cycle_emb is not None:
            return self.cycle_emb[cycle]
        idx = torch.arange(self.hidden, device=ref.device)
        ang = float(cycle) * self.cycle_freq[idx // 2]
        # cast to the activation dtype: a fp32 tag would promote the residual
        # stream out of fp16 and defeat autocast (same reason as the mask).
        return torch.where(idx % 2 == 0, torch.sin(ang), torch.cos(ang)).to(ref.dtype)

    def forward(self, instr_ids: torch.Tensor, instr_mask: torch.Tensor,
                state_ids: torch.Tensor, ans_pos: torch.Tensor,
                carry: Optional[torch.Tensor] = None,
                qbind: Optional["QuestionBinding"] = None,
                speech_ids: Optional[torch.Tensor] = None,
                speech_mask: Optional[torch.Tensor] = None) -> LoopOutput:
        """
        `state_ids` is the tokenized state (one fixed-width sentence per control
        step) and `ans_pos (B,S,Q)` holds the marker positions `collate` computed.
        `qbind` overrides the questionnaire the answers are read from: it holds no
        parameters, so passing a different one reads a different SCHEMA off the
        same weights — that is what makes the interface portable.

        `speech_ids (B,Ts)` (+ `speech_mask`, True = real token) are appended
        AFTER the state: the teacher-forced sentence during training, the tokens
        decoded so far at inference.  The mask is causal, so they cannot reach
        any answer marker, the task read-out or the carry — all of which sit at
        or before the last state token.  They run through the looped layers like
        every other token; PromptFusion stays on the state region.
        """
        B, Ti = instr_ids.shape
        SL = state_ids.shape[1]            # S × (tokens per state sentence)
        S = ans_pos.shape[1]               # control steps in this window
        dev = instr_ids.device

        # (1) input assembly ---------------------------------------------------
        #     The state is TEXT: one fixed-width sentence per control step,
        #     closed by STATE_MARKER and embedded by the SAME (frozen) embedding
        #     table as the prompt.  No sensor projector, no continuous slots —
        #     any robot that can print the sentence can drive the policy.
        text_tokens = self.trunk.embed_tokens(instr_ids)              # (B,Ti,H)
        state_tokens = self.trunk.embed_tokens(state_ids)             # (B,SL,H)
        parts = [text_tokens, state_tokens]
        valid = [instr_mask, torch.ones(B, SL, dtype=torch.bool, device=dev)]
        Ts = 0
        if speech_ids is not None and speech_ids.shape[1] > 0:
            # [carry][instruction + schema][state steps + markers][speech]
            Ts = int(speech_ids.shape[1])
            parts.append(self.trunk.embed_tokens(speech_ids))         # (B,Ts,H)
            valid.append(torch.ones(B, Ts, dtype=torch.bool, device=dev)
                         if speech_mask is None else speech_mask.bool())
        if carry is not None:
            # continuous latent memory: previous window's final mixed state,
            # re-injected as a prefix token (never a re-embedded action token).
            mem = self.carry_norm(carry).unsqueeze(1) + self.carry_tag
            parts.insert(0, mem)
            valid.insert(0, torch.ones(B, 1, dtype=torch.bool, device=dev))
        n_mem = 1 if carry is not None else 0
        h = torch.cat(parts, dim=1)                                   # (B,T,H)
        valid = torch.cat(valid, dim=1)
        T = h.shape[1]
        off = n_mem + Ti                                              # first state token
        end = off + SL                                                # first speech token
        cache_pos = torch.arange(T, device=dev)
        position_ids = cache_pos.unsqueeze(0).expand(B, T)
        pe = self._rope(h, position_ids) if self._use_pe else None
        mask = self._additive_mask(valid, h.dtype)

        # (2) preamble: the first L-R layers, once -----------------------------
        for layer in self._preamble:
            h = self._layer(layer, h, pe, mask, cache_pos)

        # (3) ACT loop: continuous latent trajectory ---------------------------
        #     h_t evolves in latent space across cycles; halted positions keep
        #     their accumulated α but stop contributing/updating.
        zero = h.new_zeros(B, T)
        cum = zero.clone()                    # Σ p, clamped at the halting cycle
        updates = zero.clone()                # hard count of computed states / position
        cost = zero.clone()                   # Σ n·α_n  (differentiable compute cost)
        mass = zero.clone()                   # Σ α  (convex weights)
        remainder = h.new_ones(B, T)          # 1-Σp at the halting cycle (ACT's R(t))
        weighted = torch.zeros_like(h)        # Σ α_t · h_t
        halted = torch.zeros(B, T, dtype=torch.bool, device=dev)
        h_t = h
        usage: List[float] = []
        # Each control step is READ at the STATE_MARKER that closes its sentence
        # (ponder, cycles, the carry and the task head), and every QUESTION at
        # its own marker, one token earlier each (`h_ans` below).  Both positions
        # come from `collate`, so a tokenizer that splits a number differently
        # moves the anchors instead of mis-reading them.
        readout = off + ans_pos[..., -1] + 1                        # (B,S) indices
        bidx = torch.arange(B, device=dev).unsqueeze(1).expand(B, S)

        def _read(t: torch.Tensor) -> torch.Tensor:
            """(B,T,…) → (B,S,…): the tensor gathered at each step's read-out."""
            return t[bidx, readout]
        for cycle in range(self.max_loops):
            active = ~halted
            if not bool(active.any()):
                break                                  # every position has halted
            usage.append(float(active.float().mean()))
            # step tag: zero-init row (learned) or gate * sinusoid(cycle) —
            # both are inert at step 0, and only the learned one has a bound.
            if self.cycle_emb is not None:
                h_t = h_t + self.cycle_emb[cycle]
            else:
                h_t = h_t + torch.tanh(self.cycle_gate) * self._cycle_tag(cycle, h_t)
            if self.grad_checkpoint and self.training and h_t.requires_grad:
                h_t = torch.utils.checkpoint.checkpoint(
                    self._recurrent, h_t, pe, mask, cache_pos, use_reentrant=False)
            else:
                h_t = self._recurrent(h_t, pe, mask, cache_pos)
            # gated multi-modal injection (exactly identity at init) — on the
            # STATE region only: the prompt is the context, and the appended
            # speech tokens pass through untouched
            h_t = torch.cat([h_t[:, :off],
                             self.fusion(h_t[:, off:end], context=h_t[:, :off]),
                             h_t[:, end:]],
                            dim=1)

            p = self.halt_head(h_t) * active.float()               # (B,T)
            new_halt = active & (cum + p >= 1.0 - self.act_eps)
            remainder = torch.where(new_halt, (1.0 - cum).clamp_min(0.0), remainder)
            alpha = torch.where(new_halt, remainder, p)            # convex weight
            weighted = weighted + alpha.unsqueeze(-1) * h_t
            mass = mass + alpha
            # expected number of loop iterations this position pays for: the state
            # produced in cycle n is used iff ≥ n iterations are executed, so the
            # weight α_n on h_n costs n iterations.  cost = Σ n·α_n ∈ [1, max_loops]
            cost = cost + alpha * float(cycle + 1)
            cum = torch.where(new_halt, torch.ones_like(cum), cum + p)
            updates = updates + active.float()
            halted = halted | new_halt

        # positions that never reached the budget: close the convex combination
        # on the last state  ⇒  Σ(weights applied to `weighted`) == 1 exactly
        open_ = ~halted
        closed = torch.where(open_, (1.0 - mass).clamp_min(0.0), zero)
        weighted = weighted + closed.unsqueeze(-1) * h_t
        mass = mass + closed
        cost = cost + closed * float(max(len(usage), 1))        # that state is cycle M
        ponder_all = cost                       # Graves-style compute cost, per position

        # (4) read-out: typed answers per control step --------------------------
        z = self.trunk.norm(weighted)                                  # (B,T,H)
        zr = _read(z)                                                  # (B,S,H)
        # The instruction, at full strength, straight into every answer: the
        # read-out must be able to answer "which of the four is my target?" from
        # the sentence, and a gated residual inside the loop is not enough for
        # that (measured: the verb head learned the sentence while the decision
        # stayed constant).  Pooled over the prompt region, which is the memory
        # slot plus the instruction + schema tokens.
        pooled = z[:, :off].mean(dim=1, keepdim=True)                  # (B,1,H)
        tp = self.task_proj(pooled)                                    # (B,1,H)
        qb = self.qbind if qbind is None else qbind
        if qb is not None:
            # Every typed answer is read out of the model's OWN next-token
            # distribution, at its marker, restricted to that question's label
            # tokens: `W` is the LM head's weight matrix (tied or untied), so
            # this is the LM head masked to the labels — Jev's mechanism, with
            # no parameters of its own, so the schema stays pure data.
            Q = ans_pos.shape[2]
            if Q != len(qb):
                raise ValueError(f"ans_pos has {Q} markers but the questionnaire "
                                 f"has {len(qb)} questions")
            bidx3 = bidx.unsqueeze(-1).expand(B, S, Q)                  # (B,S,Q)
            # `ans_pos` is relative to the STATE region (collate computes
            # `s*L + …`), so the absolute position is `off + ans_pos` — the same
            # arithmetic `readout` does.  Indexing `z` with the bare offsets
            # would silently read the prompt instead.
            h_ans = z[bidx3, off + ans_pos] + tp.unsqueeze(2)           # (B,S,Q,H)
            W = (self.lm_weight if self.lm_weight is not None
                 else self.lm_head.weight)                              # (V,H)
            q_logits = question_logits(h_ans, W, qb)                    # {qid:(B,S,K)}
        else:
            q_logits = None
        task_logits = self.task_head(zr + tp)                           # (B,S,len(TASKS))

        stats = {
            "mass_err": float((mass - 1.0).detach().abs().max()),  # convexity residual
            "cycle_usage": usage,
            "mean_cycles": float(_read(updates).detach().mean()),
            "mean_ponder": float(_read(ponder_all).detach().mean()),
            "halted_frac": float(_read(halted).float().mean()),
            "latent_norm": float(zr.detach().norm(dim=-1).mean()),
        }
        return LoopOutput(ponder=_read(ponder_all),
                          cycles=_read(updates),
                          carry=_read(weighted)[:, -1, :].detach(), stats=stats,
                          q_logits=q_logits,
                          task_logits=task_logits,
                          # last state token (-> first speech token) + every
                          # speech input (-> the token after it)
                          speech_h=z[:, end - 1:end + Ts])

    def text_logits(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        """Normalised latent (...,H) -> token logits (...,V) through the LM head."""
        w = self.lm_weight if self.lm_weight is not None else (
            None if self.lm_head is None else self.lm_head.weight)
        return None if w is None else F.linear(h, w)

    def speech_ce(self, h: torch.Tensor, tgt: torch.Tensor,
                  chunk: int = 256) -> Optional[torch.Tensor]:
        """
        Speech CE: `h = out.speech_h (B,1+Ts,H)` against `tgt (B,1+Ts)` (sentence
        + END, -100 padded — `Episode.speech_io`), computed in position chunks.

        The head is (V, H) with V = 151,936 for Qwen, so the obvious
        `cross_entropy(F.linear(h, w).float())` materialises (B, T, V) in fp32 at
        once.  Chunking the *head* as well as the loss keeps the peak at
        (chunk, V); the arithmetic is identical, because the loss is a mean over
        the valid positions and the head is position-wise.  Every row has at
        least one target (END), so a silent frame is trained too: it is how the
        policy learns to say nothing.
        """
        w = self.lm_weight if self.lm_weight is not None else (
            None if self.lm_head is None else self.lm_head.weight)
        if w is None:
            return None
        n = min(h.shape[1], tgt.shape[1])
        h, tgt = h[:, :n], tgt[:, :n].to(h.device)
        tot = h.new_zeros((), dtype=torch.float32)
        den = h.new_zeros((), dtype=torch.float32)
        for c0 in range(0, n, max(1, chunk)):
            c1 = min(c0 + chunk, n)
            lg = F.linear(h[:, c0:c1], w).float().reshape(-1, w.shape[0])
            t = tgt[:, c0:c1].reshape(-1)
            ce = F.cross_entropy(lg, t.clamp(min=0), reduction="none")
            m = (t >= 0).float()
            tot = tot + (ce * m).sum()
            den = den + m.sum()
        return tot / den.clamp_min(1.0)

    def answers_json(self, out: LoopOutput,
                     q_labels: Optional[Dict[str, int]] = None,
                     say: Optional[str] = None,
                     action_names: Sequence[str] = ACTION_NAMES) -> Dict[str, Any]:
        """
        The decision object for the last control step of `out`, in the Jev
        response shape.  Pure rendering of the distributions — no decoding, no
        grammar, no failure mode: `json.dumps(model.answers_json(out))` is valid
        JSON by construction, because nothing was ever generated.

        `q_labels` overrides the question arg-maxes with the labels a sampler
        took, so the document describes the decision that was actually executed.
        `say` is the sentence spoken this frame (top-level, `null` when silent).
        """
        return render_answers(self.bank, out.q_logits, q_labels=q_labels, say=say,
                              action_names=action_names)


# =============================================================================
# 5 · BASE MODEL / TRAINABLE SURFACE
# =============================================================================
class HashTokenizer:
    """Deterministic stub tokenizer for `--model tiny` (offline self-test)."""

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, **kwargs) -> List[int]:              # HF-compatible
        ids = [2 + zlib.crc32(w.encode("utf-8")) % (self.vocab_size - 2)
               for w in text.lower().split()]
        return ids or [self.eos_token_id]


TINY_TRUNK_SEED = 1234
"""Fixed init seed for `--model tiny`.  See `build_tiny_trunk`."""


def build_tiny_trunk(hidden: int = 64, layers: int = 4, vocab: int = 1024,
                     attn_impl: str = "sdpa") -> nn.Module:
    """Randomly-initialised Qwen2 trunk from a config — no download, no network.

    The init runs under a FIXED generator seed, and the caller's RNG state is
    restored afterwards (`torch.random.fork_rng`).  This is not cosmetic:
    `--model tiny` has to name the SAME frozen trunk in every process, or a
    checkpoint's adapters land on different random weights on every run and the
    identical command scores differently each time — measured on this box, two
    identical CPU replays of one checkpoint landed 90.0 % vs 0.0 % of frames in
    reach, and their generated narration differed on the first token ("454 454
    454" vs "759 759 454"), which is the trunk, not the sampling (the sampler is
    greedy).  A real model id is unaffected: those weights come from the hub.
    """
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.for_model(
        "qwen2", vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TINY_TRUNK_SEED)
        try:
            return AutoModel.from_config(cfg, attn_implementation=attn_impl)
        except TypeError:                   # transformers without the kwarg
            return AutoModel.from_config(cfg)


def _auto_trunk(model_id: str, attn_impl: str) -> nn.Module:
    """
    Load the decoder trunk, tolerating two real-world mismatches:

      * `dtype=` (transformers >= 4.56) vs `torch_dtype=` (<= 4.55)
      * SDPA kernels missing on a given HIP arch/dtype -> reload eager
        (Flash/mem-efficient attention is per-arch and per-dtype; the eager
        reference path always runs, just slower).
    """
    from transformers import AutoModel

    def load(impl: str) -> nn.Module:
        kw: Dict[str, Any] = {"attn_implementation": impl}
        try:                                                    # transformers >= 4.56
            return AutoModel.from_pretrained(model_id, dtype=torch.float32, **kw)
        except TypeError:                                       # transformers <= 4.55
            return AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, **kw)

    try:
        return load(attn_impl)
    except Exception as exc:                                    # noqa: BLE001
        msg = f"{exc}".lower()
        attention_problem = any(k in msg for k in ("attn", "attention", "sdpa", "flash"))
        if attn_impl == "eager" or not attention_problem:       # download/auth/etc
            raise
        print(f"  note        : {attn_impl} attention unavailable "
              f"({type(exc).__name__}: {exc}) -> reloading with eager attention")
        return load("eager")


# Short names for the two trunks the recipes use.  A bare `--model qwen` is
# not a valid HF id, so it is resolved here — one place, and the CLI stays
# readable.  Anything else (a full id, a local path) passes through untouched.
MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "smol": "HuggingFaceTB/SmolLM2-360M",
    "smollm": "HuggingFaceTB/SmolLM2-360M",
    "smollm2": "HuggingFaceTB/SmolLM2-360M",
}


def load_base(model_id: str, accel: Accelerator) -> Tuple[Any, nn.Module]:
    """
    Load tokenizer + *trunk* (no LM head: this is a policy, and dropping the
    head saves its parameters outright when embeddings are untied).
    """
    model_id = MODEL_ALIASES.get(model_id.lower(), model_id)
    if model_id == "tiny":
        tok = HashTokenizer(1024)
        trunk = build_tiny_trunk(attn_impl=accel.attn_impl)
        print(f"  base model  : tiny random Qwen2 "
              f"(h={trunk.config.hidden_size}, layers={len(trunk.layers)}, "
              f"attn={accel.attn_impl}, no download)")
        return tok, trunk

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    trunk = _auto_trunk(model_id, accel.attn_impl)
    n_par = sum(p.numel() for p in trunk.parameters())
    print(f"  base model  : {model_id}  ({n_par/1e6:.1f}M params, "
          f"h={trunk.config.hidden_size}, layers={len(trunk.layers)}, "
          f"attn={accel.attn_impl})")
    return tok, trunk


ADAPTER_MODULES = ("fusion", "halt_head", "carry_norm", "lm_head", "task_head",
                   "task_proj")
# Modules that existed in older checkpoints and were REMOVED (the action head
# path and the readiness head).  Their keys are dropped on load, with a notice.
REMOVED_MODULES = ("action_head.", "action_norm.", "ready_head.")


def filter_legacy_state(state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Split a saved `trainable_state` into (loadable, dropped legacy keys)."""
    keep: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in state.items():
        (dropped.append(k) if k.startswith(REMOVED_MODULES) else keep.__setitem__(k, v))
    return keep, dropped


def load_trainable_state(model: nn.Module, payload: Dict[str, Any],
                         qbind: Optional["QuestionBinding"] = None,
                         where: str = "checkpoint") -> Tuple[List[str], List[str]]:
    """
    Load `payload["trainable_state"]` with strict=False: legacy keys of removed
    modules are DROPPED (printed), missing/unexpected keys are PRINTED, and a
    checkpoint trained on a different question bank / label set is flagged
    loudly — its answers were read from other token ids, so they would be
    silently meaningless.  Returns (missing, unexpected) after the drop.
    """
    state, dropped = filter_legacy_state(payload.get("trainable_state", {}))
    if dropped:
        print(f"  [load] {where}: dropped {len(dropped)} key(s) of removed modules "
              f"(action head path): {sorted({k.split('.')[0] for k in dropped})}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = set(payload.get("trainable", []))
    miss_tr = [k for k in missing if k in trainable or k.startswith(ADAPTER_MODULES)]
    if miss_tr:
        print(f"  [load] {where}: {len(miss_tr)} adapter key(s) not in the checkpoint "
              f"(start fresh): {miss_tr[:8]}{' …' if len(miss_tr) > 8 else ''}")
    if unexpected:
        print(f"  [load] {where}: {len(unexpected)} key(s) ignored (no such module): "
              f"{list(unexpected)[:8]}{' …' if len(unexpected) > 8 else ''}")
    saved = payload.get("qbind")
    if qbind is not None:
        cur = qbind.signature()
        if saved is None:
            print(f"  [load] WARNING: {where} records no question bank/labels (older "
                  f"format): answers were trained on the OLD option-name tokens; "
                  f"the Jev A/B/0/1 labels read other logits — retrain before use.")
        elif json.dumps(saved, sort_keys=True) != json.dumps(cur, sort_keys=True):
            diff = [k for k in ("bank", "tags", "option_ids")
                    if json.dumps(saved.get(k), sort_keys=True)
                    != json.dumps(cur.get(k), sort_keys=True)]
            print(f"  [load] WARNING: {where} was trained with a DIFFERENT question "
                  f"bank/labelling (differs in: {', '.join(diff)}); the answers are "
                  f"read from other tokens than the ones trained.")
    return list(missing), list(unexpected)


def configure_trainable(model: LoopedACTTransformer, mode: str) -> List[str]:
    """
    adapters  : injection + ACT head + policy head only   (fastest, safest)
    lora      : adapters + a low-rank update on EVERY trunk projection
                (~8.8M params at rank 16 on a 0.5B trunk — the whole model
                adapts while the pretrained weights stay frozen, so narration
                and world knowledge are not overwritten)
    recurrent : + the R looped trunk layers + final norm  (recommended upgrade)
    full      : everything (≈8 GB of AdamW state for a 0.5B trunk — will OOM
                on an 8 GB UMA carve-out; listed for completeness)
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for name in ADAPTER_MODULES:
        mod = getattr(model, name, None)          # q_head is gone: answers are read-outs
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad_(True)
    if model.cycle_emb is not None:               # learned per-cycle step tags
        model.cycle_emb.requires_grad_(True)
    model.cycle_gate.requires_grad_(True)         # gate on the sinusoidal tag
    model.carry_tag.requires_grad_(True)          # latent-memory tag
    if mode == "lora":
        n = 0
        for name, p in model.named_parameters():  # only the low-rank factors
            if name.endswith((".lora_a", ".lora_b")):
                p.requires_grad_(True)
                n += p.numel()
        if n == 0:
            raise SystemExit("--trainable lora needs --lora-rank > 0 "
                             "(nothing was injected into the trunk)")
    if mode in ("recurrent", "full"):
        for p in model.loop_layers.parameters():
            p.requires_grad_(True)
        for p in model.trunk.norm.parameters():
            p.requires_grad_(True)
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    return [n for n, p in model.named_parameters() if p.requires_grad]


def report_params(model: LoopedACTTransformer, accel: Accelerator) -> None:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters  : {tot/1e6:.1f}M total · {tr/1e6:.2f}M trainable "
          f"({100*tr/max(tot,1):.2f}%)")
    print(f"  fp32 weights: {tot*4/2**30:.2f} GiB resident · AdamW state on "
          f"trainable ≈ {tr*8/2**30:.2f} GiB")
    if accel.is_gpu and tot * 4 / 2**30 > accel.total_gb * 0.5:
        print("  WARNING     : weights alone exceed half the memory pool — "
              "expect OOM on 8 GB; use a smaller base model.")


# =============================================================================
# 6 · OOM-SAFE OPTIMISATION STEP
# =============================================================================
_OOM_MARKERS = ("out of memory", "outofresources", "out_of_resources",
                "failed to allocate", "not enough memory", "dml_exception",
                "insufficient memory")


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(m in msg for m in _OOM_MARKERS)


def _sync(accel: Accelerator) -> None:
    """Block until the device is idle — only for `--trace` timings, which are
    otherwise measuring how fast the CPU can queue kernels."""
    if accel.backend in ("rocm", "cuda"):
        torch.cuda.synchronize()


def free_device_cache(accel: Accelerator) -> None:
    """Drop Python garbage (cycles keep tensors alive) then the device cache."""
    gc.collect()
    if accel.backend in ("rocm", "cuda"):
        torch.cuda.empty_cache()


def freeze_gc() -> int:
    """
    Move every live object into the permanent generation, once setup is done.

    MEASURED on this box (Ryzen AI 9 365 / Radeon 880M, torch 2.12+rocm7.14.1):
    the process tracks ~449k objects (torch modules, the HF trunk, mujoco's
    model/data, numpy buffers), so a *full* `gc.collect()` costs **215-261 ms**
    — 3-8x the cost of a policy forward, and 78 % of a bridge frame when it was
    called per frame.  Gen-0/gen-1 collections are free (0.00 ms) by comparison,
    and after `gc.freeze()` even a full gen-2 pass drops to 0.00 ms, because the
    long-lived graph is skipped instead of re-scanned.

    Call once, after the model and the scene exist (before the first step), so
    only per-step temporaries stay under GC.  `free_device_cache()` still does a
    full collect on the OOM path.
    """
    gc.collect()                      # construction garbage first, then freeze
    gc.freeze()
    return gc.get_freeze_count()


class MemoryGuard:
    """
    Keeps a UMA iGPU (and its desktop) alive under memory pressure:

      1. low-water check  -> empty the caching allocator before it fragments
      2. OOM on a step    -> free + retry the same batch
      3. OOM again        -> split the batch into micro-batches (grad accumulation)
      4. OOM again        -> skip the batch, count it, and warn with the exact
                             knobs to turn down.  Never raises, never freezes.
      5. every `gc_every` steps -> `reclaim()`: Python GC + allocator flush, so
                             this process never sits on freed-but-unreleased
                             blocks the DWM might need.  Both halves are gated
                             by the same counter (see `reclaim`), and the
                             long-lived graph is frozen out of GC entirely by
                             `freeze_gc()`, so this is cheap when it does run.
    """

    def __init__(self, accel: Accelerator, max_micro: int = 8,
                 low_water_mb: int = 512, gc_every: int = 0):
        self.accel = accel
        self.max_micro = max_micro
        self.low_water = low_water_mb * 2 ** 20
        self.gc_every = max(0, int(gc_every))
        self.oom_events = 0
        self.skipped = 0
        self.reclaims = 0

    def reclaim(self, step: int = 0, force: bool = False) -> bool:
        """
        Periodic hygiene for a shared-memory iGPU.  Returns True if the caching
        allocator was emptied.

        MEASURED (880M, 449k tracked objects): `gc.collect()` = **215-261 ms**,
        `empty_cache()` = 0.01 ms, `mem_get_info()` = 0.01 ms.  The explicit
        full collection is the whole cost — and it was being paid *per frame* in
        the bridge/RL loops, which is why they ran at 3 Hz instead of ~13 Hz.
        Both halves are therefore gated by the same `gc_every` counter
        (`--gc-every 0`, the default, disables the explicit pass entirely).

        Safety without it: Python's automatic generational collector still runs
        (gen-0/gen-1 are free), the allocator cap in `verify_hardware` bounds the
        pool, `flush_if_low()` still fires on real pressure, and `freeze_gc()`
        makes any automatic gen-2 pass skip the model/scene graph.  `force=True`
        (eval / checkpoint / shutdown boundaries) always does both.
        """
        if self.accel.backend not in ("rocm", "cuda"):
            return False
        if not force and (self.gc_every <= 0 or step % self.gc_every != 0):
            return False
        gc.collect()
        torch.cuda.empty_cache()
        self.reclaims += 1
        return True

    def flush_if_low(self) -> bool:
        if self.accel.backend not in ("rocm", "cuda"):
            return False
        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:                                    # pragma: no cover
            return False
        if free < self.low_water:
            free_device_cache(self.accel)
            return True
        return False

    @staticmethod
    def _split(ep: Episode, device: torch.device) -> Optional[List[Episode]]:
        if ep.sensors.shape[0] < 2:
            return None
        return [ep.half(0).to(device), ep.half(1).to(device)]

    def optimize(self, fn, episode: Episode, device: torch.device):
        micro: List[Episode] = [episode]
        notes: List[str] = []
        for attempt in range(4):
            try:
                stats = fn(micro)
                if notes:
                    stats["oom"] = "; ".join(notes)
                return stats
            except BaseException as exc:                       # noqa: BLE001
                if not is_oom(exc):
                    raise
                self.oom_events += 1
                free_device_cache(self.accel)
                if attempt == 0:
                    notes.append("OOM→cache flush+retry")
                elif len(micro) < self.max_micro:
                    split = [self._split(m, device) for m in micro]
                    if any(s is None for s in split):
                        notes.append("OOM→batch already minimal")
                    else:
                        micro = [e for s in split for e in s]
                        notes.append(f"OOM→{len(micro)} micro-batches")
                else:
                    self.skipped += 1
                    print(f"  !! OOM: skipping batch ({self.skipped} skipped). "
                          f"Lower --batch/--window or use --grad-checkpoint.")
                    return {"skipped": True, "loss": float("nan"),
                            "oom": "; ".join(notes)}
        return {"skipped": True, "loss": float("nan"), "oom": "; ".join(notes)}


# =============================================================================
# 7 · LOSS · OPTIMISER · TRAIN / EVAL LOOP
# =============================================================================
def _class_weights(lab: torch.Tensor, n_out: int) -> Optional[torch.Tensor]:
    """
    Inverse-frequency weights for a categorical question, mean-normalised.

    The axis questions are "stay" for 4 of the expert's 6 actions, so an
    unweighted CE converges to that majority class, and a policy that drives
    the arm from the answers then brakes forever.  Measured on a live frame:
    `axis_y = stay 0.682` while the expert (and the old action head reading the
    SAME latent) said `+y`.
    Returns None when a class is missing from the batch (the plain CE is then
    the honest thing to optimise) and caps the ratio so one rare sample cannot
    dominate the step.
    """
    flat = lab.reshape(-1)
    counts = torch.bincount(flat.clamp(min=0), minlength=n_out).float()
    if bool((counts == 0).any()):
        return None
    w = counts.sum() / (counts * n_out)
    return w.clamp(max=8.0).to(flat.device)


def joint_loss(out: LoopOutput, tau: float, *,
               q_labels: Optional[Dict[str, torch.Tensor]] = None,
               q_weight: float = 1.0,
               speech_ce: Optional[torch.Tensor] = None,
               text_weight: float = 0.0,
               task_ids: Optional[torch.Tensor] = None,
               task_weight: float = 0.0,
               q_class_weights: Optional[Dict[str, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Total = τ · Ponder  +  q_weight · Σ questions  +  text_weight · speech CE
            +  task_weight · task CE

    There is no action head: the control objective IS the questionnaire (the
    expert actions only produce its labels, `LABEL_SOURCES`), and the action
    is decoded from the answers by `json_action`.

    * Ponder — Graves' differentiable compute cost; its gradient pushes
      σ(halt_head) UP, buying fewer loop cycles, until halting early costs more
      cross-entropy than it saves.  That balance *is* the adaptive computation.
    * Questions — one CE per typed question, over the model's own next-token
      distribution restricted to that question's Jev label tokens (see
      `question_logits`): the answer is read off the LM head at the question's
      marker, so the questionnaire is DATA and no per-question parameters exist.
      The CE is class-balanced (`_class_weights`): the axis questions are
      "stay" for 4 of 6 expert actions, and an unweighted CE answers `stay`
      everywhere — which, driving the arm from the JSON, brakes forever.
    * Speech — next-token CE of the event sentence (+ end token) generated
      AFTER the state, from the last state position.  Arrives pre-reduced
      (`LoopedACTTransformer.speech_ce`), because scoring it here would need
      the full (B, Ts, V) logits in fp32 at once.
    * Task — the verb classifier on the enriched read-out latent.

    Returns (total, parts) — every term separately, for logging.
    """
    parts: Dict[str, torch.Tensor] = {}
    parts["ponder"] = out.ponder.mean()
    zero = parts["ponder"].new_zeros(())

    q_terms: List[torch.Tensor] = []
    if out.q_logits and q_labels:
        for qid, lg in out.q_logits.items():
            lab = q_labels.get(qid)
            if lab is None:
                continue
            # given per-option weights (LIBERO: meta.json, dataset-level) replace
            # the in-batch inverse-frequency balance; synthetic batches pass none
            w = (q_class_weights or {}).get(qid)
            q_terms.append(F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), lab.reshape(-1),
                weight=(w.to(lg.device, torch.float32) if w is not None
                        else _class_weights(lab, lg.shape[-1]))))
    parts["questions"] = torch.stack(q_terms).mean() if q_terms else zero

    parts["speech"] = (speech_ce if (speech_ce is not None and text_weight > 0.0)
                       else zero)

    if out.task_logits is not None and task_ids is not None and task_weight > 0.0:
        # The verb is carried ONLY by the prompt, so this CE cannot be reduced
        # by the sensor path — its gradient has to travel through `task_attn`.
        Bt, St, _ = out.task_logits.shape
        tgt = task_ids.reshape(-1)[:, None].expand(Bt, St).reshape(-1)
        # a row without a verb (task id < 0: LIBERO) is MASKED, not clamped onto
        # verb 0; sum / count so an all-masked batch is 0, not 0/0
        valid = tgt >= 0
        tgt = torch.where(valid, tgt.clamp(max=len(TASKS) - 1), torch.full_like(tgt, -100))
        parts["task"] = F.cross_entropy(
            out.task_logits.reshape(-1, out.task_logits.shape[-1]), tgt,
            ignore_index=-100, reduction="sum") / valid.sum().clamp_min(1)
    else:
        parts["task"] = zero

    total = (tau * parts["ponder"]
             + q_weight * parts["questions"]
             + text_weight * parts["speech"]
             + task_weight * parts["task"])
    return total, parts


def make_step_fn(model, opt, scaler, accel, tau: float, clip: float,
                 q_weight: float = 1.0, text_weight: float = 0.0,
                 task_weight: float = 0.0,
                 trace: int = 0, speak_tokens: int = 1):
    """
    `trace` (steps) prints a synchronised timing line per phase of the first
    few steps.  Without it a step that spends its time in one CUDA op — or in
    the allocator — is indistinguishable from a step that is merely slow: the
    log shows nothing until the step ends.
    """
    tstate = {"step": 0}
    # Diagnostic (no gradient): CE / accuracy of the NUMBER-dependent questions
    # split by whether that row's options were shuffled.  If the shuffle is
    # being learned, "shuf" falls toward "plain"; a delayed jump shows here
    # long before it moves the default-interface eval.  Read + reset by the
    # logger through `_step.numq`.  LIBERO rows go to "lib_plain"/"lib_shuf"
    # (synthetic keeps the plain names, comparable with the longvary log).
    numq = {k: [0.0, 0, 0] for k in ("plain", "shuf", "lib_plain", "lib_shuf")}

    def _step(micro: Sequence[Episode], carry: Optional[torch.Tensor]) -> Dict[str, Any]:
        tstate["step"] += 1
        show = tstate["step"] <= trace
        opt.zero_grad(set_to_none=True)
        n = len(micro)
        agg: Dict[str, float] = {}
        for i, mb in enumerate(micro):
            t_phase = time.time()
            if show:
                _sync(accel)
                print(f"    [trace] micro {i+1}/{n} · forward "
                      f"(instr {tuple(mb.instr_ids.shape)} "
                      f"state {tuple(mb.state_ids.shape)})", flush=True)
            # the latent carry is only meaningful for an undivided batch: a split
            # batch would need one carry per half (dropped here on purpose)
            c = carry if len(micro) == 1 else None
            with torch.amp.autocast(device_type=accel.amp_device_type,
                                    dtype=accel.dtype, enabled=accel.amp):
                sio = mb.speech_io(speak_tokens)
                sp_in, sp_mask, sp_tgt = sio if sio is not None else (None, None, None)
                out = model(mb.instr_ids, mb.instr_mask, mb.state_ids, mb.ans_pos,
                            carry=c, qbind=mb.qbind, speech_ids=sp_in,
                            speech_mask=sp_mask)
                if mb.opt_perm:                  # shuffled labels -> canonical order
                    out.q_logits = canonical_q_logits(out.q_logits, mb.opt_perm)
                if show:
                    _sync(accel)
                    print(f"    [trace]   forward {time.time()-t_phase:.2f}s · "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}G "
                          f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G",
                          flush=True)
                    t_phase = time.time()
                q_labels = episode_labels(mb, mb.qbind.bank if mb.qbind is not None
                                          else model.bank)
                sce = (model.speech_ce(out.speech_h, sp_tgt)
                       if (text_weight > 0.0 and sp_tgt is not None
                           and out.speech_h is not None) else None)
                loss, parts = joint_loss(
                    out, tau, q_labels=q_labels, q_weight=q_weight,
                    speech_ce=sce, text_weight=text_weight,
                    task_ids=mb.tasks, task_weight=task_weight,
                    q_class_weights=mb.q_weights)
                with torch.no_grad():
                    for qid in NUMERIC_QIDS:
                        lg_d, lab_d = out.q_logits.get(qid), q_labels.get(qid)
                        if lg_d is None or lab_d is None:
                            continue
                        K_d = lg_d.shape[-1]
                        ce_d = F.cross_entropy(lg_d.float().reshape(-1, K_d),
                                               lab_d.reshape(-1), reduction="none"
                                               ).view(lab_d.shape[0], -1)
                        hit = (lg_d.argmax(-1).reshape(lab_d.shape[0], -1)
                               == lab_d.reshape(lab_d.shape[0], -1))
                        perm_d = (mb.opt_perm or {}).get(qid)
                        if perm_d is None:
                            sh = torch.zeros(lab_d.shape[0], dtype=torch.bool,
                                             device=ce_d.device)
                        else:
                            p_d = perm_d.to(ce_d.device)
                            sh = (p_d != torch.arange(p_d.shape[1], device=p_d.device)
                                  ).any(-1)
                        pre = "lib_" if getattr(mb, "source", "") == "libero" else ""
                        for key, m in ((pre + "shuf", sh), (pre + "plain", ~sh)):
                            if bool(m.any()):
                                numq[key][0] += float(ce_d[m].sum())
                                numq[key][1] += int(hit[m].sum())
                                numq[key][2] += int(ce_d[m].numel())
            if show:
                _sync(accel)
                print(f"    [trace]   loss {time.time()-t_phase:.2f}s · "
                      f"total={float(loss.detach()):.4f}", flush=True)
                t_phase = time.time()
            (loss / n).backward() if scaler is None else scaler.scale(loss / n).backward()
            if show:
                _sync(accel)
                print(f"    [trace]   backward {time.time()-t_phase:.2f}s · "
                      f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}G",
                      flush=True)
                t_phase = time.time()
            # speaking frame = the ACTIVE utterance is longer than the lone END
            # (targets are masked to this frame's K tokens, so count the source)
            speak = (float(((mb.speech_ids[:, -1] >= 0).sum(1) > 1).float().mean())
                     if (sp_tgt is not None and mb.speech_ids is not None) else 0.0)
            for k, v in (("loss", float(loss.detach())),
                         ("ponder", float(parts["ponder"].detach())),
                         ("q_loss", float(parts["questions"].detach())),
                         ("speech_loss", float(parts["speech"].detach())),
                         ("speak_frac", speak),
                         ("task_loss", float(parts["task"].detach())),
                         ("mass_err", out.stats["mass_err"]),
                         ("mean_cycles", out.stats["mean_cycles"]),
                         ("mean_ponder", out.stats["mean_ponder"]),
                         ("halted_frac", out.stats["halted_frac"]),
                         ("latent_norm", out.stats["latent_norm"])):
                agg[k] = agg.get(k, 0.0) + v / n
            if i == 0:
                agg["carry"] = out.carry if len(micro) == 1 else None
                agg["cycle_usage"] = out.stats["cycle_usage"]
        if scaler is not None:
            # UNSCALE FIRST: under a GradScaler the gradients carry a 2**16
            # factor, so squaring them to measure the norm overflows fp32 and
            # the logged `gn` is nan (the clipping itself was always correct,
            # because clip_grad_norm_ recomputes the norm from the grads).
            scaler.unscale_(opt)
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        pre_clip = float(sum(float(p.grad.detach().pow(2).sum()) for p in params) ** 0.5)
        # A non-finite `gn` is the fp16 loss scale fighting an overflow: the
        # scale multiplies the loss, so an overflowing gradient is expected
        # until the scaler has halved its way down.  What matters is whether it
        # SETTLES (`scale` stops falling) and whether any step is skipped, so
        # both are reported next to the norm instead of being guessed at.
        agg["grad_bad"] = float(sum(1 for p in params
                                     if not bool(torch.isfinite(p.grad).all())))
        agg["loss_scale"] = float(scaler.get_scale()) if scaler is not None else 0.0
        torch.nn.utils.clip_grad_norm_(params, clip)
        if scaler is None:
            opt.step()
        else:
            scaler.step(opt)
            scaler.update()
        agg["grad_norm"] = pre_clip              # measured BEFORE clipping
        return agg
    _step.numq = numq
    return _step


def run_train(cfg, accel, model, tok, world,
              libero: Optional["LiberoData"] = None) -> None:
    qbind = model.qbind                    # the questionnaire lives in the prompt
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=0.01, eps=1e-8)
    warmup = max(1, int(0.05 * cfg.steps))
    # --lr-hold F keeps the peak for F·steps after warm-up before the cosine
    # starts; --lr-floor is where the cosine ends (fraction of --lr).  The old
    # schedule is hold 0 / floor 0.05.
    hold_end = warmup + int(max(0.0, getattr(cfg, "lr_hold", 0.0)) * cfg.steps)
    floor = float(getattr(cfg, "lr_floor", 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 1.0 if s < hold_end else
        floor + (1 - floor) * 0.5 * (1 + math.cos(
            math.pi * (s - hold_end) / max(1, cfg.steps - hold_end))))
    scaler = None
    if accel.amp and accel.scaler:
        try:
            scaler = torch.amp.GradScaler(accel.amp_device_type)
        except (TypeError, ValueError):                      # older torch
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    guard = MemoryGuard(accel, gc_every=cfg.gc_every)
    K_sp = int(getattr(cfg, "speak_tokens", 1))
    vary = float(getattr(cfg, "vary", 0.0) or 0.0)
    # `--vary`: the sampler draws each stream episode's interface (bank subset /
    # order + state format per batch, paraphrase / option order / layout per
    # row).  None keeps the exact pre-`--vary` code path.
    sampler = (InterfaceSampler(qbind, tok, prob=vary, seed=cfg.seed + 7919)
               if (vary > 0.0 and qbind is not None) else None)
    stream = EpisodeStream(world, cfg.batch, cfg.window, cfg.windows_per_episode,
                           tok, carry=cfg.carry, qbind=qbind,
                           speak_tokens=K_sp, interface=sampler)
    # `--libero`: LIBERO windows come in their OWN batches (their own stream,
    # carry and per-step token length), a deterministic `--libero-frac` share
    # of the steps — so no batch ever mixes two state lengths.
    lib_frac = float(getattr(cfg, "libero_frac", 0.0) or 0.0)
    lib_stream = None
    if libero is not None and lib_frac > 0.0 and qbind is not None:
        lib_stream = EpisodeStream(
            LiberoWorld(libero, "train", seed=cfg.seed + 31, window=cfg.window),
            cfg.batch, cfg.window, cfg.windows_per_episode, tok, carry=cfg.carry,
            qbind=qbind, speak_tokens=K_sp,
            interface=(InterfaceSampler(qbind, tok, prob=vary, seed=cfg.seed + 104729)
                       if vary > 0.0 else None))
    elif libero is not None and lib_frac > 0.0:
        print("  [libero] WARNING: --questions off — LIBERO rows need the "
              "questionnaire; training synthetic-only")
    lib_acc, n_lib, n_var = 0.0, 0, 0
    step_fn = make_step_fn(model, opt, scaler, accel, cfg.tau, cfg.clip,
                           q_weight=cfg.q_weight, text_weight=cfg.text_weight,
                           task_weight=cfg.task_weight,
                           trace=max(0, int(getattr(cfg, "trace", 0))),
                           speak_tokens=int(getattr(cfg, "speak_tokens", 1)))
    device = accel.device
    frozen = freeze_gc()                 # model/stream graph out of the GC scan

    print("=" * 78)
    print("TRAINING")
    print("=" * 78)
    print(f"  steps={cfg.steps}  batch={cfg.batch}  window={cfg.window} steps/window "
          f"· {cfg.windows_per_episode} windows/episode  lr={cfg.lr}  tau={cfg.tau}")
    print(f"  loops<= {cfg.loops}  recurrent_layers={cfg.recurrent_layers}  "
          f"carry={cfg.carry}  grad_checkpoint={cfg.grad_checkpoint}")
    if model.bank is not None:
        print(f"  interface   : JSON answers · {len(model.bank)} question(s) "
              f"[{' '.join(f'{s.qid}:{s.type}' for s in model.bank.specs)}] · "
              f"{model.bank.total} option logits · markers "
              f"{' '.join(model.qbind.markers)} · "
              f"q_weight={cfg.q_weight} · text_weight={cfg.text_weight} · "
              f"task_weight={cfg.task_weight}")
    if sampler is not None or lib_stream is not None:
        print(f"  variation   : --vary {vary:.2f} (bank subset/order + state format "
              f"per batch; paraphrase / option order / layout per row; held-out "
              f"paraphrase set {PARAPHRASE_HELDOUT} + '{HELDOUT_STATE_FORMAT.name}' "
              f"format are eval-only) · LIBERO "
              + (f"{lib_frac:.2f} of the steps from "
                 f"{len(libero.episodes['train'])} train demos" if lib_stream is not None
                 else "off"))
    print(f"  memory      : {accel.note or 'uncapped'} · amp={accel.amp} "
          f"({str(accel.dtype).split('.')[-1]}) · flush every "
          f"{cfg.gc_every if cfg.gc_every else 'low-water only'} batch(es) "
          f"· {frozen} objects frozen out of GC")
    t0 = time.time()
    step = 0
    trace = max(0, int(getattr(cfg, "trace", 0)))
    for step in range(1, cfg.steps + 1):
        guard.flush_if_low()
        t_phase = time.time()
        use_lib = False
        if lib_stream is not None:              # exact share, no RNG consumed
            lib_acc += lib_frac
            if lib_acc >= 1.0 - 1e-9:
                lib_acc -= 1.0
                use_lib = True
        src = lib_stream if use_lib else stream
        episode, carry = src.next(device)
        n_lib += int(use_lib)
        n_var += int(episode.variant != "default")
        if step <= trace:
            print(f"  [trace] step {step}: episode in {time.time()-t_phase:.2f}s "
                  f"(instr {tuple(episode.instr_ids.shape)} "
                  f"state {tuple(episode.state_ids.shape)} "
                  f"ans_pos {tuple(episode.ans_pos.shape)})", flush=True)
            t_phase = time.time()
        stats = guard.optimize(lambda micro, c=carry: step_fn(micro, c), episode, device)
        if step <= trace:
            _sync(accel)
            print(f"  [trace] step {step}: optimize in {time.time()-t_phase:.2f}s "
                  f"skipped={stats.get('skipped', False)} "
                  f"oom={stats.get('oom', '-')}", flush=True)
        if stats.get("skipped"):
            continue
        src.push(stats.pop("carry", None))
        guard.reclaim(step)              # desktop-safe: release after every batch
        sched.step()
        if step % cfg.log_every == 0 or step == 1:
            dt = time.time() - t0
            peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                    if accel.backend in ("rocm", "cuda") else 0.0)
            usage = stats.get("cycle_usage", [])
            extra = ""
            if model.bank is not None:
                extra += f" q={stats.get('q_loss', 0.0):.3f}"
            fusion = getattr(model, "fusion", None)
            if fusion is not None:
                extra += f" gate={float(torch.tanh(fusion.gate.detach())):+.4f}"
            if cfg.text_weight > 0.0:
                extra += f" spk={stats.get('speech_loss', 0.0):.3f}"
            extra += f" speak={stats.get('speak_frac', 0.0):.2f}"   # speaking frames
            if cfg.task_weight > 0.0:
                extra += f" tsk={stats.get('task_loss', 0.0):.3f}"
            if lib_stream is not None:
                extra += f" lib={n_lib}/{step}"
            if sampler is not None:
                extra += f" var={n_var}/{step}"
            nq = getattr(step_fn, "numq", None)
            if nq is not None and any(v[2] for v in nq.values()):
                for key in ("plain", "shuf", "lib_plain", "lib_shuf"):
                    s_, c_, n_ = nq[key]
                    if n_:
                        extra += f" {key}=ce{s_ / n_:.3f}/acc{c_ / n_:.2f}/n{n_}"
                    nq[key][:] = [0.0, 0, 0]
            print(f"  {step:5d}/{cfg.steps}  loss={stats['loss']:.4f} "
                  f"(pov={stats['ponder']:.3f}{extra}) "
                  f"cycles={stats['mean_cycles']:.2f}/{stats['mean_ponder']:.2f} "
                  f"halt={stats['halted_frac']:.2f} "
                  f"|z|={stats['latent_norm']:.1f} mass_err={stats['mass_err']:.2e} "
                  f"gn={stats['grad_norm']:.2f} lr={sched.get_last_lr()[0]:.2e} "
                       f"scale={stats.get('loss_scale', 0.0):.0f} "
                       f"bad={int(stats.get('grad_bad', 0))} "
                  f"{dt/max(step,1):.2f}s/it peak={peak:.2f}G "
                  f"usage={[round(u,2) for u in usage]}")
        if cfg.eval_every and step % cfg.eval_every == 0:
            evaluate(cfg, accel, model, tok, SensorWorld(cfg.seed + 991, world.noise),
                     step, libero=libero)
            guard.reclaim(step, force=True)         # eval reuses the same pool
        save_every = int(getattr(cfg, "ckpt_every", 0) or 0)
        if save_every and step % save_every == 0 and step < cfg.steps:
            save_checkpoint(cfg, model)
            print(f"  [save] step {step} -> {cfg.out}", flush=True)
    guard.reclaim(step, force=True)
    save_checkpoint(cfg, model)
    print(f"  done in {(time.time()-t0)/60:.1f} min   "
          f"(OOM events={guard.oom_events}, skipped batches={guard.skipped}, "
          f"allocator flushes={guard.reclaims})")


@torch.no_grad()
def evaluate(cfg, accel, model, tok, world, step: int,
             libero: Optional["LiberoData"] = None) -> Dict[str, float]:
    """
    The [eval] line.  1 · DEFAULT interface, natural windows with carry (the
    pre-`--vary` metrics, unchanged): json_action_acc (= json_action_acc_default),
    task_acc, task_cls, q_acc, cycles.  2 · the SAME trajectories under the
    HELD-OUT interface (wording set PARAPHRASE_HELDOUT + HELDOUT_STATE_FORMAT,
    never trained on): json_action_acc_heldout.  3 · speech: `decide` on the
    natural windows, `sentence` over their speaking frames PLUS probe windows
    that END on a speaking frame (natural window ends almost never do).
    4 · with `libero`: the eval split — per-question accuracy, JSON-action
    exact match (all of axis_x/axis_y/gripper/intent right: LIBERO moves x and
    y at once and its majority `stay` jaw decodes to brake, so the 6-way table
    is no measure there) and speech on natural + speaking-probe windows.
    """
    qbind = model.qbind
    model.eval()
    K_sp = int(getattr(cfg, "speak_tokens", 1))
    dev = accel.device
    end = speech_end_id(tok)
    use_json = has_json_action(model.bank)
    ratio = (lambda h, n: h / n if n else float("nan"))
    probes = max(2, int(cfg.eval_batches) // 2)
    long_len = cfg.window * cfg.windows_per_episode

    def forward(episode: Episode, carry):
        sio = episode.speech_io(K_sp)
        sp_in, sp_mask, sp_tgt = sio if sio is not None else (None, None, None)
        with torch.amp.autocast(device_type=accel.amp_device_type, dtype=accel.dtype,
                                enabled=accel.amp):
            out = model(episode.instr_ids, episode.instr_mask, episode.state_ids,
                        episode.ans_pos, carry, qbind=episode.qbind,
                        speech_ids=sp_in, speech_mask=sp_mask)
            sp_lg = (model.text_logits(out.speech_h)
                     if (sp_tgt is not None and out.speech_h is not None) else None)
        if episode.opt_perm:
            out.q_logits = canonical_q_logits(out.q_logits, episode.opt_perm)
        return out, sp_lg, sp_tgt

    def speech_update(acc: Dict[str, int], episode: Episode, sp_lg, sp_tgt) -> None:
        if sp_lg is None:
            return
        # teacher-forced argmax on THIS frame's supervised tokens (streamed:
        # tgt is -100 except at the ≤K positions predicting tgt[off:off+K])
        n = min(sp_lg.shape[1], sp_tgt.shape[1])
        pr = sp_lg[:, :n].argmax(-1)
        tg = sp_tgt[:, :n].to(pr.device)
        sup = tg >= 0
        first = sup.float().argmax(1, keepdim=True)          # first supervised pos
        pr0 = pr.gather(1, first)[:, 0]
        tg0 = tg.gather(1, first)[:, 0]
        speaks = ((episode.speech_ids[:, -1] >= 0).sum(1) > 1).to(pr.device)
        # decide = "keep talking vs END" on the frame's first emitted token
        dec = ((pr0 != end) == (tg0 != end)) & sup.any(1)
        exact = ((pr == tg) | ~sup).all(dim=1)
        for k, v in (("dec", int(dec.sum())), ("tot", int(tg.shape[0])),
                     ("dec_spk", int((dec & speaks).sum())),
                     ("sent", int((exact & speaks).sum())), ("spk", int(speaks.sum()))):
            acc[k] = acc.get(k, 0) + v

    def q_update(hit: Dict[str, int], tot: Dict[str, int], episode: Episode,
                 out) -> None:
        bank = episode.qbind.bank if episode.qbind is not None else model.bank
        if not out.q_logits or bank is None:
            return
        labels = episode_labels(episode, bank)
        for qid, lg in out.q_logits.items():
            lab = labels.get(qid)
            if lab is None:
                continue
            lab = lab.to(lg.device)
            if lg.shape[-1] == 1:                     # noul: probability of "yes"
                pred = (torch.sigmoid(lg) >= 0.5).long()
            else:
                pred = lg.argmax(-1)
            hit[qid] = hit.get(qid, 0) + int((pred == lab).sum())
            tot[qid] = tot.get(qid, 0) + lab.numel()

    def probe_speech(acc: Dict[str, int], src, rng: random.Random) -> None:
        for _ in range(probes):
            long = collate(src.episode(cfg.batch, long_len), tok, qbind,
                           speak_tokens=K_sp)
            probe = speech_probe(long, cfg.window, rng)
            if probe is None:
                return
            pe = probe.to(dev)
            out, sp_lg, sp_tgt = forward(pe, None)            # cold start: no carry
            speech_update(acc, pe, sp_lg, sp_tgt)

    # ── 1 · DEFAULT interface, natural windows ───────────────────────────────
    stream = EpisodeStream(world, cfg.batch, cfg.window, cfg.windows_per_episode,
                           tok, carry=cfg.carry, qbind=qbind, speak_tokens=K_sp)
    tot = correct = 0
    q_hit: Dict[str, int] = {}
    q_tot: Dict[str, int] = {}
    task_hit: Dict[int, int] = {}
    task_tot: Dict[int, int] = {}
    cls_hit = cls_tot = 0                       # task_head verb accuracy
    sp_nat: Dict[str, int] = {}                 # speech on natural windows
    sp_prb: Dict[str, int] = {}                 # speech on speaking probes
    cycles: List[torch.Tensor] = []
    difficulty: List[torch.Tensor] = []
    for _ in range(cfg.eval_batches):
        episode, carry = stream.next(dev)
        out, sp_lg, sp_tgt = forward(episode, carry)
        stream.push(out.carry)
        if use_json and out.q_logits:
            # HEADLINE: the action decoded from the JSON answers (json_action)
            pred_a = json_action(out.q_logits).to(episode.actions.device)
            hit = (pred_a == episode.actions)                           # (B,S)
            correct += int(hit.sum())
            tot += episode.actions.numel()
            if episode.tasks is not None:
                # per-task accuracy: with four verbs sharing one 28-D observation,
                # the aggregate number hides whether the instruction is bound at all
                for k in range(len(TASKS)):
                    m = episode.tasks.to(hit.device) == k
                    if bool(m.any()):
                        task_hit[k] = task_hit.get(k, 0) + int(hit[m].sum())
                        task_tot[k] = task_tot.get(k, 0) + int(m.sum()) * hit.shape[1]
        if out.task_logits is not None and episode.tasks is not None:
            tg = episode.tasks.to(out.task_logits.device)[:, None].expand(
                -1, out.task_logits.shape[1])
            cls_hit += int((out.task_logits.argmax(-1) == tg).sum())
            cls_tot += tg.numel()
        speech_update(sp_nat, episode, sp_lg, sp_tgt)
        q_update(q_hit, q_tot, episode, out)
        cycles.append(out.cycles.mean(dim=-1))                  # (B,) cycles/sample
        difficulty.append(episode.goal_dist.mean(dim=-1))        # (B,) mean |goal-pos|

    # ── 2 · HELD-OUT interface on the same trajectories ─────────────────────
    ho_hit = ho_tot = 0
    hq_hit: Dict[str, int] = {}
    hq_tot: Dict[str, int] = {}
    if use_json and qbind is not None and isinstance(world, SensorWorld):
        # type(world): the same world class (a SensorWorld subclass in the self-test)
        ho = EpisodeStream(type(world)(world.seed, world.noise, world.reach), cfg.batch,
                           cfg.window, cfg.windows_per_episode, tok, carry=cfg.carry,
                           qbind=qbind, speak_tokens=K_sp,
                           interface=InterfaceSampler(qbind, tok, mode="heldout"))
        try:
            for _ in range(cfg.eval_batches):
                episode, carry = ho.next(dev)
                out, _, _ = forward(episode, carry)
                ho.push(out.carry)
                if out.q_logits:
                    pred_a = json_action(out.q_logits).to(episode.actions.device)
                    ho_hit += int((pred_a == episode.actions).sum())
                    ho_tot += episode.actions.numel()
                q_update(hq_hit, hq_tot, episode, out)
        except ValueError as exc:            # held-out state format not fixed-width
            print(f"  [eval] held-out interface skipped: {exc}")
            ho_hit = ho_tot = 0
            hq_hit.clear()
            hq_tot.clear()

    # ── 3 · speech probes (only when the model speaks at all) ───────────────
    if sp_nat.get("tot") and isinstance(world, SensorWorld):
        probe_speech(sp_prb, type(world)(world.seed + 7, world.noise, world.reach),
                     random.Random(world.seed + 11))

    # ── 4 · LIBERO eval split ───────────────────────────────────────────────
    lib: Dict[str, float] = {}
    l_str = ""
    lw = None
    if libero is not None and qbind is not None and libero.episodes.get("eval"):
        try:
            lw = LiberoWorld(libero, "eval", seed=cfg.seed + 977, window=cfg.window)
            lsw = LiberoWorld(libero, "eval", seed=cfg.seed + 1597, window=cfg.window,
                              speaking=True)
        except ValueError as exc:
            print(f"  [eval] LIBERO eval skipped: {exc}")
            lw = None
    if lw is not None:
        ls = EpisodeStream(lw, cfg.batch, cfg.window, cfg.windows_per_episode, tok,
                           carry=cfg.carry, qbind=qbind, speak_tokens=K_sp)
        lq_hit: Dict[str, int] = {}
        lq_tot: Dict[str, int] = {}
        lj_hit = lj_tot = 0
        sl_nat: Dict[str, int] = {}
        sl_prb: Dict[str, int] = {}
        for _ in range(cfg.eval_batches):
            episode, carry = ls.next(dev)
            out, sp_lg, sp_tgt = forward(episode, carry)
            ls.push(out.carry)
            q_update(lq_hit, lq_tot, episode, out)
            speech_update(sl_nat, episode, sp_lg, sp_tgt)
            labels = episode_labels(episode, model.bank) or {}
            have = [q for q in JSON_ACTION_QIDS if q in (out.q_logits or {}) and q in labels]
            if have:
                ok = torch.stack([out.q_logits[q].argmax(-1)
                                  == labels[q].to(out.q_logits[q].device)
                                  for q in have]).all(0)
                lj_hit += int(ok.sum())
                lj_tot += ok.numel()
        if sl_nat.get("tot"):
            probe_speech(sl_prb, lsw, random.Random(cfg.seed + 2203))
        l_spk = sl_nat.get("spk", 0) + sl_prb.get("spk", 0)
        lib = {"libero_json_action_acc": ratio(lj_hit, lj_tot),
               "libero_speech_decide_acc": ratio(sl_nat.get("dec", 0), sl_nat.get("tot", 0)),
               "libero_speech_sentence_acc": ratio(sl_nat.get("sent", 0)
                                                   + sl_prb.get("sent", 0), l_spk)}
        lib.update({f"libero_q_acc_{k}": lq_hit[k] / max(lq_tot[k], 1) for k in lq_hit})
        l_str = (f"  libero: json_action_acc={lib['libero_json_action_acc']:.3f}"
                 + (" q_acc=" + " ".join(f"{k}={lq_hit[k]/max(lq_tot[k],1):.2f}"
                                         for k in lq_hit) if lq_hit else "")
                 + (f" speech: decide={lib['libero_speech_decide_acc']:.2f} "
                    f"sentence={lib['libero_speech_sentence_acc']:.2f} (speaking "
                    f"{sl_nat.get('spk', 0)}/{sl_nat.get('tot', 0)} + probe "
                    f"{sl_prb.get('spk', 0)}/{sl_prb.get('tot', 0)})"
                    if sl_nat.get("tot") else ""))
    model.train()

    acc = correct / max(tot, 1)
    c = torch.cat(cycles).float()
    d = torch.cat(difficulty)
    mid = d.median()
    easy_mask, hard_mask = d <= mid, d > mid
    easy = float(c[easy_mask].mean()) if bool(easy_mask.any()) else float("nan")
    hard = float(c[hard_mask].mean()) if bool(hard_mask.any()) else float("nan")
    q_str = ("  q_acc=" + " ".join(f"{k}={q_hit[k]/max(q_tot[k],1):.2f}"
                                   for k in q_hit)) if q_hit else ""
    t_str = ("  task_acc=" + " ".join(
        f"{TASKS[k]}={task_hit.get(k,0)/max(task_tot.get(k,1),1):.2f}"
        for k in range(len(TASKS)) if task_tot.get(k))) if task_tot else ""
    cls = cls_hit / cls_tot if cls_tot else float("nan")
    c_str = f"  task_cls={cls:.2f}" if cls_tot else ""
    n_spk = sp_nat.get("spk", 0) + sp_prb.get("spk", 0)
    sp_dec = ratio(sp_nat.get("dec", 0), sp_nat.get("tot", 0))
    sp_sent = ratio(sp_nat.get("sent", 0) + sp_prb.get("sent", 0), n_spk)
    sp_dec_spk = ratio(sp_nat.get("dec_spk", 0) + sp_prb.get("dec_spk", 0), n_spk)
    s_str = (f"  speech: decide={sp_dec:.2f} decide_spk={sp_dec_spk:.2f} "
             f"sentence={sp_sent:.2f} (speaking {sp_nat.get('spk', 0)}/"
             f"{sp_nat.get('tot', 0)} + probe {sp_prb.get('spk', 0)}/"
             f"{sp_prb.get('tot', 0)})") if sp_nat.get("tot") else ""
    ho_acc = ratio(ho_hit, ho_tot)
    a_str = (f"json_action_acc={acc:.3f}" if tot
             else "json_action_acc=n/a (no axis/jaw questions)")
    if ho_tot:
        a_str += f" json_action_acc_heldout={ho_acc:.3f}"
    print(f"  [eval {step:5d}] {a_str}{t_str}{c_str}{q_str}{s_str}{l_str}  "
          f"cycles_easy={easy:.2f} "
          f"cycles_hard={hard:.2f}  (harder windows should get more cycles once the "
          f"policy fits)")
    res = {"acc": acc, "json_action_acc": acc,
           "json_action_acc_default": acc if tot else float("nan"),
           "json_action_acc_heldout": ho_acc, "task_cls": cls,
           "speech_decide_acc": sp_dec, "speech_sentence_acc": sp_sent,
           "speech_decide_speaking_acc": sp_dec_spk,
           "speech_speaking_frames": float(n_spk),
           "cycles_easy": easy, "cycles_hard": hard}
    res.update({f"heldout_q_acc_{k}": hq_hit[k] / max(hq_tot[k], 1) for k in hq_hit})
    res.update(lib)
    return res


def interface_record(qbind: Optional["QuestionBinding"], cfg=None
                     ) -> Optional[Dict[str, Any]]:
    """
    What a deployment needs to rebuild the DEFAULT interface the policy is
    served with: bank JSON (ids, types, options, wording), schema text,
    markers, Jev labels, state sentence format.  `--vary` renderings are
    training-only and deliberately not stored: bridge and live stack serve the
    default rendering (the one `json_action_acc` measures).
    """
    if qbind is None:
        return None
    return {"version": 1, "bank": qbind.bank.json(), "schema": qbind.schema,
            "markers": list(qbind.markers), "tags": qbind.tags,
            "state_format": DEFAULT_STATE_FORMAT.name, "state_keys": list(STATE_KEYS),
            "state_marker": STATE_MARKER,
            "vary": float(getattr(cfg, "vary", 0.0) or 0.0),
            "libero": getattr(cfg, "libero", None),
            "libero_frac": float(getattr(cfg, "libero_frac", 0.0) or 0.0)}


def bank_from_payload(payload: Any) -> Optional[QuestionBank]:
    """The DEFAULT bank stored by `interface_record` (None for checkpoints
    written before it existed — callers keep their `--questions` path)."""
    rec = payload.get("interface") if isinstance(payload, dict) else None
    if not isinstance(rec, dict) or not rec.get("bank"):
        return None
    try:
        return QuestionBank.from_json(rec["bank"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"  [interface] stored bank unreadable ({exc}); using --questions")
        return None


def save_checkpoint(cfg, model) -> None:
    """Save only the trainable tensors — a compact adapter payload, not the trunk."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
          if k in keep}
    payload = {"model_id": cfg.model, "trainable": cfg.trainable, "args": vars(cfg),
               "sensor_dim": SENSOR_DIM, "action_names": ACTION_NAMES,
               "qbind": (model.qbind.signature() if model.qbind is not None else None),
               "interface": interface_record(model.qbind, cfg),
               "trainable_state": sd}
    torch.save(payload, cfg.out)
    print(f"  checkpoint  : {cfg.out}  ({len(sd)} tensors, "
          f"{sum(v.numel() for v in sd.values())/1e6:.2f}M params, "
          f"{sum(v.numel() for v in sd.values())*4/2**20:.0f} MiB fp32)")


# =============================================================================
# 8 · COLLATION (tokenizer-dependent, so it lives outside the world)
# =============================================================================
def collate(episode: Episode, tok, qbind: Optional["QuestionBinding"] = None,
            speak_tokens: int = 1, schemas: Optional[Sequence[str]] = None) -> Episode:
    """
    Tokenize the batch: the prompt (instruction + SCHEMA), the state sentences,
    the answer-marker positions, and the per-step SPEECH targets.

    The prompt is the instruction followed by the QUESTIONNAIRE (no narration
    any more — speech is generated AFTER the state, see `speech_targets`): the
    schema is prompt TEXT, not parameters, which is what makes the interface
    portable — a different robot sends a different schema and the same weights
    answer it.  The state is the concatenation of one fixed-width sentence per
    control step, each closed by the answer markers (`qbind.step_suffix()`), so
    `ans_pos (B,S,Q)` is the token index of question q's marker for step s; the
    model adds the prompt length to get the absolute read-out position.

    The renderer is fixed-width on purpose: one step must always tokenize to the
    same number of tokens, otherwise the read-out positions would drift.  A
    tokenizer that breaks that invariant raises here instead of silently
    mis-aligning every answer.
    """
    pad = (getattr(tok, "pad_token_id", None)
           or getattr(tok, "eos_token_id", None) or 0)
    texts = list(episode.texts)
    if qbind is not None:
        if schemas is None:
            texts = [f"{t}\n{qbind.schema}" for t in texts]      # questions are INPUT
        else:                          # `--vary`: this row's rendering of the SAME
            texts = [f"{t}\n{s}" for t, s in zip(texts, schemas)]   # binding
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    T = max(1, max(len(e) for e in enc))
    ids = torch.full((len(enc), T), int(pad), dtype=torch.long)
    mask = torch.zeros(len(enc), T, dtype=torch.bool)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        mask[i, :len(e)] = True

    sids: Optional[torch.Tensor] = None
    apos: Optional[torch.Tensor] = None
    if episode.state_texts is not None:
        suffix = qbind.step_suffix() if qbind is not None else ""
        rows = [[tok.encode(f"{t}{suffix} {STATE_MARKER}", add_special_tokens=False)
                 for t in row] for row in episode.state_texts]
        lens = {len(e) for row in rows for e in row}
        if len(lens) != 1:
            raise ValueError(
                f"state sentences must tokenize to a constant length, got {sorted(lens)} "
                f"— `render_state` is fixed-width, so this tokenizer is splitting the "
                f"numbers differently; the read-out positions cannot be computed.")
        L = lens.pop()
        B, S = len(rows), len(rows[0])
        Q = len(qbind) if qbind is not None else 1
        sids = torch.full((B, S * L), int(pad), dtype=torch.long)
        apos = torch.zeros(B, S, Q, dtype=torch.long)
        for b, row in enumerate(rows):
            for s, e in enumerate(row):
                sids[b, s * L:(s + 1) * L] = torch.tensor(e, dtype=torch.long)
                for q in range(Q):                    # Q markers, bank order,
                    apos[b, s, q] = s * L + (L - 1 - Q + q)   # then STATE_MARKER

    sp, so = (speech_stream_targets(episode.speech, tok, speak_tokens)
              if episode.speech is not None else (None, None))
    return Episode(episode.texts, episode.sensors, episode.actions, episode.goal_dist,
                   ids, mask, episode.tasks, episode.speech, episode.heights,
                   episode.state_texts, sids, apos, sp, so, **episode._extras())


def speech_end_id(tok) -> int:
    """
    The token that ends an utterance — and, alone, means "silent".  The
    tokenizer's eos (`<|endoftext|>` for Qwen2.5; `<|im_end|>` for SmolLM2,
    which is also STATE_MARKER there — harmless, it is only ever a TARGET);
    the HashTokenizer's fixed eos id 1, which no word can hash to.
    """
    for name in ("eos_token_id", "pad_token_id"):
        v = getattr(tok, name, None)
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if v is not None:
            return int(v)
    return 0


def encode_sentence(tok, sentence: str) -> List[int]:
    """
    Sentence tokens as they are generated right after the state's closing
    STATE_MARKER: NO leading space (the marker is a special token, so the first
    word starts fresh).  Used identically by collate and the live bridge.
    """
    return list(tok.encode(sentence, add_special_tokens=False)) if sentence else []


def speech_targets(speech: List[List[str]], tok) -> torch.Tensor:
    """B × steps sentences ('' = silent) -> (B, steps, Ls) long: sentence + END,
    -100 padded.  A silent step is just [END].  (UNSCHEDULED: each row is the
    event frame's own sentence; training uses `speech_stream_targets`.)"""
    end = speech_end_id(tok)
    rows = [[encode_sentence(tok, s) + [end] for s in row] for row in speech]
    return _pad_speech_rows(rows)


def _pad_speech_rows(rows: List[List[List[int]]]) -> torch.Tensor:
    B = len(rows)
    S = max((len(r) for r in rows), default=0)
    Ls = max((len(e) for r in rows for e in r), default=1)
    out = torch.full((B, S, Ls), -100, dtype=torch.long)
    for b, row in enumerate(rows):
        for s, e in enumerate(row):
            out[b, s, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


def speech_stream_targets(speech: List[List[str]], tok, speak_tokens: int = 1
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    B × steps EVENT sentences ('' = no event) -> `stream_schedule`d targets:
    `(speech_ids (B,steps,Ls) -100 padded, speech_off (B,steps) long)` — per
    step the sentence + END of the utterance ACTIVE there and the tokens of it
    already spoken.  The schedule runs over the WHOLE episode, so a window cut
    later keeps the offsets of utterances that started in an earlier window.
    """
    end = speech_end_id(tok)
    rows, offs = [], []
    for row in speech:
        ev = [encode_sentence(tok, s) if s else None for s in row]
        r, o = stream_schedule(ev, speak_tokens, end)
        rows.append(r)
        offs.append(o)
    out = _pad_speech_rows(rows)
    off = torch.zeros(out.shape[0], out.shape[1], dtype=torch.long)
    for b, o in enumerate(offs):
        if o:
            off[b, :len(o)] = torch.tensor(o, dtype=torch.long)
    return out, off


def speak_stream(first_logits: torch.Tensor,
                 extra_fn: Callable[[List[int]], torch.Tensor],
                 spoken: List[int], K: int, end_id: int,
                 max_n: int) -> Tuple[List[int], bool, int]:
    """
    ONE control frame of streamed greedy speech.  `first_logits (V,)` is the
    LM head at the LAST position of the frame's forward, whose speech input
    was `spoken` (the utterance so far; empty -> the last state position), so
    the first new token is free.  With K=2 and a non-END first token exactly
    ONE `extra_fn(spoken + [t]) -> logits (V,)` forward yields the second.
    `max_n` is the utterance length in tokens INCLUDING END (0 = mute): once
    `max_n - 1` sentence tokens are held the END is forced and nothing more is
    decoded.  Returns (new tokens (END excluded), utterance finished, extra
    forwards ∈ {0, 1}).
    """
    if max_n <= 0:
        return [], True, 0
    cap = max_n - 1
    new: List[int] = []
    n = 0
    logits = first_logits
    for i in range(max(1, int(K))):
        if len(spoken) + len(new) >= cap:
            return new, True, n                      # --speak-max forces the END
        if i > 0:
            logits = extra_fn(list(spoken) + new)
            n += 1
        t = int(logits.argmax(-1))
        if t == end_id:
            return new, True, n
        new.append(t)
    return new, len(spoken) + len(new) >= cap, n


def speech_prefix(spoken: List[int], device: torch.device
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Batch-of-ONE speech input for the tokens spoken so far (None when empty)."""
    if not spoken:
        return None, None
    sp = torch.tensor([list(spoken)], dtype=torch.long, device=device)
    return sp, torch.ones_like(sp, dtype=torch.bool)


@torch.no_grad()
def speak_frame(model: "LoopedACTTransformer", out: "LoopOutput",
                instr_ids: torch.Tensor, instr_mask: torch.Tensor,
                state_ids: torch.Tensor, ans_pos: torch.Tensor,
                carry: Optional[torch.Tensor], spoken: List[int], K: int,
                end_id: int, max_n: int) -> Tuple[List[int], bool, int]:
    """
    `speak_stream` on a real model, batch of ONE.  `out` is the frame's forward
    (state window + `spoken` as speech input) that ALSO produced the answers;
    `carry` is the carry that forward was GIVEN, so the extra forward sees
    exactly the same prefix.  Speech sits after the state, so by causality it
    moves neither an answer nor the carry.
    """
    if max_n <= 0 or out.speech_h is None:
        return [], True, 0
    first = model.text_logits(out.speech_h[:, -1])
    if first is None:
        return [], True, 0

    def _extra(ids: List[int]) -> torch.Tensor:
        sp, sm = speech_prefix(ids, state_ids.device)
        o = model(instr_ids, instr_mask, state_ids, ans_pos, carry=carry,
                  speech_ids=sp, speech_mask=sm)
        return model.text_logits(o.speech_h[:, -1])[0].float()

    return speak_stream(first[0].float(), _extra, spoken, K, end_id, max_n)


# =============================================================================
# 9 · SELF-TEST  (architecture invariants; `--model tiny`, CPU-safe, no download)
# =============================================================================
def run_self_test(cfg, accel) -> int:
    print("=" * 78)
    print("SELF-TEST (tiny random trunk, architectural invariants)")
    print("=" * 78)
    tok = HashTokenizer(1024)
    trunk = build_tiny_trunk(hidden=64, layers=4, vocab=1024,
                             attn_impl=accel.attn_impl)
    bank = resolve_questions("auto")
    qbind = QuestionBinding(bank, tok)               # schema + option token ids
    model = LoopedACTTransformer(trunk, num_looped_layers=cfg.recurrent_layers,
                                 max_loops=cfg.loops, act_eps=1e-2, pond_tau=cfg.tau,
                                 fusion_heads=8,
                                 dropout=0.0, qbind=qbind,
                                 lora_rank=cfg.lora_rank,
                                 lora_alpha=cfg.lora_alpha,
                                 cycle_tag=cfg.cycle_tag).to(accel.device)
    configure_trainable(model, "lora" if cfg.lora_rank > 0 else "adapters")
    model.eval()
    B, S = 3, 6
    fails: List[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")
        if not cond:
            fails.append(name)

    # ── t0 · LoRA (only when --lora-rank > 0) ────────────────────────────────
    # Two failures here would be silent: a wrapper that is not identity at init
    # (it would corrupt a frozen trunk on step 0) and a base weight that quietly
    # became trainable (it would defeat the whole point of the mode).
    if cfg.lora_rank > 0:
        ltrunk = build_tiny_trunk(hidden=64, layers=2, vocab=1024,
                                  attn_impl=accel.attn_impl)
        lmodel = LoopedACTTransformer(ltrunk, num_looped_layers=1, max_loops=2,
                                      qbind=qbind, lora_rank=cfg.lora_rank,
                                      lora_alpha=cfg.lora_alpha).to(accel.device)
        lin = ltrunk.layers[0].self_attn.q_proj
        x = torch.randn(4, 64, device=accel.device)
        with torch.no_grad():
            y0, ybase = lin(x), lin.base(x)
        n_par = sum(p.numel() for n, p in lmodel.named_parameters()
                    if n.endswith((".lora_a", ".lora_b")))
        ok("LoRA: every trunk projection is wrapped",
           lmodel.n_lora == 2 * len(LORA_TARGETS),
           f"{lmodel.n_lora} projections · r={cfg.lora_rank} · "
           f"{n_par/1e3:.1f}K factors")
        ok("LoRA: identity at init (B = 0) — the frozen forward is untouched",
           torch.equal(y0, ybase), f"max|d|={float((y0-ybase).abs().max()):.1e}")
        ok("LoRA: the wrapped base weight stays frozen",
           not lin.base.weight.requires_grad and not lin.base.bias.requires_grad)
        configure_trainable(lmodel, "lora")
        names = [n for n, p in lmodel.named_parameters() if p.requires_grad]
        ok("LoRA: --trainable lora selects only the factors + the adapters",
           all((".lora_" in n or n.startswith(ADAPTER_MODULES) or
                n.startswith(("cycle_emb", "carry_tag", "cycle_gate"))) for n in names),
           f"{len(names)} tensors · "
           f"{sum(p.numel() for p in lmodel.parameters() if p.requires_grad)/1e3:.0f}K params")

    sensors = torch.randn(B, S, SENSOR_DIM, device=accel.device)
    instr_ids = torch.randint(2, 1024, (B, 5), device=accel.device)
    instr_mask = torch.ones(B, 5, dtype=torch.bool, device=accel.device)
    instr_mask[0, 3:] = False                                   # exercise padding
    raw = SensorWorld(0).episode(B, S)
    ep = collate(raw, tok, qbind).to(accel.device)   # prompt + SCHEMA + state + markers
    actions = raw.actions.to(accel.device)

    def qcat(o: LoopOutput) -> torch.Tensor:
        """All question logits of an output, (B, S, bank.total) in bank order."""
        return torch.cat([o.q_logits[s.qid] for s in bank.specs], dim=-1)

    # t1 — the zero gate makes the prompt injection an exact identity at init
    h_slots = torch.randn(B, S * 3, model.hidden, device=accel.device)
    ctx = torch.randn(B, 7, model.hidden, device=accel.device)      # prompt latents
    fused = model.fusion(h_slots, context=ctx)
    ok("fusion identity at init (gate=0 ⇒ the trunk is untouched)",
       torch.equal(fused, h_slots) and
       float(model.fusion.gate.detach().abs().max()) == 0.0,
       f"max|Δ|={float((fused-h_slots).detach().abs().max()):.3e}")

    # t2/t3 — ACT convexity + ponder bounds + finiteness
    out = model(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None)
    ok("ACT weights form a convex combination (Σα=1)",
       out.stats["mass_err"] < 1e-4, f"max|Σα-1|={out.stats['mass_err']:.2e}")
    ok("ponder ∈ [1, max_loops]",
       bool((out.ponder >= 1.0 - 1e-4).all() and (out.ponder <= cfg.loops + 1e-4).all()),
       f"min={float(out.ponder.detach().min()):.3f} "
       f"max={float(out.ponder.detach().max()):.3f}")
    qo = qcat(out)
    ok("question logits finite, shape (B,S,Σ options) per qid (B,S,K_q)",
       bool(torch.isfinite(qo).all()) and tuple(qo.shape) == (B, S, bank.total)
       and all(tuple(out.q_logits[s.qid].shape) == (B, S, s.n_out) for s in bank.specs),
       f"{tuple(qo.shape)} · " + " ".join(
           f"{s.qid}:{tuple(out.q_logits[s.qid].shape)[-1]}" for s in bank.specs))
    ja = json_action(out.q_logits)
    ok("json_action(q_logits) -> (B,S) long action indices",
       tuple(ja.shape) == (B, S) and ja.dtype == torch.long and
       bool(((ja >= 0) & (ja < NUM_ACTIONS)).all()), str(tuple(ja.shape)))

    # t4 — causality: perturbing the OBSERVATION of step s+1 may not move step ≤ s
    s_probe = 2
    sids_b = ep.state_ids.clone()
    Lb = ep.step_len
    sids_b[:, (s_probe + 1) * Lb:(s_probe + 2) * Lb] = ep.state_ids[:, :Lb]  # other state
    out_b = model(instr_ids, instr_mask, sids_b, ep.ans_pos, carry=None)
    qb = qcat(out_b)
    d = (qo[:, :s_probe + 1] - qb[:, :s_probe + 1]).abs().max().detach()
    d_future = (qo[:, s_probe + 1:] - qb[:, s_probe + 1:]).abs().max().detach()
    ok("no future leak (state step s+1 cannot move answers ≤ s)",
       float(d) < 1e-5 and float(d_future) > 0, f"Δpast={float(d):.2e} Δfuture={float(d_future):.3e}")

    # t4s — speech is appended AFTER the state: under the causal mask its
    #       tokens cannot move a single answer logit (nor the task logits)
    sp_ids = torch.randint(2, 1024, (B, 4), device=accel.device)
    sp_msk = torch.ones(B, 4, dtype=torch.bool, device=accel.device)
    sp_msk[1, 2:] = False
    out_s = model(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None,
                  speech_ids=sp_ids, speech_mask=sp_msk)
    d_sp = float((qcat(out_s) - qo).abs().max().detach())
    d_tk = float((out_s.task_logits - out.task_logits).abs().max().detach())
    ok("speech tokens do not change any answer logit (causality)",
       d_sp < 1e-5 and d_tk < 1e-5 and out_s.speech_h is not None and
       tuple(out_s.speech_h.shape) == (B, 1 + 4, model.hidden),
       f"Δanswers={d_sp:.2e} Δtask={d_tk:.2e} speech_h="
       f"{None if out_s.speech_h is None else tuple(out_s.speech_h.shape)}")

    # t4b — the mask is built in the attention dtype at finfo.min: in fp16 that
    #       must not overflow to -inf (which would NaN a fully-masked row)
    valid_all = torch.ones(B, 5, dtype=torch.bool, device=accel.device)
    m16 = LoopedACTTransformer._additive_mask(valid_all, torch.float16)
    ok("fp16 additive mask stays finite (finfo(fp16).min, not -inf)",
       bool(torch.isfinite(m16).all()) and
       float(m16.min()) == float(torch.finfo(torch.float16).min),
       f"min={float(m16.min()):.0f} (fp16 min {torch.finfo(torch.float16).min:.0f})")

    # t5 — forced halting: p≈1 ⇒ exactly one cycle, ponder ≈ 1, loop breaks early
    with torch.no_grad():
        model.halt_head.proj.bias.fill_(12.0)
    out_h = model(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None)
    ok("forced halt ⇒ 1 cycle, ponder ≈ 1, single loop iteration",
       abs(out_h.stats["mean_cycles"] - 1.0) < 1e-3 and
       abs(out_h.stats["mean_ponder"] - 1.0) < 1e-3 and
       len(out_h.stats["cycle_usage"]) == 1 and
       float(out_h.stats["cycle_usage"][0]) == 1.0,
       f"cycles={out_h.stats['mean_cycles']:.4f} "
       f"ponder={out_h.stats['mean_ponder']:.4f} "
       f"iterations={len(out_h.stats['cycle_usage'])}")

    # t5b — UNBOUNDED ACT.  The loop is halting-driven, so the cycle count is
    #       the model's own decision; what makes `--loops` *structural* is the
    #       learned cycle table (`cycle_emb[cycle]` cannot be read past its
    #       last row).  The sinusoidal tag is a function of the INDEX, so
    #       raising the ceiling leaves nothing capped — and the ponder penalty
    #       only *rewards* finishing early, it never forces a count.
    ubt = build_tiny_trunk(hidden=64, layers=4, vocab=1024,
                           attn_impl=accel.attn_impl)
    ub = LoopedACTTransformer(ubt, num_looped_layers=2, max_loops=64,
                              act_eps=1e-2, pond_tau=cfg.tau,
                              fusion_heads=8,
                              dropout=0.0, qbind=qbind,
                              cycle_tag="sinusoidal").to(accel.device)
    configure_trainable(ub, "adapters")
    # NB: the tag is a sinusoid, so |tag| is BOUNDED by 1, it does not reach
    # 1 at every index — the properties that matter are "defined and finite
    # for any index" and "two cycles are distinguishable".
    t500 = ub._cycle_tag(500, ub.cycle_gate)
    t501 = ub._cycle_tag(501, ub.cycle_gate)
    ok("unbounded: the sinusoidal tag is defined for any cycle index",
       ub.cycle_emb is None and ub.cycle_gate.requires_grad
       and tuple(ub.cycle_freq.shape) == ((ub.hidden + 1) // 2,)
       and bool(torch.isfinite(t500).all())
       and float(t500.abs().max()) <= 1.0 and not torch.equal(t500, t501),
       f"cycle_emb=None · gate={float(ub.cycle_gate):+.1f} · "
       f"freq={tuple(ub.cycle_freq.shape)} · "
       f"|tag(500)|max={float(t500.abs().max()):.3f} · tag(500)≠tag(501)")
    with torch.no_grad():                      # a head that never halts
        ub.halt_head.proj.bias.fill_(-30.0)
    o_ub = ub(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None)
    ok("unbounded: 64 cycles when the head never halts, Σα still 1",
       abs(o_ub.stats["mass_err"]) < 1e-4 and
       abs(o_ub.stats["mean_cycles"] - 64.0) < 1e-3 and
       abs(o_ub.stats["mean_ponder"] - 64.0) < 1e-3,
       f"cycles={o_ub.stats['mean_cycles']:.2f} · "
       f"Σα-1={o_ub.stats['mass_err']:.1e} · "
       f"ponder={o_ub.stats['mean_ponder']:.2f}")
    ub.zero_grad(set_to_none=True)
    o_ub = ub(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None)
    qcat(o_ub).float().pow(2).mean().backward()        # through the ANSWERS
    g_ub = (float(ub.cycle_gate.grad.abs().sum())
            if ub.cycle_gate.grad is not None else 0.0)
    ok("unbounded: the loop takes gradient from the answers (cycle tag not dead)",
       g_ub > 0.0, f"|d/d cycle_gate|={g_ub:.3e}")

    # t6 — latent carry: continuity across windows, no NaN, shape contract
    with torch.no_grad():
        model.halt_head.reset_bias(min(cfg.loops, 8))
    out_a = model(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None)
    out_c = model(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=out_a.carry)
    qa, qc = qcat(out_a), qcat(out_c)
    d_carry = (qa - qc).abs().max().detach()
    ok("carry is (B,H), finite, and changes the answers",
       tuple(out_a.carry.shape) == (B, model.hidden) and
       bool(torch.isfinite(qc).all()) and float(d_carry) > 0,
       f"|Δanswer logits|={float(d_carry):.3e}")

    # t7 — one real optimization step: adapters learn, frozen trunk is untouched
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(set_to_none=True)
    sp_t = speech_targets([[s for s in row] for row in raw.speech], tok).to(accel.device)
    # full-sentence teacher forcing (off=0, K=None) keeps the t9s/CE checks
    # exercising every speech position; the streamed masking is checked in t9st
    sp_in_t, sp_m_t, sp_tg_t = speech_io_from_targets(sp_t[:, -1])
    out_t = model(instr_ids, instr_mask, ep.state_ids, ep.ans_pos, carry=None,
                  speech_ids=sp_in_t, speech_mask=sp_m_t)
    q_lab = derive_question_labels(model.bank, sensors, actions,
                                   extra={"height": raw.heights.to(accel.device)})
    sce_t = model.speech_ce(out_t.speech_h, sp_tg_t)
    loss, parts = joint_loss(out_t, cfg.tau, q_labels=q_lab,
                             speech_ce=sce_t, text_weight=0.5)
    watch = {n: p.detach().clone() for n, p in model.named_parameters()
             if p.requires_grad and (n.startswith(("lm_head", "task_proj"))
                                     or n.endswith(".lora_b"))}
    loss.backward()
    g_proj = model.task_proj.weight.grad
    g_gate = model.fusion.gate.grad
    opt.step()
    moved = sorted({n.split(".")[0] if not n.endswith(".lora_b") else "lora"
                    for n, p in model.named_parameters()
                    if n in watch and not torch.equal(p.detach(), watch[n])})
    need = {"task_proj"} | ({"lm_head"} if model.lm_head is not None else set()) \
        | ({"lora"} if cfg.lora_rank > 0 else set())
    ok("optimizer step: finite loss + adapter grads + frozen trunk untouched",
       bool(torch.isfinite(loss.detach())) and g_proj is not None
       and float(g_proj.detach().abs().sum()) > 0
       and model.trunk.embed_tokens.weight.grad is None,
       f"loss={float(loss.detach()):.4f} q={float(parts['questions'].detach()):.4f} "
       f"speech={float(parts['speech'].detach()):.4f} "
       f"ponder={float(parts['ponder'].detach()):.3f}")
    ok("optimizer step changes task_proj (+ lm_head / LoRA factors when present)",
       need.issubset(set(moved)), f"moved={moved} need={sorted(need)}")
    # The zero gate must still RECEIVE gradient, otherwise zeroing anything else
    # in the fusion path (e.g. out_proj) silently kills cross-attention forever.
    ok("fusion gate is learnable at init (no zero-init deadlock)",
       g_gate is not None and float(g_gate.detach().abs().sum()) > 0,
       f"|dL/dgate|={0.0 if g_gate is None else float(g_gate.detach().abs().sum()):.4e} "
       f"· tanh(gate)={float(torch.tanh(model.fusion.gate.detach())):+.4f}")
    # t7b — the schema is DATA, not parameters: send a DIFFERENT questionnaire
    #       (other questions, other option counts) and the same weights answer it,
    #       because every answer is the model's own next-token distribution
    #       restricted to that question's option tokens.  This is the Jev
    #       property — the interface is portable because nothing is baked in.
    alt = QuestionBank.from_json(json.dumps({"questions": {
        "axis_x": {"type": "choice", "instructions": "Which way along x?",
                   "criteria": {"negative": "toward −x", "positive": "toward +x"}},
        "tilt": {"type": "score", "instructions": "How tilted is the tool?",
                 "criteria": ["flat", "slight", "steep", "inverted"]}}}))
    qbind_alt = QuestionBinding(alt, tok)
    ep_alt = collate(raw, tok, qbind_alt).to(accel.device)
    with torch.no_grad():
        out_alt = model(instr_ids, instr_mask, ep_alt.state_ids, ep_alt.ans_pos,
                        qbind=qbind_alt)
    ok("schema is DATA: a different questionnaire answers on the same weights",
       set(out_alt.q_logits) == {"axis_x", "tilt"} and
       tuple(out_alt.q_logits["axis_x"].shape) == (B, S, 2) and
       tuple(out_alt.q_logits["tilt"].shape) == (B, S, 4) and
       bool(torch.isfinite(out_alt.q_logits["tilt"]).all()),
       f"2 questions · 6 option logits · markers {' '.join(qbind_alt.markers)} · "
       f"schema {len(qbind_alt.schema)} chars vs {len(qbind.schema)}")

    # The verb is carried by the PROMPT alone.  Same observation, two different
    # instructions: they can only be separated through the prompt (the fusion
    # cross-attention and the pooled `task_proj` shortcut), so a falling CE here
    # is the sentence reaching the read-out — the exact signal `--task-weight`
    # adds to training.  (The expert's steering gives the prompt almost
    # nothing: all four verbs steer identically until contact.)
    ids_a = tok.encode("move to the red block", add_special_tokens=False)
    ids_b = tok.encode("hurl the red block", add_special_tokens=False)
    ti2 = max(len(ids_a), len(ids_b))
    ii2 = torch.zeros(2, ti2, dtype=torch.long, device=accel.device)
    mm2 = torch.zeros(2, ti2, dtype=torch.bool, device=accel.device)
    for r, seq in enumerate((ids_a, ids_b)):
        ii2[r, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=accel.device)
        mm2[r, :len(seq)] = True
    st2 = ep.state_ids[:1].expand(2, -1)              # identical observations
    ap2 = ep.ans_pos[:1].expand(2, -1, -1)            # ...and marker positions
    t_ids = torch.tensor([0, 2], dtype=torch.long, device=accel.device)  # reach, throw
    opt.zero_grad(set_to_none=True)
    out_k = model(ii2, mm2, st2, ap2, carry=None)
    l0, p0 = joint_loss(out_k, cfg.tau, task_ids=t_ids, task_weight=1.0)
    l0.backward()
    g_taskhead = model.task_head[1].weight.grad
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        o = model(ii2, mm2, st2, ap2, carry=None)
        lk, pk = joint_loss(o, cfg.tau, task_ids=t_ids, task_weight=1.0)
        lk.backward()
        opt.step()
    with torch.no_grad():
        guess = model(ii2, mm2, st2, ap2, carry=None).task_logits[:, -1].argmax(-1)
    ok("task head: two prompts on ONE observation are told apart "
       "(the instruction reaches the read-out)",
       g_taskhead is not None and float(g_taskhead.detach().abs().sum()) > 0
       and guess.tolist() == [0, 2],
       f"task CE {float(p0['task'].detach()):.4f} → "
       f"{float(pk['task'].detach()):.4f} · guess={guess.tolist()} want=[0, 2] · "
       f"|dL/dW_task_head|="
       f"{0.0 if g_taskhead is None else float(g_taskhead.detach().abs().sum()):.3e} · "
       f"shape={tuple(out_k.task_logits.shape)}")

    # t8 — data path: collate right-pads + masks exactly the real tokens, and the
    #      OOM micro-batch split is a partition (no sample dropped or duplicated).
    #      The prompt is `instruction` + schema ONLY: speech is no longer in the
    #      prompt, it is generated after the state (targets in `speech_ids`).
    raw_long = SensorWorld(3).episode(4, 5)
    col = collate(raw_long, tok, qbind)
    want = [tok.encode(f"{t}\n{qbind.schema}", add_special_tokens=False)
            for t in raw_long.texts]
    ids_ok = all(col.instr_ids[i, :len(w)].tolist() == w for i, w in enumerate(want))
    mask_ok = all(int(col.instr_mask[i].sum()) == len(w) for i, w in enumerate(want))
    ok("collate: ids match `instruction` + schema (no narration), pad keys masked out",
       ids_ok and mask_ok and col.instr_ids.shape[1] == max(len(w) for w in want),
       f"Ti={col.instr_ids.shape[1]} masks={[int(m.sum()) for m in col.instr_mask]} "
       f"· schema={len(qbind.schema)} chars")
    end_id = speech_end_id(tok)
    sp_ok = (col.speech_ids is not None and col.speech_off is not None
             and tuple(col.speech_ids.shape[:2]) == (4, 5)
             and tuple(col.speech_off.shape) == (4, 5))
    if sp_ok:
        for b in range(4):
            ev_b = [encode_sentence(tok, s) if s else None for s in raw_long.speech[b]]
            want_ids, want_off = stream_schedule(ev_b, 1, end_id)
            for s in range(5):
                row = [int(v) for v in col.speech_ids[b, s] if int(v) >= 0]
                sp_ok &= row == want_ids[s] and int(col.speech_off[b, s]) == want_off[s]
    w2 = col.window(2, 3) if sp_ok else None
    h2 = col.half(1) if sp_ok else None
    sp_ok = sp_ok and torch.equal(w2.speech_off, col.speech_off[:, 2:5]) \
        and torch.equal(h2.speech_off, col.speech_off[2:]) \
        and torch.equal(col.to(torch.device("cpu")).speech_off, col.speech_off)
    ok("collate: speech target = the ACTIVE utterance's sentence + END with its "
       "stream offset (silent = [END], off 0); window/half/to carry speech_off",
       sp_ok, f"speech_ids={None if col.speech_ids is None else tuple(col.speech_ids.shape)} "
       f"· END={end_id}")

    # t8b — the task family is a real behavioural difference, not a relabelling:
    #       same geometry, and the action sequence after arrival depends on the
    #       verb (`grasp` holds, `throw` carries off, `reach` brakes, `push`
    #       never stops).  The instruction is the ONLY channel that says which.
    big = SensorWorld(11).episode(16, 32)
    hit = {t: [] for t in range(len(TASKS))}
    for b in range(big.actions.shape[0]):
        seq = big.actions[b].tolist()
        hit[int(big.tasks[b])].append(seq)
    def after(seq, first, then):
        return any(first in seq[:i] and seq[i] == then for i in range(len(seq)))
    # `grasp` and `throw` may only follow a close with hold / carry-off
    # respectively, `reach` may only brake, and `push` may do neither — those
    # zero-counts are structural.  The positive counts need at least one sample
    # to actually arrive inside `reach`, which is geometry-dependent.
    g_hold = sum(after(s, 4, 5) for s in hit[1])
    g_dep = sum(after(s, 4, 3) for s in hit[1])
    t_dep = sum(after(s, 4, 3) for s in hit[2])
    t_hold = sum(after(s, 4, 5) for s in hit[2])
    r_hold = sum(after(s, 4, 5) for s in hit[0])
    p_any = sum(after(s, 4, 5) or after(s, 4, 3) for s in hit[3])
    # speech events: every sentence is a template for the TARGET colour, and the
    # event matches the verb (grasp/throw: grasped|released, reach: reached,
    # push: pushing); every other frame is silent
    allowed = {0: ("reached",), 1: ("grasped", "released"),
               2: ("grasped", "released"), 3: ("pushing",)}
    n_speak = 0
    speech_ok = True
    for b in range(len(big.speech)):
        k = int(big.tasks[b])
        for s in big.speech[b]:
            if not s:
                continue
            n_speak += 1
            speech_ok &= any(s == speech_sentence(ev, c) for ev in allowed[k]
                             for c in COLORS if c in big.texts[b])
    frames = sum(len(r) for r in big.speech)
    ok("task family: the verb changes the expert's action sequence",
       all(hit[t] for t in range(len(TASKS)))
       and g_hold > 0 and t_dep > 0 and g_dep == 0 and t_hold == 0
       and r_hold > 0 and p_any == 0 and len(set(big.texts)) >= 2,
       f"tasks {[len(hit[t]) for t in range(len(TASKS))]}/16 · grasp hold={g_hold} "
       f"depart={g_dep} · throw depart={t_dep} hold={t_hold} · reach hold={r_hold} "
       f"· push acted={p_any}")
    ok("speech events: template sentences for the target colour, verb-consistent, "
       "sparse", speech_ok and 0 < n_speak < frames,
       f"speaking {n_speak}/{frames} frames · e.g. "
       f"{[s for r in big.speech for s in r if s][:3]}")

    h0, h1 = raw_long.half(0), raw_long.half(1)
    rejoin = torch.cat([h0.sensors, h1.sensors], dim=0)
    act_ok = torch.equal(torch.cat([h0.actions, h1.actions], 0), raw_long.actions)
    ok("micro-batch split is a partition of the batch",
       torch.equal(rejoin, raw_long.sensors) and act_ok and
       h0.texts + h1.texts == raw_long.texts,
       f"{tuple(h0.sensors.shape)} + {tuple(h1.sensors.shape)}")

    # t9 — the JSON decision interface: typed questions read off the looped
    #      latent, rendered as Jev-shaped JSON, trained by the same step
    bank = model.bank
    bank2 = QuestionBank.from_json(bank.json())
    ok("question bank round-trips through JSON text",
       [s.qid for s in bank2.specs] == [s.qid for s in bank.specs] and
       [s.labels for s in bank2.specs] == [s.labels for s in bank.specs] and
       bank2.total == bank.total,
       f"{len(bank)} questions · {bank.total} logits · "
       f"{' '.join(f'{s.qid}:{s.n_out}' for s in bank.specs)}")
    ok("read-out: one distribution per question, over its OWN option tokens",
       out_t.q_logits is not None and
       set(out_t.q_logits) == {s.qid for s in bank.specs} and
       all(tuple(out_t.q_logits[s.qid].shape) == (B, S, s.n_out)
           for s in bank.specs),
       f"{len(bank)} question(s) · {bank.total} option logits · "
       f"{' '.join(f'{s.qid}:{s.n_out}' for s in bank.specs)}")
    # the read-out must land ON the schema markers, in bank order, and they must
    # be adjacent — that is what makes `ans_pos` computable for any schema
    step0 = [int(ep.state_ids[0, ep.ans_pos[0, 0, q]]) for q in range(len(qbind))]
    ok("the read-out positions are the schema markers, in bank order",
       step0 == list(qbind.marker_ids) and
       bool((ep.ans_pos[:, 0, 1:] - ep.ans_pos[:, 0, :-1] == 1).all()),
       f"markers {step0} == {qbind.marker_ids} · schema {len(qbind.schema)} chars "
       f"· step block {ep.step_len} tokens")
    # the window slice must not move the anchors: a marker is only meaningful
    # relative to the state region it was computed in
    epw = ep.window(2, min(3, S - 2))
    ok("window(): the markers stay on their own tokens after a slice",
       torch.equal(epw.state_ids[0, epw.ans_pos[0, :, -1]],
                   ep.state_ids[0, ep.ans_pos[0, 2:2 + epw.ans_pos.shape[1], -1]]) and
       int(epw.ans_pos.max()) < epw.state_ids.shape[1] and int(epw.ans_pos.min()) >= 0,
       f"start=2 · marker {int(epw.ans_pos.min())}..{int(epw.ans_pos.max())} of "
       f"{epw.state_ids.shape[1]} tokens · step block {epw.step_len}")
    doc = model.answers_json(out_t)
    sums_ok = all(abs(sum(e["probabilities"].values()) - 1.0) < 1e-3
                  for e in doc.values() if isinstance(e, dict) and "probabilities" in e)
    names_ok = all(e["choice"] in bank[q].labels for q, e in doc.items()
                   if q in bank and isinstance(e, dict) and "choice" in e)
    ok("answers_json: probabilities sum to 1, option NAMES (not labels), "
       "json.dumps() round-trips",
       sums_ok and names_ok and isinstance(json.loads(json.dumps(doc)), dict) and
       doc["action"]["choice"] in ACTION_NAMES,
       f"{len(doc)} field(s) · {json.dumps(doc)[:64]}… · "
       f"answered without generating a single token")
    doc_s = model.answers_json(out_t, say="grasped the red block")
    ok("JSON carries a top-level `say` (null when silent)",
       "say" in doc and doc["say"] is None and
       json.loads(json.dumps(doc_s))["say"] == "grasped the red block"
       and '"say": null' in json.dumps(doc),
       f"silent={json.dumps(doc['say'])} · spoken={json.dumps(doc_s['say'])}")
    labs = derive_question_labels(bank, sensors, actions,
                                  extra={"height": raw.heights.to(accel.device)})
    ok("question labels derive from the expert trajectory, all in range",
       all(0 <= int(v.min()) and int(v.max()) < bank[q].n_out
           for q, v in labs.items()),
       " ".join(f"{q}:{tuple(v.shape)}" for q, v in labs.items()))

    # t9j — Jev labels: A B C … / 0 1 2 …, one token each, distinct per question,
    #       and a tokenizer that breaks that is refused at bind time
    lab_ok = all(len(qbind.option_ids[s.qid]) == s.n_out and
                 len(set(qbind.option_ids[s.qid])) == s.n_out and
                 all(tok.encode(t, add_special_tokens=False) == [i]
                     for t, i in zip(qbind.tags[s.qid], qbind.option_ids[s.qid]))
                 for s in bank.specs)
    tag_ok = (qbind.tags["axis_x"] == ["A", "B", "C"] and
              "options: A=negative B=stay C=positive" in qbind.schema and
              qbind.tags["speed"] == ["0", "1", "2"])

    class _SplitTok(HashTokenizer):                   # "A" -> two tokens
        def encode(self, text, **kw):
            ids = super().encode(text, **kw)
            return ids + ids if text in ("A", "0") else ids

    class _SameTok(HashTokenizer):                    # every label -> one id
        def encode(self, text, **kw):
            return [7] if (len(text) == 1 and text.isalnum() and text.isupper()) \
                else super().encode(text, **kw)
    refused = []
    for bad in (_SplitTok(1024), _SameTok(1024)):
        try:
            QuestionBinding(bank, bad)
            refused.append(False)
        except ValueError:
            refused.append(True)
    ok("Jev labels are single, distinct tokens (and bad tokenizers are refused)",
       lab_ok and tag_ok and all(refused),
       f"axis_x={qbind.tags['axis_x']}->{qbind.option_ids['axis_x']} · "
       f"speed={qbind.tags['speed']} · refused(split, same)={refused}")

    # t9a — json_action is the exact inverse of LABEL_SOURCES on all 6 actions,
    #       whether the jaw answer comes from `gripper`, `intent`, or both, and
    #       from labels or from (one-hot) logits
    a6 = torch.arange(NUM_ACTIONS).view(1, NUM_ACTIONS)
    s6 = torch.zeros(1, NUM_ACTIONS, SENSOR_DIM)
    fwd = {q: LABEL_SOURCES[q](a6, s6, {}) for q in JSON_ACTION_QIDS}
    inv_ok = True
    for keep in (JSON_ACTION_QIDS, ("axis_x", "axis_y", "gripper"),
                 ("axis_x", "axis_y", "intent")):
        sub = {q: fwd[q] for q in keep}
        inv_ok &= json_action_from_labels(sub).tolist() == a6.tolist()
        lg = {q: F.one_hot(v, 3).float() * 5.0 for q, v in sub.items()}
        inv_ok &= json_action(lg).tolist() == a6.tolist()
    inv_ok &= all(json_action_one({q: int(fwd[q][0, a]) for q in JSON_ACTION_QIDS})
                  == a for a in range(NUM_ACTIONS))
    ok("json_action inverts LABEL_SOURCES on all 6 actions",
       inv_ok, " ".join(f"{ACTION_NAMES[a]}->{ACTION_NAMES[json_action_one({q: int(fwd[q][0, a]) for q in JSON_ACTION_QIDS})]}"
                        for a in range(NUM_ACTIONS)))

    # t9s — speech read-out: the first token comes from the SAME forward as the
    #       answers; an END there is a silent frame at zero extra forwards
    sl = model.text_logits(out_t.speech_h)
    ok("speech path: LM logits at the last state position + every speech input",
       sl is not None and tuple(sl.shape) == (B, sp_in_t.shape[1] + 1, 1024) and
       bool(torch.isfinite(sl).all()),
       f"{None if sl is None else tuple(sl.shape)} · "
       f"{'tied' if model.lm_weight is not None else 'untied'}")
    # t9st — STREAMED speech.  (a) the schedule, exact per frame
    E = 1
    ids_k1, off_k1 = stream_schedule([None, [10, 11, 12], None, None, None, None], 1, E)
    ids_k2, off_k2 = stream_schedule([None, [10, 11, 12], None, None], 2, E)
    u = [10, 11, 12, E]
    sched_ok = (ids_k1 == [[E], u, u, u, u, [E]] and off_k1 == [0, 0, 1, 2, 3, 0]
                and ids_k2 == [[E], u, u, [E]] and off_k2 == [0, 0, 2, 0])
    # queue: [20] arrives while [10,11] streams -> starts after its END;
    # [30] arrives while the queue is full -> dropped; cut-off at the end
    ids_q, off_q = stream_schedule([[10, 11], [20], [30], None, None, None], 1, E)
    q_ok = (ids_q == [[10, 11, E]] * 3 + [[20, E]] * 2 + [[E]]
            and off_q == [0, 1, 2, 0, 1, 0])
    ids_c, off_c = stream_schedule([None, None, [10, 11, 12]], 1, E)
    cut_ok = ids_c == [[E], [E], u] and off_c == [0, 0, 0]
    ok("stream_schedule: K=1 / K=2 offsets, queued utterance after END, "
       "third event dropped, cut-off at the episode end",
       sched_ok and q_ok and cut_ok,
       f"K1 off={off_k1} · K2 off={off_k2} · queue off={off_q} "
       f"ids={[len(r) for r in ids_q]} · cut={off_c}")

    # (b) speech_io supervises EXACTLY tgt[off:off+K] (clipped at END); the
    #     prefix is input only; a silent row predicts END at the last state pos
    tg3 = torch.tensor([[10, 11, 12, E], [E, -100, -100, -100], [10, 11, 12, E]])
    si, sm, st = speech_io_from_targets(tg3, torch.tensor([1, 0, 3]), 2)
    io_ok = (st.tolist() == [[-100, 11, 12, -100], [E, -100, -100, -100],
                             [-100, -100, -100, E]]
             and sm.tolist() == [[True, True, False], [False, False, False],
                                 [True, True, True]]
             and si[0, :2].tolist() == [10, 11] and si[2].tolist() == [10, 11, 12])
    si1, sm1, st1 = speech_io_from_targets(tg3[:2], torch.tensor([0, 0]), 1)
    io_ok &= (st1.tolist() == [[10], [E]] and tuple(si1.shape) == (2, 0))
    ok("speech_io: loss only on tgt[off:off+K] (clipped at END), input = "
       "tgt[:off+K-1], per-row offsets padded with mask False / -100",
       io_ok, f"tgt={st.tolist()} · mask={sm.int().tolist()}")

    # (c) speak_stream bookkeeping: silent = 0 extra forwards; K=2 ≤ 1
    calls = [0]

    def _never(ids):
        calls[0] += 1
        return torch.zeros(1024)
    oh = (lambda t: F.one_hot(torch.tensor(t), 1024).float())
    s_sil = speak_stream(oh(end_id), _never, [], 2, end_id, 8)
    s_two = speak_stream(oh(4), lambda ids: oh(5), [], 2, end_id, 8)
    s_end2 = speak_stream(oh(4), lambda ids: oh(end_id), [7], 2, end_id, 8)
    s_cap = speak_stream(oh(4), _never, [7], 2, end_id, 3)
    s_mute = speak_stream(oh(4), _never, [], 1, end_id, 0)
    ok("speak_stream: END first = silent at 0 extra forwards; K=2 costs ≤1; "
       "--speak-max (incl. END) forces the end; 0 mutes",
       s_sil == ([], True, 0) and calls[0] == 0 and s_two == ([4, 5], False, 1)
       and s_end2 == ([4], True, 1) and s_cap == ([4], True, 0)
       and s_mute == ([], True, 0),
       f"silent={s_sil} · two={s_two} · end={s_end2} · cap={s_cap} · mute={s_mute}")

    # (d) the real tiny model: streamed greedy decode (1 forward per frame,
    #     +≤1 for K=2) reproduces the teacher-forced arg-max sequence, K=1 and
    #     K=2 say the same thing, and the speech prefix moves no answer
    model.eval()
    i1, m1, s1, a1 = instr_ids[:1], instr_mask[:1], ep.state_ids[:1], ep.ans_pos[:1]
    dev1 = s1.device
    MAXN = 6

    def _stream(K):
        spoken: List[int] = []
        fwd: List[int] = []
        fin = False
        for _f in range(MAXN + 1):
            sp, spm = speech_prefix(spoken, dev1)
            o = model(i1, m1, s1, a1, None, speech_ids=sp, speech_mask=spm)
            new, fin, n = speak_frame(model, o, i1, m1, s1, a1, None, spoken, K,
                                      end_id, MAXN)
            spoken += new
            fwd.append(n)
            if fin:
                break
        return spoken, fwd, fin
    with torch.no_grad():
        dec1, fwd1, fin1 = _stream(1)
        dec2, fwd2, fin2 = _stream(2)
        o0 = model(i1, m1, s1, a1, None)
        sp, spm = speech_prefix(dec1, dev1)
        otf = model(i1, m1, s1, a1, None, speech_ids=sp, speech_mask=spm)
        tf = model.text_logits(otf.speech_h)[0].argmax(-1).tolist()
        d_ans = max([float((o0.q_logits[q] - otf.q_logits[q]).abs().max())
                     for q in (o0.q_logits or {})] + [0.0])
        d_car = (0.0 if o0.carry is None
                 else float((o0.carry - otf.carry).abs().max()))
    model.train()
    tf_ok = tf[:len(dec1)] == dec1 and (len(dec1) >= MAXN - 1 or tf[len(dec1)] == end_id)
    ok("streamed decode == teacher-forced arg-max; K=1 costs 0 and K=2 ≤1 extra "
       "forward per frame; K=1 and K=2 agree",
       fin1 and fin2 and tf_ok and dec1 == dec2 and all(n == 0 for n in fwd1)
       and all(n <= 1 for n in fwd2) and len(dec1) <= MAXN - 1,
       f"decoded={dec1} · tf={tf[:len(dec1) + 1]} · fwd K1={fwd1} K2={fwd2}")
    ok("answers and carry are unchanged by the speech prefix (causal mask)",
       d_ans < 1e-5 and d_car < 1e-5, f"|Δanswers|={d_ans:.2e} · |Δcarry|={d_car:.2e}")

    # the chunked speech CE has to be the same number as the monolithic one: the
    # head is position-wise and the loss is a mean over valid targets, so
    # chunking is an implementation detail — but a boundary bug would train on a
    # different objective than the one every other check measures
    if sl is not None:
        sce1 = model.speech_ce(out_t.speech_h, sp_tg_t, chunk=1)
        sce2 = model.speech_ce(out_t.speech_h, sp_tg_t, chunk=1 << 20)
        _t = sp_tg_t[:, :sl.shape[1]].reshape(-1)
        _ce = F.cross_entropy(sl.reshape(-1, sl.shape[-1]).float(),
                              _t.clamp(min=0), reduction="none")
        _m = (_t >= 0).float()
        _ref = (_ce * _m).sum() / _m.sum().clamp_min(1.0)
        ok("speech CE: the chunked head equals the monolithic head",
           sce1 is not None and sce2 is not None
           and abs(float(sce1) - float(_ref)) < 1e-4
           and abs(float(sce1) - float(sce2)) < 1e-5,
           f"chunk=1 {float(sce1):.6f} · single {float(sce2):.6f} · "
           f"reference {float(_ref):.6f}")

    # ── t-vary · --vary / --libero / eval speech probes / checkpoint interface ──
    # Each section runs in its own try: these paths are exercised nowhere else in
    # the self-test, and one crash must fail ONE named check, not hide the rest
    # of the report.
    import tempfile
    import traceback

    def _crashed(nm_: str, exc_: BaseException) -> None:
        traceback.print_exc()
        ok(nm_, False, f"raised {type(exc_).__name__}: {exc_}")

    W_v = 4
    dev_v = accel.device
    cpu_v = torch.device("cpu")
    sm_id = tok.encode(STATE_MARKER, add_special_tokens=False)[-1]

    def _markers_ok(e_: Episode) -> bool:
        """Every answer slot holds the RENDERED binding's marker (in its order)
        and STATE_MARKER closes the step right after the last one."""
        qb_ = e_.qbind if e_.qbind is not None else qbind
        Bm, Sm, Qm = e_.ans_pos.shape
        if Qm != len(qb_):
            return False
        got = e_.state_ids.gather(1, e_.ans_pos.reshape(Bm, -1))
        want_m = torch.tensor(qb_.marker_ids, dtype=torch.long,
                              device=got.device).repeat(Sm).expand(Bm, -1)
        close = e_.state_ids.gather(1, e_.ans_pos[..., -1] + 1)
        return bool((got == want_m).all()) and bool((close == sm_id).all())

    # A · varied renderings keep every marker on its token
    nm = "vary: markers sit on the rendered binding's marker tokens"
    try:
        vs = InterfaceSampler(qbind, tok, prob=1.0, seed=5)
        orders: set = set()
        m_ok = core_ok = True
        for i in range(8):
            e = vs.apply(SensorWorld(50 + i).episode(2, 2 * W_v), tok)
            qb_e = e.qbind if e.qbind is not None else qbind
            orders.add(tuple(s.qid for s in qb_e.bank.specs))
            core_ok &= all(q in qb_e.bank for q in JSON_ACTION_QIDS)
            m_ok &= _markers_ok(e) and _markers_ok(e.window(W_v, W_v))
        ok(nm, m_ok and core_ok and len(orders) > 1 and vs.varied == 8
           and vs.fallbacks == 0,
           f"{len(orders)} distinct question subsets/orders · json_action questions "
           f"kept {core_ok} · varied {vs.varied}/{vs.drawn} · fallbacks {vs.fallbacks}")
    except Exception as exc:
        _crashed(nm, exc)

    # B · option shuffle: A/B/C name other options; the un-shuffled logits
    #     decode to the ORIGINAL labels and action
    nm = "vary: option shuffle remaps A/B/C and decodes back to the original"
    ep_v = None
    try:
        raw_v = SensorWorld(29).episode(3, 2 * W_v)
        seed_v = None
        for s_try in range(1000, 1200):
            d = InterfaceSampler(qbind, tok, prob=1.0, seed=s_try).draw(3)
            if (d.perms and "grip_state" in d.qbind.bank
                    and any(p != sorted(p) for rows in d.perms.values() for p in rows)):
                seed_v = s_try
                break
        if seed_v is None:
            raise RuntimeError("no draw with a shuffle + the distractor in 200 seeds")
        # twin samplers, same seed: the first draw of apply() IS iface_v
        iface_v = InterfaceSampler(qbind, tok, prob=1.0, seed=seed_v).draw(3)
        ep_v = InterfaceSampler(qbind, tok, prob=1.0, seed=seed_v).apply(raw_v, tok)
        qb_v = ep_v.qbind
        want = [tok.encode(f"{t}\n{iface_v.schemas[b]}", add_special_tokens=False)
                for b, t in enumerate(raw_v.texts)]
        p_ok = all(ep_v.instr_ids[b, :len(w)].tolist() == w
                   and int(ep_v.instr_mask[b].sum()) == len(w)
                   for b, w in enumerate(want))
        s_ok = True
        for q, rows in iface_v.perms.items():
            for b, p in enumerate(rows):
                pairs = list(zip(qb_v.tags[q], [qb_v.bank[q].labels[j] for j in p]))
                s_ok &= (" ".join(f"{t}={n}" for t, n in pairs) in iface_v.schemas[b]
                         or json.dumps(dict(pairs)) in iface_v.schemas[b])
        perm_ok = (ep_v.opt_perm is not None and set(ep_v.opt_perm) == set(iface_v.perms)
                   and all(ep_v.opt_perm[q].tolist() == iface_v.perms[q]
                           for q in iface_v.perms))
        # a model that answers exactly the SHOWN (re-lettered) targets …
        lab_v = episode_labels(ep_v, qb_v.bank)
        shown = shown_targets(lab_v, ep_v.opt_perm)
        fake = {q: F.one_hot(shown[q], qb_v.bank[q].n_out).float() * 5.0
                for q in lab_v if qb_v.bank[q].n_out > 1}
        # … decodes, after un-shuffling, to the original labels and action
        canon = canonical_q_logits(fake, ep_v.opt_perm)
        r_ok = all(bool((canon[q].argmax(-1) == lab_v[q]).all()) for q in fake)
        a_ok = bool((json_action(canon).long() == ep_v.actions.long()).all())
        moved = sum(int((shown[q] != lab_v[q]).sum()) for q in iface_v.perms)
        hand = shown_targets({"q": torch.tensor([[0, 1, 2]])},
                             {"q": torch.tensor([[2, 0, 1]])})["q"].tolist()
        hand_lg = canonical_q_logits({"q": torch.tensor([[[10.0, 20.0, 30.0]]])},
                                     {"q": torch.tensor([[2, 0, 1]])})["q"].tolist()
        h_ok = hand == [[1, 2, 0]] and hand_lg == [[[20.0, 30.0, 10.0]]]
        ok(nm, p_ok and s_ok and perm_ok and r_ok and a_ok and h_ok
           and ep_v.variant == "train" and _markers_ok(ep_v),
           f"seed {seed_v} · prompt {p_ok} · schema shows the shuffle {s_ok} · "
           f"opt_perm {perm_ok} · labels {r_ok} · action {a_ok} · {moved} targets "
           f"re-lettered · hand case {h_ok}")
    except Exception as exc:
        _crashed(nm, exc)

    # C · the held-out wording set and state format never reach training
    nm = "vary: held-out wording set + state format are never drawn for training"
    try:
        tr = InterfaceSampler(qbind, tok, prob=1.0, seed=17)
        ho_keys = set(HELDOUT_STATE_FORMAT.keys)
        held_txt = [paraphrase(s, PARAPHRASE_HELDOUT)
                    for s in list(qbind.bank.specs) + [DISTRACTOR_QUESTION]]
        seen_idx: set = set()
        leak: List[str] = []
        for i in range(200):
            d = tr.draw(2, libero=bool(i % 2))
            if (set(d.fmt.keys) & ho_keys or d.fmt.name == "heldout"
                    or d.fmt == HELDOUT_STATE_FORMAT):
                leak.append(f"format {d.fmt}")
            for row in d.paras or []:
                seen_idx.update(row.values())
            for sch in d.schemas or []:
                leak += [t for t in held_txt if t in sch]
        hs = InterfaceSampler(qbind, tok, mode="heldout")
        hd = hs.draw(2)
        he = hs.apply(SensorWorld(31).episode(2, W_v), tok)
        h_ok = (hd.fmt == HELDOUT_STATE_FORMAT and hd.variant == "heldout"
                and all(paraphrase(s, PARAPHRASE_HELDOUT) in sch
                        for sch in hd.schemas for s in qbind.bank.specs)
                and he.variant == "heldout" and _markers_ok(he)
                and all(t.startswith("cm eef ") for row in he.state_texts for t in row))
        para_ok = all(len(set(v)) == len(v) for v in PARAPHRASES.values())
        ok(nm, not leak and seen_idx == {0, 1, 2} and h_ok and para_ok,
           f"200 training draws: wording sets {sorted(seen_idx)} · leaks {leak[:2]} · "
           f"held-out draw {h_ok} · wordings distinct {para_ok}")
    except Exception as exc:
        _crashed(nm, exc)

    # D · LIBERO: a converter-format jsonl on disk -> LiberoData / LiberoWorld
    tmp_lib = tempfile.TemporaryDirectory()
    ld = lraw = None
    nm = "libero: loader maps answers to bank labels, splits, skips, meta.json weights"
    try:
        AX3 = ("negative", "stay", "positive")

        def _lib_frame(n_: int, t_: int) -> Dict[str, Any]:
            return {"t": t_,
                    "state": (f"tool {0.1 * ((t_ + n_) % 5) - 0.2:+.2f} "
                              f"{0.03 * t_ - 0.2:+.2f} {0.9 - 0.02 * t_:+.2f} "
                              f"vel {0.01 * (t_ % 3) - 0.01:+.2f} {0.02:+.2f} "
                              f"{-0.01 * (t_ % 2):+.2f} grip {0.080 - 0.005 * t_:.3f}"),
                    "answers": {"axis_x": AX3[(t_ + n_) % 3],
                                "axis_y": AX3[(2 * t_ + n_) % 3],
                                "gripper": ("open", "close", "stay")[(t_ // 4) % 3],
                                "intent": ("approach", "grasp", "hold")[(t_ // 4) % 3],
                                "height": ("down", "stay", "up")[t_ % 3]},
                    "say": {5: "grasped the bowl", 9: "released the bowl"}.get(t_)}

        lib_path = os.path.join(tmp_lib.name, "episodes.jsonl")
        with open(lib_path, "w", encoding="utf-8") as fh:
            for n_d, split_d, len_d in ((0, "train", 12), (1, "train", 12),
                                        (2, "train", 12), (3, "eval", 12),
                                        (4, "train", 2)):          # 2 frames: skipped
                fh.write(json.dumps({"episode": n_d, "split": split_d,
                                     "prompt": "put the bowl on the plate",
                                     "frames": [_lib_frame(n_d, t) for t in range(len_d)]})
                         + "\n")
        with open(os.path.join(tmp_lib.name, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"class_weights": {
                "axis_x": {"negative": 2.0, "stay": 0.5, "positive": 2.0},
                "gripper": {"open": 1.5, "close": 3.0, "stay": 0.25},
                "speed": {"slow": 9.0, "medium": 9.0, "fast": 9.0}}}, fh)
        ld = LiberoData(lib_path, bank, min_len=W_v)
        d_ok = (ld.qids == ["axis_x", "axis_y", "gripper", "intent", "height"]
                and len(ld.episodes["train"]) == 3 and len(ld.episodes["eval"]) == 1
                and ld.skipped == 1)
        w_ok = (ld.q_weights is not None and set(ld.q_weights) == {"axis_x", "gripper"}
                and ld.q_weights["axis_x"].tolist() == [2.0, 0.5, 2.0]
                and ld.q_weights["gripper"].tolist() == [1.5, 3.0, 0.25])
        e0 = ld.episodes["train"][0]
        l_ok = (e0.labels["axis_x"][:3].tolist() == [0, 1, 2]
                and e0.labels["gripper"][:5].tolist() == [0, 0, 0, 0, 1]
                and e0.say[5] == "grasped the bowl" and e0.say[4] == ""
                and tuple(e0.vals.shape) == (12, 7))
        ev = ld.episodes["eval"][0]
        rt_ok = ([render_state_as(DEFAULT_STATE_FORMAT, v[0:3], v[3:6], v[6], None,
                                  grip_len=True) for v in ev.vals.tolist()]
                 == [_lib_frame(3, t)["state"] for t in range(len(ev))])
        bad = _lib_frame(0, 0)
        bad["answers"]["axis_x"] = "sideways"
        bad_path = os.path.join(tmp_lib.name, "bad.jsonl")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"split": "train", "prompt": "x", "frames": [bad] * W_v})
                     + "\n")
        try:
            LiberoData(bad_path, bank)
            bad_ok = False
        except ValueError:
            bad_ok = True
        try:
            LiberoData(os.path.join(tmp_lib.name, "missing.jsonl"), bank)
            miss_ok = False
        except FileNotFoundError:
            miss_ok = True
        ok(nm, d_ok and w_ok and l_ok and rt_ok and bad_ok and miss_ok,
           f"{ld.summary()} · weights {w_ok} · labels {l_ok} · state sentence round "
           f"trip {rt_ok} · unknown answer raises {bad_ok} · missing file raises {miss_ok}")
    except Exception as exc:
        _crashed(nm, exc)

    nm = "libero: windows collate with markers, given labels, speech and speaking probes"
    try:
        if ld is None:
            raise RuntimeError("the LIBERO loader check failed")
        lw_t = LiberoWorld(ld, "train", seed=3, window=W_v)
        lraw = lw_t.episode(2, 2 * W_v)
        lcol = collate(lraw, tok, qbind)
        llab = episode_labels(lcol, bank)
        b_ok = (tuple(lraw.sensors.shape) == (2, 2 * W_v, 7)
                and lraw.tasks.tolist() == [-1, -1] and lcol.source == "libero"
                and _markers_ok(lcol) and _markers_ok(lcol.window(W_v, W_v))
                and list(llab) == ld.qids
                and all(tuple(v.shape) == (2, 2 * W_v) for v in llab.values())
                and bool((lraw.actions == json_action_from_labels(lraw.labels)).all())
                and lcol.speech_ids is not None
                and lcol.q_weights is not None and set(lcol.q_weights) == set(ld.q_weights)
                and tuple(lcol.window(W_v, W_v).labels["axis_x"].shape) == (2, W_v))
        lsw = LiberoWorld(ld, "eval", seed=9, window=W_v, speaking=True)
        pr = speech_probe(collate(lsw.episode(2, 2 * W_v), tok, qbind), W_v,
                          random.Random(4))
        spk_ok = (pr is not None and tuple(pr.sensors.shape[:2]) == (2, W_v)
                  and bool(((pr.speech_ids[:, -1] >= 0).sum(-1) > 1).all())
                  and _markers_ok(pr))
        lv = InterfaceSampler(qbind, tok, prob=1.0, seed=23)
        v_ok = True
        for _ in range(60):
            d = lv.draw(2, libero=True)
            v_ok &= "grip_state" not in d.qbind.bank and _F_VEL not in d.fmt.drop
        lep = lv.apply(lw_t.episode(2, 2 * W_v), tok)
        va_ok = (lep.variant == "train" and lep.source == "libero" and _markers_ok(lep)
                 and lep.labels is not None and set(lep.labels) == set(ld.qids)
                 and lep.q_weights is not None)
        ok(nm, b_ok and spk_ok and v_ok and va_ok,
           f"batch {b_ok} · speaking probe {spk_ok} · varied draws (no distractor, "
           f"velocity kept) {v_ok} · varied batch {va_ok}")
    except Exception as exc:
        _crashed(nm, exc)

    # E · joint_loss: given per-option weights; task CE masked on task < 0 rows
    nm = "loss: given class weights replace in-batch balance; task CE masks task<0 rows"
    try:
        g_v = torch.Generator().manual_seed(0)
        lg_e = torch.randn(2, 3, 3, generator=g_v)
        lab_e = torch.tensor([[0, 1, 2], [1, 1, 0]])
        w_e = torch.tensor([2.0, 0.5, 4.0])
        tl_e = torch.randn(2, 3, len(TASKS), generator=g_v)

        def _fake_out() -> LoopOutput:
            return LoopOutput(ponder=torch.zeros(2, 3), cycles=torch.zeros(2, 3),
                              carry=torch.zeros(2, 4), q_logits={"axis_x": lg_e},
                              task_logits=tl_e)

        flat_lg, flat_lab = lg_e.reshape(-1, 3), lab_e.reshape(-1)
        _, p_w = joint_loss(_fake_out(), 0.0, q_labels={"axis_x": lab_e},
                            q_class_weights={"axis_x": w_e})
        _, p_n = joint_loss(_fake_out(), 0.0, q_labels={"axis_x": lab_e})
        ref_w = F.cross_entropy(flat_lg, flat_lab, weight=w_e)
        ref_n = F.cross_entropy(flat_lg, flat_lab, weight=_class_weights(flat_lab, 3))
        _, p_t1 = joint_loss(_fake_out(), 0.0, task_ids=torch.tensor([1, -1]),
                             task_weight=1.0)
        _, p_t0 = joint_loss(_fake_out(), 0.0, task_ids=torch.tensor([-1, -1]),
                             task_weight=1.0)
        _, p_t2 = joint_loss(_fake_out(), 0.0, task_ids=torch.tensor([1, 2]),
                             task_weight=1.0)
        ref_t1 = F.cross_entropy(tl_e[0], torch.full((3,), 1, dtype=torch.long))
        ref_t2 = F.cross_entropy(tl_e.reshape(-1, len(TASKS)),
                                 torch.tensor([1, 2])[:, None].expand(2, 3).reshape(-1))
        e_ok = (abs(float(p_w["questions"]) - float(ref_w)) < 1e-5
                and abs(float(p_n["questions"]) - float(ref_n)) < 1e-5
                and abs(float(p_w["questions"]) - float(p_n["questions"])) > 1e-4
                and abs(float(p_t1["task"]) - float(ref_t1)) < 1e-5
                and float(p_t0["task"]) == 0.0
                and abs(float(p_t2["task"]) - float(ref_t2)) < 1e-5)
        ok(nm, e_ok,
           f"weighted {float(p_w['questions']):.4f}/{float(ref_w):.4f} · balanced "
           f"{float(p_n['questions']):.4f}/{float(ref_n):.4f} · one row masked "
           f"{float(p_t1['task']):.4f}/{float(ref_t1):.4f} · all masked "
           f"{float(p_t0['task'])} · none masked {float(p_t2['task']):.4f}/"
           f"{float(ref_t2):.4f}")
    except Exception as exc:
        _crashed(nm, exc)

    # F · the real train step on varied synthetic, varied LIBERO and default
    #     batches (the three kinds run_train interleaves)
    nm = "steps: varied synthetic, LIBERO and default batches train with finite loss"
    try:
        if ep_v is None or lraw is None:
            raise RuntimeError("an earlier vary / LIBERO check failed")
        opt_v = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4)
        step_v = make_step_fn(model, opt_v, None, accel, cfg.tau, 1.0, q_weight=1.0,
                              text_weight=0.5, task_weight=0.5)
        lvar = InterfaceSampler(qbind, tok, prob=1.0, seed=41).apply(
            LiberoWorld(ld, "train", seed=5, window=W_v).episode(2, 2 * W_v), tok)
        seen_w: List[Any] = []
        _jl = joint_loss

        def _spy(*a_, **k_):
            seen_w.append(k_.get("q_class_weights"))
            return _jl(*a_, **k_)

        globals()["joint_loss"] = _spy           # the step calls the module global
        try:
            a_syn = step_v([ep_v.window(0, W_v).to(dev_v)], None)
            a_lib = step_v([lvar.window(0, W_v).to(dev_v)], None)
            a_def = step_v([collate(SensorWorld(61).episode(2, W_v), tok,
                                    qbind).to(dev_v)], None)
        finally:
            globals()["joint_loss"] = _jl
        fin = all(math.isfinite(a["loss"]) and a["grad_bad"] == 0
                  for a in (a_syn, a_lib, a_def))
        f_ok = (fin and len(seen_w) == 3 and seen_w[0] is None and seen_w[2] is None
                and seen_w[1] is not None and set(seen_w[1]) == {"axis_x", "gripper"}
                and a_lib["task_loss"] == 0.0 and a_def["task_loss"] > 0.0)
        ok(nm, f_ok,
           f"loss varied={a_syn['loss']:.3f} libero={a_lib['loss']:.3f} "
           f"default={a_def['loss']:.3f} · LIBERO CE weights "
           f"{sorted(seen_w[1]) if len(seen_w) > 1 and seen_w[1] else None} · task "
           f"loss libero={a_lib['task_loss']:.3f} default={a_def['task_loss']:.3f}")
    except Exception as exc:
        _crashed(nm, exc)

    # G · eval: natural window ends that never speak (the 'speaking 0/32' run)
    #     still yield a speech sentence accuracy, from the speaking probes
    nm = "eval: speech sentence acc is measured when natural window ends are silent"
    try:
        class _TalkyWorld(SensorWorld):
            """One short utterance at frame W_v (active W_v and W_v+1): every
            natural window of W_v frames ends silent."""

            def episode(self, batch: int, steps: int) -> Episode:
                e_ = super().episode(batch, steps)
                e_.speech = [["done" if t == W_v else "" for t in range(steps)]
                             for _ in range(batch)]
                return e_

        st_nat = EpisodeStream(_TalkyWorld(5), 2, W_v, 2, tok, qbind=qbind)
        ends: List[int] = []
        for _ in range(2):
            win_e, _c = st_nat.next(cpu_v)
            ends.append(int(((win_e.speech_ids[:, -1] >= 0).sum(-1) > 1).sum()))
        ecfg = argparse.Namespace(speak_tokens=1, eval_batches=2, window=W_v,
                                  windows_per_episode=2, batch=2, carry=True, seed=0)
        res = evaluate(ecfg, accel, model, tok, _TalkyWorld(5), 0, libero=ld)
        fin_ = (lambda k: k in res and math.isfinite(float(res[k])))
        g_ok = (ends == [0, 0] and fin_("speech_sentence_acc")
                and float(res.get("speech_speaking_frames", 0.0)) > 0
                and fin_("json_action_acc_heldout") and fin_("libero_json_action_acc")
                and fin_("libero_speech_decide_acc") and fin_("libero_speech_sentence_acc")
                and any(k.startswith("libero_q_acc_") for k in res)
                and any(k.startswith("heldout_q_acc_") for k in res))
        ok(nm, g_ok,
           f"natural window ends speaking {ends} · sentence="
           f"{res.get('speech_sentence_acc')} over {res.get('speech_speaking_frames')} "
           f"frames · heldout json={res.get('json_action_acc_heldout')} · libero "
           f"json={res.get('libero_json_action_acc')} "
           f"sentence={res.get('libero_speech_sentence_acc')}")
    except Exception as exc:
        _crashed(nm, exc)

    # H · the checkpoint's interface record rebuilds the DEFAULT bank
    nm = "checkpoint: interface record rebuilds the default bank; old payloads -> None"
    try:
        rec = interface_record(qbind, argparse.Namespace(vary=0.75, libero="x.jsonl",
                                                         libero_frac=0.3))
        back = bank_from_payload({"interface": json.loads(json.dumps(rec))})
        rb_ok = (back is not None
                 and [s.qid for s in back.specs] == [s.qid for s in qbind.bank.specs]
                 and QuestionBinding(back, tok).schema == qbind.schema
                 and rec["schema"] == qbind.schema
                 and rec["markers"] == list(qbind.markers)
                 and rec["vary"] == 0.75 and rec["libero_frac"] == 0.3
                 and bank_from_payload({}) is None
                 and bank_from_payload({"qbind": qbind.signature()}) is None
                 and bank_from_payload(None) is None)
        ok(nm, rb_ok, f"record keys {sorted(rec)}")
    except Exception as exc:
        _crashed(nm, exc)

    # I · regression: without --vary / --libero the batches are the pre-vary
    #     ones (schema text, token ids, marker positions, speech targets)
    nm = "default path: schema, collate ids and stream tensors equal the pre-vary code"
    try:
        legacy = ["schema:"]
        for m, s in zip(qbind.markers, qbind.bank.specs):
            opts = ("|".join(s.labels) if s.type == "noul" else
                    " ".join(f"{t}={n}" for t, n in zip(qbind.tags[s.qid], s.labels)))
            legacy.append(f"- {m}: {s.instructions or f'What is {s.qid}?'} "
                          f"options: {opts}")
        legacy.append("answer the markers in order (" + " ".join(qbind.markers)
                      + ") right after each state.")
        legacy_schema = "\n".join(legacy)
        rraw = SensorWorld(41).episode(3, 2 * W_v)
        rcol = collate(rraw, tok, qbind)
        want_p = [tok.encode(f"{t}\n{legacy_schema}", add_special_tokens=False)
                  for t in rraw.texts]
        pad_id = int(getattr(tok, "pad_token_id", None)
                     or getattr(tok, "eos_token_id", None) or 0)
        want_ids = torch.full((3, max(1, max(len(w) for w in want_p))), pad_id,
                              dtype=torch.long)
        for b, w in enumerate(want_p):
            want_ids[b, :len(w)] = torch.tensor(w, dtype=torch.long)
        suffix_l = " " + " ".join(qbind.markers)
        rows_l = [[tok.encode(f"{st}{suffix_l} {STATE_MARKER}", add_special_tokens=False)
                   for st in row] for row in rraw.state_texts]
        L_l, Q_l = len(rows_l[0][0]), len(qbind.bank.specs)
        want_s = torch.tensor([sum(row, []) for row in rows_l], dtype=torch.long)
        want_a = torch.tensor([[[s_ * L_l + (L_l - 1 - Q_l + q_) for q_ in range(Q_l)]
                                for s_ in range(2 * W_v)]] * 3, dtype=torch.long)
        sp_w, so_w = speech_stream_targets(rraw.speech, tok, 1)
        c_ok = (qbind.schema == legacy_schema
                and torch.equal(rcol.instr_ids, want_ids)
                and torch.equal(rcol.state_ids, want_s)
                and torch.equal(rcol.ans_pos, want_a)
                and torch.equal(rcol.speech_ids, sp_w)
                and torch.equal(rcol.speech_off, so_w)
                and rcol.qbind is None and rcol.opt_perm is None
                and rcol.labels is None and rcol.variant == "default"
                and render_episode_states(rraw, DEFAULT_STATE_FORMAT) == rraw.state_texts)
        s_plain = EpisodeStream(SensorWorld(43), 2, W_v, 2, tok, qbind=qbind)
        s_iface = EpisodeStream(SensorWorld(43), 2, W_v, 2, tok, qbind=qbind,
                                interface=InterfaceSampler(qbind, tok, prob=0.0, seed=1))
        same = True
        for _ in range(4):
            a_w, _ = s_plain.next(cpu_v)
            b_w, _ = s_iface.next(cpu_v)
            for f_ in ("instr_ids", "instr_mask", "state_ids", "ans_pos", "speech_ids",
                       "speech_off", "actions", "sensors", "tasks"):
                same &= torch.equal(getattr(a_w, f_), getattr(b_w, f_))
            same &= (b_w.qbind is None and b_w.opt_perm is None
                     and b_w.variant == "default")
        ok(nm, c_ok and same, f"collate vs legacy formula {c_ok} · stream with a "
           f"prob-0 interface sampler {same}")
    except Exception as exc:
        _crashed(nm, exc)
    try:
        tmp_lib.cleanup()
    except OSError:
        pass

    print("-" * 78)
    print(f"SELF-TEST {'PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


# =============================================================================
# 10 · CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Continuous-latent looped transformer + ACT + gated sensor fusion "
                    "(AMD Radeon 880M / Windows 11).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("hardware")
    g.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    g.add_argument("--gpu-index", type=int, default=0)
    g.add_argument("--vram-fraction", type=float, default=0.85,
                   help="hard allocator ceiling; leaves head-room for the desktop")
    g.add_argument("--vram-budget-gib", type=float, default=8.0,
                   help="absolute allocator ceiling in GiB — the effective cap is "
                        "min(--vram-fraction x reported pool, this). On an integrated "
                        "GPU the reported pool is shared system RAM, so this must be "
                        "the BIOS UMA carve-out (8 for this machine). 0 = disabled")
    g.add_argument("--trace", type=int, default=0, metavar="N",
                   help="print synchronised per-phase timings for the first N "
                        "steps (episode build / forward / loss / backward / "
                        "optimiser).  A step that spends its life inside one "
                        "CUDA op otherwise looks exactly like a hung one: the "
                        "log says nothing until the step ends.")
    g.add_argument("--gc-every", type=int, default=0,
                   help="every N steps, run gc.collect() + empty the caching "
                        "allocator (0 = off, the default: the full Python GC "
                        "pass measures 215-261 ms on this box, so it belongs at "
                        "eval/checkpoint boundaries, not in the control loop; "
                        "flush_if_low() still fires under real pressure)")
    g.add_argument("--attn-impl", default="auto", choices=("auto", "sdpa", "eager"),
                   help="trunk attention backend; 'auto' = probed on the device")
    g.add_argument("--gfx-override", default="",
                   help="HSA_OVERRIDE_GFX_VERSION, e.g. 11.0.0 — it is read before "
                        "`import torch`, so pass it as a command-line flag")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16"))

    g = p.add_argument_group("model")
    g.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help="HF id · local path · 'tiny' (random Qwen2, offline)")
    g.add_argument("--trainable", default="adapters",
                   choices=("adapters", "lora", "recurrent", "full"),
                   help="which parameters receive gradients; 'lora' adapts the "
                        "whole trunk through low-rank updates (needs --lora-rank)")
    g.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA rank r; 0 disables it.  16 adapts a 0.5B trunk "
                        "with ~8.8M trainable params")
    g.add_argument("--lora-alpha", type=float, default=16.0,
                   help="LoRA scaling alpha (the effective step is alpha/r)")
    g.add_argument("--lora-dropout", type=float, default=0.0,
                   help="dropout on the LoRA input path")
    g.add_argument("--loops", type=int, default=4,
                   help="hard CEILING on ACT cycles per window.  The loop is "
                        "halting-driven: it breaks the moment every position has "
                        "halted, so this is a safety net, not a budget.  With "
                        "--cycle-tag learned the cycle table has one row per "
                        "cycle, so this is also the structural bound; with "
                        "sinusoidal the tag is a function of the cycle index and "
                        "--loops 64 is effectively unbounded thinking")
    g.add_argument("--cycle-tag", choices=("learned", "sinusoidal"), default="learned",
                   help="how the loop tags each cycle.  learned = a zero-init "
                        "row per cycle (bounded by --loops, the default, "
                        "checkpoint-compatible); sinusoidal = a parameter-free "
                        "encoding of the cycle INDEX behind a zero-init gate, "
                        "which is what makes the loop unbounded — the number of "
                        "cycles is then the model's own call")
    g.add_argument("--recurrent-layers", type=int, default=2,
                   help="trunk layers reused by the loop")
    g.add_argument("--fusion-heads", type=int, default=8)
    g.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute the recurrent block in backward (saves VRAM)")
    g.add_argument("--carry", action=argparse.BooleanOptionalAction, default=True,
                   help="thread the latent across windows (never reset on action)")
    g.add_argument("--questions", default="auto",
                   help="questionnaire for the JSON decision interface — the schema "
                        "is rendered into the prompt, so it is DATA: 'auto' = "
                        "axis_x/axis_y/gripper/intent/height/speed, 'off', a JSON "
                        "string, or a path to a .json file")
    g.add_argument("--q-weight", type=float, default=1.0,
                   help="weight of the per-question option cross-entropy")
    g.add_argument("--text-weight", type=float, default=0.0,
                   help="weight of the SPEECH cross-entropy: the sentence (plus the "
                        "end token) the model says after the last state step; a "
                        "silent frame's target is the end token alone")
    g.add_argument("--speak-tokens", type=int, default=1, choices=[1, 2],
                   help="STREAMED speech: tokens emitted per control frame (K). "
                        "An utterance is spread over consecutive frames; K=2 costs "
                        "at most one extra forward per frame")
    g.add_argument("--untie-lm-head", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="give the token path its own trainable Linear(hidden,vocab) "
                        "instead of the trunk's tied (frozen) embedding matrix — "
                        "required for speech to actually train under "
                        "--trainable adapters; ~136M params on Qwen2.5-0.5B")
    g.add_argument("--task-weight", type=float, default=0.0,
                   help="auxiliary CE that names the VERB the prompt asked for. "
                        "The question loss alone gives the instruction a sparse "
                        "signal (all four verbs steer the same way until the tool "
                        "is near the target), so the sentence never reaches the "
                        "read-out.  This term forces the prompt through the "
                        "trainable task_proj shortcut that is added to every "
                        "answer latent; 0.5 is a good value")

    g = p.add_argument_group("data")
    g.add_argument("--batch", type=int, default=4)
    g.add_argument("--window", type=int, default=24, help="control steps per window")
    g.add_argument("--windows-per-episode", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--vary", type=float, nargs="?", const=0.75, default=0.0,
                   metavar="P",
                   help="training-time interface variation: each stream episode is "
                        "re-rendered with probability P (bare flag = 0.75) — question "
                        "subset/order, paraphrases, option order (A/B/C remapped), "
                        "schema layout, a distractor question, state field names/"
                        "order/units/dropped velocity/noise.  Paraphrase set 3 and "
                        "the 'heldout' state format stay eval-only.  0 = off (the "
                        "default interface only)")
    g.add_argument("--libero", default=None, metavar="PATH",
                   help="libero_convert.py episodes.jsonl (meta.json next to it "
                        "gives per-option class weights); a missing/unreadable file "
                        "warns and training continues synthetic-only")
    g.add_argument("--libero-frac", type=float, default=0.0, metavar="F",
                   help="share of training batches drawn from --libero (whole "
                        "LIBERO-only batches; 0 = LIBERO for eval only)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=1000)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--lr-hold", type=float, default=0.0,
                   help="fraction of --steps held at peak lr after warm-up "
                        "before the cosine decay starts")
    g.add_argument("--lr-floor", type=float, default=0.05,
                   help="final lr as a fraction of --lr")
    g.add_argument("--ckpt-every", type=int, default=0,
                   help="also save the checkpoint every N steps (overwrites --out)")
    g.add_argument("--init-from", default=None, metavar="CKPT",
                   help="start from this checkpoint's trainable tensors (fine-tune); "
                        "optimiser and lr schedule start fresh")
    g.add_argument("--tau", type=float, default=1e-2, help="ponder penalty weight")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=250)
    g.add_argument("--eval-batches", type=int, default=2)
    g.add_argument("--out", default="robot_act.pt")
    g.add_argument("--self-test", action="store_true",
                   help="run architecture invariants on a tiny random trunk and exit")
    return p


def main() -> int:
    cfg = build_parser().parse_args()
    if cfg.self_test:
        cfg.model = "tiny"                      # self-test never needs a download
    accel = verify_hardware(cfg)
    if cfg.self_test:
        return run_self_test(cfg, accel)

    tok, trunk = load_base(cfg.model, accel)
    bank = resolve_questions(cfg.questions)
    qbind = QuestionBinding(bank, tok) if bank is not None else None
    model = LoopedACTTransformer(
        trunk, num_looped_layers=cfg.recurrent_layers, max_loops=cfg.loops,
        pond_tau=cfg.tau, fusion_heads=cfg.fusion_heads,
        grad_checkpoint=cfg.grad_checkpoint,
        qbind=qbind,
        untie_lm_head=cfg.untie_lm_head,
        lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        cycle_tag=cfg.cycle_tag,
        lora_dropout=cfg.lora_dropout,
    ).to(accel.device)
    configure_trainable(model, cfg.trainable)
    report_params(model, accel)
    if bank is not None:
        print(f"  interface   : JSON answers · {len(bank)} question(s) "
              f"[{' '.join(f'{s.qid}:{s.type}' for s in bank.specs)}] · "
              f"{bank.total} option logits read at markers "
              f"{' '.join(qbind.markers)} · schema {len(qbind.schema)} chars · "
              f"q_weight={cfg.q_weight} · text_weight={cfg.text_weight} · "
              f"task_weight={cfg.task_weight}")
    if model.lm_head is not None:
        print(f"  lm head     : untied trunk → trainable Linear(hidden, vocab) "
              f"({model.lm_head.weight.numel()/1e6:.2f}M params)")
    elif model.lm_weight is not None:
        print("  lm head     : tied to embed_tokens (no extra parameters)")
        if cfg.text_weight > 0.0:
            print("  NOTE        : the tied head IS the frozen embedding matrix, so "
                  "the token loss can only move the latent — add --untie-lm-head "
                  "to train a real decoder for speech")
    if cfg.trainable == "full":
        print("  WARNING     : full fine-tuning needs ≈8 GB of AdamW state for a "
              "0.5B trunk; prefer --trainable recurrent on an 8 GB carve-out.")
    world = SensorWorld(seed=cfg.seed)
    libero = None
    if getattr(cfg, "libero", None):
        if bank is None:
            print(f"  [libero] WARNING: --questions off — ignoring --libero {cfg.libero}")
        elif not os.path.isfile(cfg.libero):
            print(f"  [libero] WARNING: {cfg.libero} not found — continuing "
                  f"synthetic-only")
        else:
            t_l = time.time()
            try:
                libero = LiberoData(cfg.libero, bank, min_len=cfg.window)
                print(f"  libero      : {libero.summary()} · loaded in "
                      f"{time.time() - t_l:.1f}s · --libero-frac {cfg.libero_frac}")
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                print(f"  [libero] WARNING: could not load {cfg.libero} "
                      f"({type(exc).__name__}: {exc}) — continuing synthetic-only")
                libero = None
    init = getattr(cfg, "init_from", None)
    if init:
        if not os.path.isfile(init):
            raise SystemExit(f"--init-from: {init} not found")
        payload = torch.load(init, map_location="cpu")
        if not isinstance(payload, dict) or "trainable_state" not in payload:
            raise SystemExit(f"--init-from: {init} is not a train_loop_robot checkpoint")
        src_args = payload.get("args", {}) or {}
        diff = [f"{k}: ckpt {src_args.get(k)!r} vs now {getattr(cfg, k, None)!r}"
                for k in ("trainable", "lora_rank", "lora_alpha", "loops",
                          "recurrent_layers", "fusion_heads", "carry", "untie_lm_head")
                if k in src_args and src_args.get(k) != getattr(cfg, k, None)]
        if diff:
            raise SystemExit("--init-from: architecture differs from the checkpoint "
                             "(pass matching flags): " + "; ".join(diff))
        missing, unexpected = load_trainable_state(model, payload, qbind,
                                                   where="--init-from")
        own = {n for n, p in model.named_parameters() if p.requires_grad}
        got = own & set(payload["trainable_state"])
        print(f"  init from   : {init} · {len(got)}/{len(own)} trainable tensors "
              f"loaded · {len(unexpected)} unexpected · optimiser + lr schedule "
              f"start fresh")
        if len(got) < len(own):
            raise SystemExit(f"--init-from: only {len(got)}/{len(own)} trainable "
                             f"tensors found in {init}")
        del payload
    run_train(cfg, accel, model, tok, world, libero=libero)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
