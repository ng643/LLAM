#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mujoco_env_bridge.py — live MuJoCo control loop for the continuous-latent
Looped-Transformer policy trained by `train_loop_robot.py`.

WHAT THIS IS
------------
`train_loop_robot.py` learns a policy on a synthetic 28-D sensorimotor stream.
This script replaces that stream with a *real* physics engine: a procedural
MJCF manipulator (3 revolute joints + a tendon-driven gripper) chasing four
dynamic spheres.  The policy keeps thinking exactly as it was trained — the
latent state is threaded across physics frames and is NEVER reset on action
emission — while the actuator command path becomes physical torque.

DATA FLOW (one control step)
----------------------------
    MuJoCo (CPU, physics)                    PyTorch (Radeon 880M / ROCm)
    ─────────────────────                    ────────────────────────────
    d.qpos / d.qvel / site_xpos
        └─ observe()  ─────────►  28-D float32 observation
                                  └─ pinned host buffer
                                  └─ .to(dev, non_blocking=True)   async H2D
                                  └─ rolling window  (1,S,28)  S=--window
                                  └─ LoopedACTTransformer.forward(instr+schema,
                                                                  state, carry)
                                       ├─ PromptFusion     : cross-attn on the
                                       │    STATE region, tanh(gate) -> 0 at init
                                       ├─ ACT loop         : halt_head -> Σα = 1
                                       │    h* = Σ α_n · h_n   (weighted state)
                                       ├─ answers          : h_ans + task_proj
                                       │    -> LM head on Jev labels (A/B/C, 0/1/2)
                                       └─ speech           : LM head at the last
                                            state position -> first token
                                  JSON document = typed answers + derived
                                       `action` (tlr.json_action) + `say`
                                  control is ALWAYS the JSON:
                                       typed answers -> Command
                                                 (axis_x/axis_y -> direction,
                                                  speed -> scale, gripper/intent
                                                  -> jaw + hold, height -> z)
                                  speech: STREAMED --speak-tokens (1|2) per
                                       frame, END first = silent (no extra
                                       forward); K=2 ≤1 extra; ≤ --speak-max
        data.ctrl (torque) ◄────── τ = Jᵀ·F + qfrc_bias   (or hold / brake)
    mj_step(nstep=--physics-hz/--control-hz)
        └─ carry = out.carry   ──►  next frame (no reset, ever)

WIRING MAP (brief -> implementation in train_loop_robot.py)
-----------------------------------------------------------
  "GatedSensorInjection layer"  -> `GatedSensorFusion` (+ its `SensorTokenizer`
                                   projector); cross-attention with the
                                   zero-initialised `nn.Parameter(torch.zeros(1))`
                                   gate, so the trunk is untouched at step 0.
  "ACT halting head"            -> `ACTHaltingHead` (Linear + sigmoid) driven by
                                   the loop inside `LoopedACTTransformer.forward`;
                                   budget 1.0, convex combination of all loop
                                   states, ponder cost ∈ [1, max_loops].
  "hidden state not reset"      -> `LoopOutput.carry` fed back as the previous
                                   window's mixed latent (a prefix memory token),
                                   exactly as `EpisodeStream` does during training.
  "28-D feature space"          -> `tlr.SENSOR_DIM`, re-asserted below.
  "MemoryGuard"                 -> reused verbatim (`tlr.MemoryGuard`).
  "motor output as JSON"        -> `--json-out`: the model's typed answers
                                   rendered as a document (`tlr.render_answers`
                                   via `PolicyRunner.act`), plus the telemetry
                                   the harness owns (goal distance, ACT cycles).
                                   Nothing is decoded token by token, so the
                                   document is exact and cannot be malformed.
  "JSON drives the arm"         -> `command_from_json()`: the document is bound
                                   to a `Command` (direction / speed scale /
                                   jaw / hold / brake) and the torque map is
                                   applied to that.  There is no action head
                                   any more: the document's `action` field is
                                   DERIVED from the answers (`tlr.json_action`).
                                   Jev-as-Policy's split: the model classifies,
                                   the harness owns geometry, limits, timing
                                   and the e-stop.
  "one policy, many tasks"      -> `--task {reach,grasp,throw,push}`: the task
                                   lives in the instruction TEXT
                                   (`tlr.TASK_TEMPLATES`), never in the
                                   architecture, and the expert that labels the
                                   live states is task-conditioned
                                   (`expert_action(..., task)`).  A new task is
                                   a new sentence.
  "speaks while it acts"        -> `--speak-max N` (default 8 tokens incl. END,
                                   0 = mute), `--speak-tokens K` (1|2): speech
                                   is STREAMED.  Each frame's single forward
                                   carries the utterance spoken so far AFTER the
                                   state; its last position gives the first new
                                   token (END = silent / utterance complete, zero
                                   extra cost); K=2 costs exactly one extra
                                   forward.  The causal mask keeps speech out of
                                   the answers and the carry.  The finished
                                   sentence ("grasped the red block", …) is
                                   `info['say']` on the completing frame and the
                                   document's top-level `say` (null otherwise);
                                   `info['say_partial']` is the text so far.

RUN
---
  # CPU smoke test with a random tiny trunk (no download, no checkpoint)
  python mujoco_env_bridge.py --model tiny --frames 200 --headless

  # NOTE -- `--model tiny` is a FIXED trunk (TINY_TRUNK_SEED, see
  # `tlr.build_tiny_trunk`), not a fresh random one per process.  Before that fix
  # two identical replays of one checkpoint scored 90.0 % and 0.0 % of frames in
  # reach: the adapters were being loaded onto different frozen weights each run.
  # A real model id (Qwen/SmolLM) was never affected -- those weights come from
  # the hub.

  # GPU + a trained adapter (path printed by train_loop_robot.py --out)
  python mujoco_env_bridge.py --checkpoint robot_act.pt --frames 1000

  # watch it think: opens the MuJoCo passive viewer, 5 control Hz
  python mujoco_env_bridge.py --checkpoint robot_act.pt --duration 120

  # headless + periodic PNG frames (also how the loop was verified)
  python mujoco_env_bridge.py --model tiny --frames 120 --headless \
      --render-out frame.png --render-every 40

  # motor output as JSON: one document per control step, atomically replaced
  python mujoco_env_bridge.py --checkpoint robot_act.pt --frames 500 \
      --headless --goal red --json-out decision.json --json-every 1
  #   {"axis_x": {"choice": "positive", "probabilities": {...}}, ...,
  #    "speed": {"score": 1.092, "legend": [...]},
  #    "action": {"type": "derived", "choice": "+x"}, "say": null,
  #    "telemetry": {"goal_m": 0.279, "in_reach": false, "cycles": 4.0, ...}}

NOTES
-----
* Throughput (880M, after the `--gc-every` fix removed a ~215-260 ms full GC pass
  per frame): the tiny trunk runs this loop at ~13-15 Hz (RTF ~2.6-2.9) and
  Qwen2.5-0.5B at ~7.6 Hz (RTF 1.5), so `--realtime` is usable with either — the
  limit is kernel-launch overhead per forward, not the physics.  The physics is
  stepped in `--physics-hz/--control-hz` substeps per policy step and the
  achieved rate + real-time factor are printed.
* Training-only flags inherited from `train_loop_robot.build_parser()`
  (`--steps`, `--out`, `--lr`, `--tau`, `--batch`, `--self-test`, ...) are
  ignored here; the hardware flags (`--vram-budget-gib`, `--vram-fraction`,
  `--gc-every`, `--attn-impl`, `--amp`, `--dtype`, `--device`) are live.
* `--no-carry` exists only as an ablation (it resets the latent every frame and
  is NOT the intended operating mode).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# IMPORT ORDER IS LOAD-BEARING.
# `train_loop_robot` installs the allocator / MIOpen / HSA env guards at import
# time — *before* torch is imported — and also parses `--gfx-override` out of
# sys.argv.  Importing it first therefore configures this process exactly like
# a training run, and gives us the model, the ACT loop and the MemoryGuard.
# ---------------------------------------------------------------------------
import train_loop_robot as tlr                                       # noqa: E402

import torch                                                         # noqa: E402

import mujoco                                                        # noqa: E402

try:                                    # the viewer needs a real GL context
    import mujoco.viewer                                             # noqa: E402
    _VIEWER_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:                                             # pragma: no cover
    _VIEWER_IMPORT_ERROR = exc


# =============================================================================
# 1 · HARNESS STATE CONTRACT  (telemetry + expert labels; the MODEL reads text)
# =============================================================================
# The policy's input is a SENTENCE per control step (`tlr.render_state`), built
# by `state_sentence()` below from this 28-D vector:
#     [ pos(2) | vel(2) | grip(1) | tool_z(1) | rel_xy(O*2) | rel_z(O) | dist(O)
#       | prev_action(A) ]
# where rel_xy/rel_z/dist are stacked per *object* (all rel components first,
# then all height offsets, then all distances — NOT interleaved) and prev_action
# is a one-hot that is all-zero on the very first step of an episode.  The vector
# is what the harness *knows* — the expert rule, the reward, the telemetry and
# the JSON document all read it — but it is no longer fed to the network: the
# sentence is, and any robot that can print that sentence can drive the policy.
NUM_OBJECTS = len(tlr.COLORS)                      # 4 colour slots
SENSOR_DIM = tlr.SENSOR_DIM                        # 28

SL_POS = slice(0, 2)
SL_VEL = slice(2, 4)
SL_GRIP = slice(4, 5)
SL_TOOLZ = slice(5, 6)                             # the tool's own height
SL_REL = slice(6, 6 + 2 * NUM_OBJECTS)             # 4 × (dx, dy)
SL_RELZ = slice(6 + 2 * NUM_OBJECTS, 6 + 3 * NUM_OBJECTS)     # 4 × dz
SL_DIST = slice(6 + 3 * NUM_OBJECTS, 6 + 4 * NUM_OBJECTS)     # 4 × |d_xy|
SL_PREV = slice(6 + 4 * NUM_OBJECTS, SENSOR_DIM)

assert SENSOR_DIM == 28, f"training layout changed: SENSOR_DIM={SENSOR_DIM}"
assert SL_PREV.stop == 6 + 4 * NUM_OBJECTS + tlr.NUM_ACTIONS == SENSOR_DIM
assert tlr.NUM_ACTIONS == 6 and NUM_OBJECTS == 4, "action/object space drifted"

# Height is carried in *training units* (the world keeps `tool_z` in [0, 1] and
# the objects at 0.0 / 0.25 / 0.5), so the bridge maps metres into that scale
# and the `height` answer moves the tool by exactly one world step (HEIGHT_VECS
# is ±0.35 units per control step):
#   training z=0.0 -> Z_FLOOR m   ·   training z=1.0 -> Z_FLOOR + Z_SPAN m
Z_FLOOR = 0.02                                     # a sphere's centre height (m)
Z_SPAN = 0.30                                      # metres per training unit
HEIGHT_STEP_M = 0.35 * Z_SPAN                      # one `down`/`up` answer (m)
Z_MIN, Z_MAX = 0.030, 0.42                         # the height servo's travel (m)

# Training-time magnitudes, used to keep the physical scene inside the
# distribution the policy was fitted on (see SensorWorld.episode):
#   pos ∈ [-1.2, 1.2], obj ∈ [-1, 1], |rel| ≤ ~2.2, dist ≤ ~3
#   the expert moved 0.12 world-units per control step (reach = 0.10 units).
TRAIN_POS_RANGE = 1.2
TRAIN_STEP_UNITS = 0.12                            # units / control step
TRAIN_REACH_UNITS = 0.10                           # "close enough to grasp"


def sensor_layout_table() -> str:
    """Human-readable map of the 28-D vector (printed once at startup)."""
    rows = [
        (SL_POS, "pos_xy", "tool-site x,y (m × --obs-scale)"),
        (SL_VEL, "vel_xy", "tool linear velocity (J·q̇, × --obs-scale)"),
        (SL_GRIP, "gripper", "finger aperture, 0=open 1=closed"),
        (SL_TOOLZ, "tool_z", f"tool height in training units ((z-{Z_FLOOR})/{Z_SPAN})"),
        (SL_REL, "obj_rel", f"{NUM_OBJECTS}× relative xy (all x,y first)"),
        (SL_RELZ, "obj_rel_z", f"{NUM_OBJECTS}× height offset (obj_z - tool_z)"),
        (SL_DIST, "obj_dist", f"{NUM_OBJECTS}× |obj - tool| (planar)"),
        (SL_PREV, "prev_act", f"one-hot of the last emitted token ({tlr.NUM_ACTIONS})"),
    ]
    out = [f"HARNESS STATE (S={SENSOR_DIM})   indices  field      source",
           "  -> rendered per control step as: " + tlr.render_state(
               (0.0, 0.0, 0.0), (0.0, 0.0), 0.0,
               [(0.0, 0.0, 0.0)] * NUM_OBJECTS)[:46] + " …"]
    for sl, name, src in rows:
        idx = f"[{sl.start:2d}:{sl.stop:2d}]" if sl.stop - sl.start > 1 \
            else f"[{sl.start:2d}   ]"
        out.append(f"  {idx} {name:<10} {src}")
    return "\n".join(out)


# =============================================================================
# 2 · PROCEDURAL MJCF  (no external .xml — the scene is generated in Python)
# =============================================================================
# Kinematics: base(yaw) -> shoulder(pitch) -> elbow(pitch) -> wrist + 2 fingers.
# Link lengths 0.28 + 0.24 m over a 0.15 m pedestal give a ~0.6 m planar reach,
# enough to sweep the four spheres that live at radius 0.1–0.45 m.
LINK1, LINK2, PEDESTAL = 0.28, 0.24, 0.15
GRIP_SPAN = 0.028                                  # finger travel (m)

# Object (colour) slots: fixed start positions + an independent Lissajous goal
# per sphere, so every one of the 12 object features is genuinely live.
OBJ_START = ((0.30, 0.12), (-0.27, 0.25), (0.11, -0.34), (-0.33, -0.17))
OBJ_RGBA = ("1.0 0.25 0.25 1", "0.25 1.0 0.35 1",
            "0.30 0.45 1.0 1", "1.0 0.95 0.25 1")
#            red                green            blue              yellow
OBJ_LISSAJOUS = (  # (amp_x, amp_y, freq_x, freq_y, phase)
    (0.10, 0.08, 0.19, 0.13, 0.0),
    (0.08, 0.11, 0.14, 0.21, 1.7),
    (0.11, 0.07, 0.23, 0.11, 3.1),
    (0.07, 0.09, 0.17, 0.19, 4.6),
)

# Task-space PD gains for the action→torque map (N per m/s, N/m).
KP_TASK, KD_TASK = 90.0, 14.0                      # XY velocity tracking
K_Z, KD_Z = 220.0, 22.0                            # Z hold (no sagging)
HOLD_KP, HOLD_KD = 30.0, 3.0                       # joint-space hold (grasp)
BRAKE_KD = 6.0                                     # viscous stop
OBJ_KP, OBJ_KD = 28.0, 9.0                         # sphere goal tracking


def build_scene_xml(cfg: argparse.Namespace) -> str:
    """Generate the whole MJCF scene as a string (self-contained, no files)."""
    dt = 1.0 / cfg.physics_hz
    objs = "\n".join(
        f'    <body name="obj{i}" pos="{x:.3f} {y:.3f} {cfg.obj_radius:.3f}">\n'
        f'      <freejoint name="obj{i}_free"/>\n'
        f'      <geom name="obj{i}_geom" type="sphere" size="{cfg.obj_radius}" '
        f'mass="{cfg.obj_mass}" condim="3" friction="1.0 0.01 0.002" '
        f'rgba="{OBJ_RGBA[i]}"/>\n'
        f'    </body>'
        for i, (x, y) in enumerate(OBJ_START))
    return f"""<mujoco model="loop_arm">
  <!-- Procedurally generated by mujoco_env_bridge.py — there is no .xml file
       on disk: the string is handed straight to MjModel.from_xml_string. -->
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{dt:.6f}" integrator="implicitfast" cone="elliptic"
          impratio="10" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{cfg.render_width}" offheight="{cfg.render_height}"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35"/>
  </visual>
  <default>
    <joint type="hinge" armature="0.02" damping="0.35" limited="true"/>
    <geom type="capsule" size="0.025" rgba="0.75 0.76 0.80 1" contype="1"
          conaffinity="1"/>
    <motor ctrlrange="-40 40" gear="1"/>
  </default>
  <worldbody>
    <light pos="0.4 0.4 1.8" dir="-0.3 -0.3 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.1" rgba="0.20 0.21 0.25 1"
          condim="3" friction="1.0 0.01 0.002"/>
    <body name="base" pos="0 0 0.05">
      <geom name="pedestal" type="cylinder" size="0.075 0.05" mass="4"
            rgba="0.28 0.29 0.33 1"/>
      <joint name="yaw" axis="0 0 1" range="-180 180" damping="0.6"/>
      <geom name="link0" type="capsule" fromto="0 0 0 0 0 {PEDESTAL:.3f}"
            size="0.028" contype="0" conaffinity="0"/>
      <body name="upper" pos="0 0 {PEDESTAL:.3f}">
        <joint name="shoulder" axis="0 1 0" range="-10 125" damping="0.5"/>
        <geom name="link1" type="capsule" fromto="0 0 0 0 0 {LINK1:.3f}"
              size="0.026" mass="0.9"/>
        <body name="fore" pos="0 0 {LINK1:.3f}">
          <joint name="elbow" axis="0 1 0" range="-150 5" damping="0.4"/>
          <geom name="link2" type="capsule" fromto="0 0 0 0 0 {LINK2:.3f}"
                size="0.022" mass="0.6"/>
          <body name="wrist" pos="0 0 {LINK2:.3f}">
            <geom name="palm" type="box" size="0.022 0.032 0.018" mass="0.15"
                  pos="0 0 0.012"/>
            <site name="tool" pos="0 0 0.035" size="0.012"
                  rgba="1 0.4 0.1 1"/>
            <body name="finger_l" pos="0 0.024 0.032">
              <joint name="grip_l" type="slide" axis="0 -1 0"
                     range="0 {GRIP_SPAN:.4f}" damping="2" armature="0.001"/>
              <geom name="fl" type="box" size="0.007 0.011 0.026" mass="0.02"
                    pos="0 0 0.02" contype="0" conaffinity="0"/>
            </body>
            <body name="finger_r" pos="0 -0.024 0.032">
              <joint name="grip_r" type="slide" axis="0 1 0"
                     range="0 {GRIP_SPAN:.4f}" damping="2" armature="0.001"/>
              <geom name="fr" type="box" size="0.007 0.011 0.026" mass="0.02"
                    pos="0 0 0.02" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
{objs}
  </worldbody>
  <tendon>
    <!-- coef=+1 on both finger slides -> one scalar actuator closes the jaws -->
    <fixed name="grip"><joint joint="grip_l" coef="1"/><joint joint="grip_r" coef="1"/></fixed>
  </tendon>
  <actuator>
    <motor name="a_yaw" joint="yaw" ctrlrange="-40 40"/>
    <motor name="a_shoulder" joint="shoulder" ctrlrange="-40 40"/>
    <motor name="a_elbow" joint="elbow" ctrlrange="-40 40"/>
    <motor name="a_grip" tendon="grip" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""


# =============================================================================
# 2b · MOTOR BINDING — how a *decision* becomes a torque
# =============================================================================
# The policy expresses its decision as typed answers; this is where they meet
# the actuators (the expert/harness still speaks the 6-way token table):
#
#   JSON answers  (typed questions)   ─┐
#                                      ├─► Command ─► τ = Jpᵀ·(Kp(v*−v)+…) ─► mj_step
#   expert token  (harness only)      ─┘
#
# Control is always the JSON — the Jev-as-Policy split, where the model decides
# *semantically* (which way, how fast, grasp or hold) and the harness owns the
# geometry.  The binding is deliberately dumb and total: every combination of
# answers maps to a command, an abstained field falls back to "keep doing what
# you were doing", and a questionnaire that says "stay/stay" brakes rather than
# guessing.
@dataclass
class Command:
    """One motor intent, independent of how the policy expressed it."""
    v_dir: Tuple[float, float] = (0.0, 0.0)   # task-space XY direction (unit)
    speed: float = 1.0                        # multiplier on `--ee-speed`
    grip: Optional[float] = None              # -1 open .. +1 close · None = keep
    hold: bool = False                        # joint-space hold (pose latch)
    brake: bool = False                       # viscous stop
    z_dir: float = 0.0                        # -1 descend · 0 hold · +1 climb
    action: int = 5                           # closest ACTION_NAMES entry (telemetry)
    note: str = ""                            # what the answers said (logging)


SPEED_SCALE = (0.5, 1.0, 1.5)                 # slow / medium / fast, when a level runs


def command_from_action(action: int) -> Command:
    """The 6-token table, unchanged: 0-3 nudge, 4 grasp+hold, 5 brake."""
    action = int(action)
    if action in (0, 1, 2, 3):
        vec = tlr.ACTION_VECS[action].numpy()
        return Command(v_dir=(float(vec[0]), float(vec[1])), speed=1.0,
                       grip=-1.0, action=action)
    if action == 4:
        return Command(grip=+1.0, hold=True, action=4)
    return Command(grip=-0.3, brake=True, action=5)


def _answer(doc: Dict[str, Any], qid: str) -> Dict[str, Any]:
    e = doc.get(qid)
    return e if isinstance(e, dict) and not e.get("abstained") else {}


def command_from_json(doc: Dict[str, Any], cfg: Optional[argparse.Namespace] = None
                      ) -> Command:
    """
    Bind a decision document to actuators.  Reads only the typed fields — never
    the free text — so an external harness could replace this function wholesale
    (that is the point of emitting labels instead of coordinates).

        axis_x/axis_y : negative | stay | positive   -> task-space direction
        speed         : a level (0..2) or its probability-weighted value -> scale
        gripper       : open | close | stay          -> jaw
        intent        : approach | grasp | hold      -> move / latch / brake
        height        : down | stay | up             -> the height servo's setpoint
    """
    ax = str(_answer(doc, "axis_x").get("choice", "stay"))
    ay = str(_answer(doc, "axis_y").get("choice", "stay"))
    hz = str(_answer(doc, "height").get("choice", "stay"))
    z_dir = {"down": -1.0, "up": +1.0}.get(hz, 0.0)
    vx = {"negative": -1.0, "positive": +1.0}.get(ax, 0.0)
    vy = {"negative": -1.0, "positive": +1.0}.get(ay, 0.0)
    norm = float(np.hypot(vx, vy))
    if norm > 0.0:                                   # diagonals: unit, not 1.41
        vx, vy = vx / norm, vy / norm

    sp = _answer(doc, "speed")
    raw = sp.get("score")
    if isinstance(raw, int):                         # a level was executed
        scale = SPEED_SCALE[int(np.clip(raw, 0, len(SPEED_SCALE) - 1))]
    elif raw is None:                                # abstained -> default
        scale = 1.0
    else:                                            # expectation (0..2)
        scale = float(np.clip(0.5 + 0.5 * float(raw), 0.4, 1.6))

    grip_lbl = str(_answer(doc, "gripper").get("choice", "stay"))
    intent = str(_answer(doc, "intent").get("choice", "approach"))
    hold = intent == "grasp"
    grip = {"open": -1.0, "close": +1.0}.get(grip_lbl, None)   # None = keep
    if hold:
        grip = +1.0
    brake = (intent == "hold") or (norm == 0.0 and not hold)

    if hold:
        action = 4
    elif brake:
        action = 5
    elif abs(vx) >= abs(vy):
        action = 1 if vx > 0 else 0
    else:
        action = 3 if vy > 0 else 2

    parts = [f"{ax[0]}{ax[-1]}" if ax != "stay" else "x0",
             f"{ay[0]}{ay[-1]}" if ay != "stay" else "y0",
             f"z{hz[0]}" if hz != "stay" else "z0",
             f"spd{scale:.2f}", intent, grip_lbl]
    return Command(v_dir=(vx, vy), speed=scale, grip=grip, hold=hold, brake=brake,
                   z_dir=z_dir, action=action, note="·".join(parts))


def unstuck_guard(env: "MujocoArm", cmd: Command, goal_m: float,
                  prev_goal: Optional[float], stuck: int, rescue: int,
                  cfg: argparse.Namespace) -> Tuple[Command, int, int]:
    """
    Break the deadlock a typed-answer policy can fall into.

    Both axis questions answer `stay` -> the binding brakes -> the arm stops ->
    `vel ≈ 0` and `prev_action = brake` both vote `stay` again.  Measured on
    `robot_live_json.pt`: `goal` frozen at 0.077 m for 50 consecutive frames
    while every decision read `x0·y0·spd0.51·approach·open`, i.e. the policy
    had no way back out of its own stop.

    This is a HARNESS recovery, not a model output — the same split
    Jev-as-Policy uses (the model classifies, the harness owns geometry, limits
    and the e-stop).  After `--unstuck N` frames of brake with no progress the
    harness drives toward the target itself, and stops helping the moment the
    answers command motion again (or after `--unstuck-rescue` frames, whichever
    comes first).  The model still decides everything else; the guard only
    refuses to stand still forever.

    Returns (cmd, stuck, rescue) — pass both counters back in next frame.
    """
    braking = (not cmd.hold) and (cmd.brake or cmd.v_dir == (0.0, 0.0))
    if not braking:
        return cmd, 0, 0                       # the answers are moving again
    if prev_goal is not None and goal_m > prev_goal - 1e-3:
        stuck += 1                             # brake + no progress = stuck
    else:
        stuck = 0
    if rescue > 0:
        rescue -= 1                            # keep helping until it expires
    elif stuck >= cfg.unstuck:
        rescue, stuck = cfg.unstuck_rescue, 0
    if rescue <= 0:
        return cmd, stuck, rescue
    d = env.object_xyz()[env.target][:2] - env.d.site_xpos[env.tool][:2]
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return cmd, stuck, rescue
    v = d / n
    action = int(np.argmax([float(np.dot(v, av)) for av in tlr.ACTION_VECS[:4]]))
    return (Command(v_dir=(float(v[0]), float(v[1])), speed=1.0, grip=None,
                    action=action, note="harness·unstuck"), stuck, rescue)


class MujocoArm:
    """
    Thin, explicit wrapper around MjModel/MjData: state readout, the 28-D
    observation builder, the action->torque map and the sphere drive.

    Everything here runs on the CPU in float64 (MuJoCo's native precision);
    nothing in this class touches torch.
    """

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
        self.d = mujoco.MjData(self.m)

        def sid(objtype, name: str) -> int:
            i = mujoco.mj_name2id(self.m, objtype, name)
            assert i >= 0, f"missing {name} in the scene"
            return i

        self.tool = sid(mujoco.mjtObj.mjOBJ_SITE, "tool")
        arm_joints = ("yaw", "shoulder", "elbow")
        self.arm_dofs = np.array([self.m.jnt_dofadr[sid(mujoco.mjtObj.mjOBJ_JOINT, j)]
                                  for j in arm_joints])
        self.arm_qpos = np.array([self.m.jnt_qposadr[sid(mujoco.mjtObj.mjOBJ_JOINT, j)]
                                  for j in arm_joints])
        self.grip_dofs = np.array(
            [self.m.jnt_dofadr[sid(mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in ("grip_l", "grip_r")])
        self.grip_qpos = np.array(
            [self.m.jnt_qposadr[sid(mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in ("grip_l", "grip_r")])
        self.obj_dofs = np.array(
            [self.m.jnt_dofadr[sid(mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")]
             for i in range(NUM_OBJECTS)])
        self.obj_qpos = np.array(
            [self.m.jnt_qposadr[sid(mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")]
             for i in range(NUM_OBJECTS)])

        # Jacobians are pre-allocated once (3 x nv each) — never per frame.
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))

        # The spheres are force-driven: give them viscous dof damping so the PD
        # goal tracker stays critically damped and they do not spin up.
        for dof in self.obj_dofs:
            self.m.dof_damping[dof:dof + 6] = 1.2
        self.m.dof_damping[self.grip_dofs] = 2.0

        # Sub-steps per policy step: physics stays smooth while the 0.5B policy
        # thinks at a few Hz.
        self.substeps = max(1, int(round(cfg.physics_hz / cfg.control_hz)))
        self.dt = self.substeps / cfg.physics_hz

        # Torque limits straight from the actuator model (applied in command()).
        self.tau_lim = float(self.m.actuator_ctrlrange[0, 1])
        self.grip_lim = float(self.m.actuator_ctrlrange[3, 1])
        # The task is planar (like the training world), so the tool is held at a
        # constant height just above the spheres while the ±x/±y tokens command
        # its world-XY velocity.  Without this the arm would stay upright and
        # never come near an object.
        # Height servo setpoint (m).  It starts at `--ee-height` and is moved by
        # the `height` answer (one step per decision), so z is *commanded* by the
        # policy instead of held constant by the harness.
        self.z_cmd = float(cfg.ee_height)

        self.scale = cfg.obs_scale
        self.reach = TRAIN_REACH_UNITS / cfg.obs_scale       # metres
        self.prev_onehot = np.zeros(tlr.NUM_ACTIONS, dtype=np.float32)
        self.sim_time = 0.0
        self.target = 0                                      # colour slot in play
        self.task = 0                                        # task index (tlr.TASKS)
        self.prev_action = -1
        self.last_action = -1                                # torque latch
        self.last_cmd = command_from_action(5)               # brake until told
        self.last_grip = -self.grip_lim                      # jaws start open
        self.q_hold = np.zeros(3)                            # grasp pose latch

    # ── state readout ───────────────────────────────────────────────────────
    def tool_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tool-site position and linear velocity (v = Jp · q̇)."""
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.tool)
        p = self.d.site_xpos[self.tool].copy()
        v = self.jacp[:, self.arm_dofs] @ self.d.qvel[self.arm_dofs]
        return p, v

    def object_xyz(self) -> np.ndarray:
        """(O,3) centres of the four spheres (from qpos, not from the geoms)."""
        q = self.d.qpos
        return np.stack([q[self.obj_qpos], q[self.obj_qpos + 1],
                         q[self.obj_qpos + 2]], axis=1)

    def object_xy(self) -> np.ndarray:
        """(O,2) xy of the four spheres (from qpos, not from the geoms).

        `qpos[qadr]` alone is only the x of a free joint, so read x and y
        explicitly: qpos[qadr] = x, qpos[qadr+1] = y, then z, then the quat.
        """
        q = self.d.qpos
        return np.stack([q[self.obj_qpos], q[self.obj_qpos + 1]], axis=1)

    def gripper(self) -> float:
        """Finger aperture normalised to 0 (open) .. 1 (closed)."""
        travelled = float(self.d.qpos[self.grip_qpos].sum())
        return float(np.clip(travelled / (2.0 * GRIP_SPAN), 0.0, 1.0))

    # ── the 28-D observation ────────────────────────────────────────────────
    def observe(self) -> Tuple[np.ndarray, float]:
        """
        Build one training-compatible frame:
            [ pos_xy | vel_xy | gripper | tool_z | rel(O*2) | rel_z(O) | dist(O)
              | prev_action(A) ]
        `--obs-scale` converts metres into the world-units the policy saw; the
        height channels are converted into the world's own [0,1] height units.
        Returns (obs (28,) float32, planar distance to the target sphere in m).
        """
        p, v = self.tool_state()
        rel = (self.object_xy() - p[:2]) * self.scale            # (O,2)
        tool_z = (float(p[2]) - Z_FLOOR) / Z_SPAN                # training units
        obj_z = (self.object_xyz()[:, 2] - Z_FLOOR) / Z_SPAN
        obs = np.empty(SENSOR_DIM, dtype=np.float32)
        obs[SL_POS] = p[:2] * self.scale
        obs[SL_VEL] = v[:2] * self.scale
        obs[SL_GRIP] = self.gripper()
        obs[SL_TOOLZ] = tool_z
        obs[SL_REL] = rel.reshape(-1)                            # x0,y0,x1,y1,...
        obs[SL_RELZ] = obj_z - tool_z                            # − = sphere below
        obs[SL_DIST] = np.linalg.norm(rel, axis=1)
        obs[SL_PREV] = self.prev_onehot
        goal = float(np.linalg.norm(rel[self.target]))
        # `goal` stays PLANAR — that is the contract the reward, the eval and the
        # expert labels were all fitted on.  `goal3d` is the honest distance to
        # the sphere centre; the height answer is what closes it.
        self.goal3d = float(np.linalg.norm(
            self.object_xyz()[self.target] - self.d.site_xpos[self.tool]))
        return obs, goal / self.scale

    def note_action(self, action: int) -> None:
        self.prev_onehot = np.zeros(tlr.NUM_ACTIONS, dtype=np.float32)
        self.prev_onehot[action] = 1.0
        self.prev_action = action

    # ── moving spheres: a PD tracker per sphere, applied as a generalised force
    def drive_objects(self) -> None:
        """Lissajous goals in the XY plane -> qfrc_applied on the free joints.
        This is what makes the target *dynamic* rather than teleported."""
        t = self.sim_time
        for i, (ax, ay, fx, fy, ph) in enumerate(OBJ_LISSAJOUS):
            gx = OBJ_START[i][0] + ax * np.sin(2 * np.pi * fx * t + ph)
            gy = OBJ_START[i][1] + ay * np.sin(2 * np.pi * fy * t + ph * 1.7)
            # NB: a free joint has 7 qpos (xyz + quat) but 6 dofs — the two
            # address arrays are NOT interchangeable.
            dof, qadr = self.obj_dofs[i], self.obj_qpos[i]
            p = self.d.qpos[qadr:qadr + 3]
            v = self.d.qvel[dof:dof + 3]
            g = np.array([gx, gy, self.cfg.obj_radius])
            self.d.qfrc_applied[dof:dof + 3] = self.cfg.obj_mass * (
                OBJ_KP * (g - p) - OBJ_KD * v)
            # kill residual spin so the spheres slide, not twirl
            self.d.qfrc_applied[dof + 3:dof + 6] = -0.05 * self.d.qvel[dof + 3:dof + 6]

    # ── ACTION -> TORQUE ───────────────────────────────────────────────────
    def command(self, action: int) -> np.ndarray:
        """
        Map one discrete policy token onto actuator commands, then write
        `data.ctrl`.  Tokens (see tlr.ACTION_NAMES / ACTION_VECS):
            0 -x   1 +x   2 -y   3 +y   : task-space velocity command,
                                           τ = Jpᵀ·(Kp·(v* - v) + Z-hold) + bias
            4 grasp                      : close the jaws + hold the joint pose
            5 brake                      : viscous stop on every joint
        Gravity/Coriolis compensation (`qfrc_bias`) is always added, so the arm
        does not sag while the policy ponders.

        This is one of two ways in; `set_command()` is the other (the JSON
        questionnaire).  Both latch a `Command` and `step()` re-applies it every
        physics sub-step, so the PD closes at `--physics-hz` (500 Hz) while the
        policy still decides at `--control-hz` (5 Hz).
        """
        return self.set_command(command_from_action(action))

    def set_command(self, cmd: Command) -> np.ndarray:
        """Latch a `Command` (from a token or from JSON answers) and apply it.

        One control step carries at most one height answer, so the servo
        setpoint moves here — once per decision — rather than inside `apply()`,
        which runs at `--physics-hz` (100× per decision at the defaults).
        """
        self.last_cmd = cmd
        self.last_action = int(cmd.action)
        if cmd.z_dir:
            self.z_cmd = float(np.clip(self.z_cmd + cmd.z_dir * HEIGHT_STEP_M,
                                       Z_MIN, Z_MAX))
        return self.apply(cmd)

    def apply(self, cmd: Command) -> np.ndarray:
        """`Command` -> actuator torques.  Pure kinematics, no policy logic."""
        cfg = self.cfg
        dq = self.d.qvel[self.arm_dofs]
        p, v = self.tool_state()                                 # refreshes jacp
        Jp = self.jacp[:, self.arm_dofs]                         # (3,3) copy
        grip = self.last_grip if cmd.grip is None else float(cmd.grip) * self.grip_lim

        if cmd.hold:                                             # grasp & hold
            q = self.d.qpos[self.arm_qpos]
            tau = (HOLD_KP * (self.q_hold - q) - HOLD_KD * dq
                   + self.d.qfrc_bias[self.arm_dofs])
        elif cmd.brake or cmd.v_dir == (0.0, 0.0):               # brake
            tau = -BRAKE_KD * dq + self.d.qfrc_bias[self.arm_dofs]
        else:                                                    # planar nudge
            v_des = np.array([cmd.v_dir[0], cmd.v_dir[1], 0.0]) * cfg.ee_speed * cmd.speed
            F = KP_TASK * (v_des - v)
            F[2] = K_Z * (self.z_cmd - p[2]) - KD_Z * v[2]       # commanded height
            tau = Jp.T @ F + self.d.qfrc_bias[self.arm_dofs]

        self.last_grip = float(np.clip(grip, -self.grip_lim, self.grip_lim))
        self.d.ctrl[:3] = np.clip(tau, -self.tau_lim, self.tau_lim)
        self.d.ctrl[3] = self.last_grip
        return self.d.ctrl

    # ── physics ────────────────────────────────────────────────────────────
    def step(self) -> None:
        """Advance one POLICY step (`--control-hz`), with the torque refreshed
        every physics sub-step.

        Holding one open-loop torque across all `substeps` (the previous
        behaviour: `mj_step(nstep=100)`) lets a saturated command accelerate the
        joints for the whole 0.2 s decision window: measured 35 cm of tool travel
        per 8 cm velocity pulse at 2.7 m/s with 100 % torque saturation, which
        made the task unlearnable for either the expert or the policy.
        """
        if self.last_cmd.hold:
            self.q_hold = self.d.qpos[self.arm_qpos].copy()      # latch pose
        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.drive_objects()
            self.apply(self.last_cmd)           # 500 Hz inner servo loop
            mujoco.mj_step(self.m, self.d, nstep=1)
            self.sim_time += sub_dt

    def reset(self) -> None:
        """Ready pose: [0, 125, -85] deg — the tool sits at (0.429, 0, 0.250),
        already on the Z-term's reference, with ∂x/∂q = (-0.161, +0.211), i.e.
        real authority along x.  Two poses had to be measured out first: the
        original upright [0, 70, -40] held the tool 0.5 m above the reference so
        the Z term saturated and slammed it, and [0, 125, -40] put the tool at
        the arm's x-extremum where ∂x/∂q = (-0.137, 0.024) — commanding ±x there
        moved nothing (measured |dq| = 0, 0.00 cm/frame for 15 straight frames).
        qpos is radians even though the MJCF declares its joint ranges in
        degrees."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_qpos] = np.deg2rad([0.0, 125.0, -85.0])
        mujoco.mj_forward(self.m, self.d)
        self.sim_time = 0.0
        self.prev_onehot = np.zeros(tlr.NUM_ACTIONS, dtype=np.float32)
        self.prev_action = -1
        self.last_action = -1
        self.last_cmd = command_from_action(5)               # brake until told
        self.last_grip = -self.grip_lim
        self.goal3d = 0.0          # observe() fills this: 3-D distance to the sphere
        self.q_hold = self.d.qpos[self.arm_qpos].copy()

    def new_instruction(self, rng: np.random.Generator,
                        forced: str = "any", task: str = "any") -> Tuple[str, int]:
        """Pick the task + colour the policy must execute (mirrors SensorWorld).

        `forced` pins the colour instead of sampling it: a run on one goal has a
        coherent return (the reward is distance to *that* goal), and it makes two
        runs comparable — the per-goal ceiling on this arm is NOT the same
        (red 97.5 %, green 0 % measured on the same protocol), because the
        workspace annulus and the ready pose make some approach paths harder.
        `task` pins the verb (`any` samples it); the verb changes what the arm
        should do once it arrives — brake, hold, carry off, or drive through."""
        if forced != "any":
            self.target = tlr.COLORS.index(forced)
        else:
            self.target = int(rng.integers(0, NUM_OBJECTS))
        self.task = (int(rng.integers(0, len(tlr.TASKS))) if task == "any"
                     else tlr.TASKS.index(task))
        text = str(rng.choice(tlr.TASK_TEMPLATES[self.task])).format(
            c=tlr.COLORS[self.target])
        return text, self.target


# =============================================================================
# 3 · HOST -> DEVICE STAGING  (pinned, non-blocking, zero per-frame allocation)
# =============================================================================
def state_sentence(obs: np.ndarray) -> str:
    """
    The harness frame as the model's TEXT state — the same renderer the trainer
    used (`tlr.render_state`), fed the same quantities in the same units:

        tool +0.41 +0.88 +0.77 vel -0.02 +0.01 grip 0.00 obj red +0.12 -0.44 -0.72 …

    Fixed width, so one control step always tokenizes to the same number of
    tokens and the answer markers sit at a computable offset (`StateWindow`).
    The vector slices are already in training units (xy scaled by --obs-scale,
    `tool_z`/`rel_z` in the world's [0,1] height units), so no conversion here.
    """
    xy = obs[SL_REL].reshape(NUM_OBJECTS, 2)
    dz = obs[SL_RELZ].reshape(NUM_OBJECTS, 1)
    rel = np.concatenate([xy, dz], axis=1)
    pos = obs[SL_POS]
    vel = obs[SL_VEL]
    return tlr.render_state(
        (float(pos[0]), float(pos[1]), float(np.ravel(obs[SL_TOOLZ])[0])),
        (float(vel[0]), float(vel[1])), float(np.ravel(obs[SL_GRIP])[0]), rel)


class StateWindow:
    """
    Rolling window of the last `steps` state sentences, oldest first:

        sentence --tok.encode--> ids (L) -> state_ids (1, S·L) + ans_pos (1, S, Q)

    `ans_pos` is the token index of each question's answer marker inside the
    flattened window — the position whose next-token distribution IS the answer
    (Jev's mechanism: nothing is generated).  It is recomputed per frame from the
    CURRENT questionnaire, so switching schemas (`--questions FILE`) re-points
    the read-out without touching the weights.

    One CPU→device copy per frame (a few hundred int64), no pinned float buffer:
    the state is text, so there is nothing to stage.
    """

    def __init__(self, steps: int, tok, qbind, device: torch.device):
        self.steps, self.tok, self.qbind, self.device = steps, tok, qbind, device
        self.suffix = qbind.step_suffix() if qbind is not None else ""
        self.q = len(qbind) if qbind is not None else 1
        self.texts: List[str] = []
        self.L = 0
        self._probe()          # L and q are known before the first live frame

    def _probe(self) -> None:
        """Resolve the per-frame geometry without consuming a real state."""
        keep = self.texts
        self.texts = [tlr.render_state((0.0, 0.0, 0.0), (0.0, 0.0), 0.0,
                                       [(0.0, 0.0, 0.0)] * len(tlr.COLORS))]
        self._encode()
        self.texts = keep

    def restore(self, texts: Sequence[str]) -> None:
        """Put a snapshotted window back — eval probes must not disturb training."""
        self.texts = list(texts)
        if self.texts:
            self._encode()

    def reset(self, first: str) -> None:
        """First-frame history: repeat the sentence (no past exists yet)."""
        self.texts = [first] * self.steps
        self._encode()

    def _encode(self) -> None:
        rows = [self.tok.encode(f"{t}{self.suffix} {tlr.STATE_MARKER}",
                                add_special_tokens=False) for t in self.texts]
        lens = {len(r) for r in rows}
        if len(lens) != 1:
            raise SystemExit(
                f"state sentences must tokenize to a constant length, got {sorted(lens)}"
                f" — `render_state` is fixed-width, so this tokenizer is splitting the "
                f"numbers differently and the answer positions cannot be computed.")
        self.L = lens.pop()
        flat = torch.tensor([i for r in rows for i in r], dtype=torch.long)
        self.sids = flat.unsqueeze(0).to(self.device)                    # (1, S·L)
        base = torch.arange(self.steps).unsqueeze(1) * self.L + (
            self.L - 1 - self.q + torch.arange(self.q))
        self.apos = base.unsqueeze(0).to(self.device)                    # (1, S, Q)

    def push(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.texts:
            self.reset(text)
            return self.sids, self.apos
        self.texts = self.texts[1:] + [text]
        self._encode()
        return self.sids, self.apos

    def note(self) -> str:
        return (f"text state · {self.steps}×{self.L} = {self.steps * self.L} tokens "
                f"· {self.q} answer marker(s)/step")


# =============================================================================
# 4 · POLICY RUNNER  (model build = exactly what train_loop_robot.main() does)
# =============================================================================
def _flags_present(argv: Sequence[str]) -> set:
    """Set of `--flag` spellings the user actually typed (override detection)."""
    out = set()
    for a in argv[1:]:
        if a.startswith("--"):
            out.add(a.split("=", 1)[0])
    return out


ARCH_KEYS = ("recurrent_layers", "fusion_heads", "loops", "window", "questions",
             "cycle_tag")


def load_checkpoint(cfg: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Load the adapter payload and adopt its architecture (state_dict shapes
    must match, so the checkpoint wins unless the user forced a value)."""
    if not cfg.checkpoint:
        return None
    path = cfg.checkpoint
    if not os.path.isfile(path):
        raise SystemExit(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "trainable_state" not in payload:
        raise SystemExit(f"{path}: not a checkpoint produced by train_loop_robot.py")

    given = _flags_present(sys.argv)
    for key, val in (payload.get("args") or {}).items():
        if key not in ARCH_KEYS + ("model", "trainable"):
            continue
        if getattr(cfg, key, None) == val:
            continue
        flag = "--" + key.replace("_", "-")
        if flag in given:
            print(f"  WARNING     : {flag} overrides the checkpoint value "
                  f"{key}={val!r} — the adapter state may not load cleanly")
        else:
            setattr(cfg, key, val)
            print(f"  adopted     : {key}={val!r} from {os.path.basename(path)}")
    return payload


def checkpoint_bank(cfg: argparse.Namespace, payload: Optional[Dict[str, Any]]):
    """The DEFAULT question bank the policy was trained with: the checkpoint's
    `interface` record (train_loop_robot.interface_record — survives a
    `--questions path.json` that does not exist on this machine) unless
    `--questions` is given.  Checkpoints written before the record existed
    keep the `--questions` / adopted-args path unchanged."""
    if payload is not None and "--questions" not in _flags_present(sys.argv):
        bank = getattr(tlr, "bank_from_payload", lambda _p: None)(payload)
        if bank is not None:
            return bank
    return tlr.resolve_questions(str(getattr(cfg, "questions", "auto")))


def encode_instruction(tok, text: str, device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same padding contract as `train_loop_robot.collate`, batch of one."""
    ids = tok.encode(text, add_special_tokens=False)
    ids = list(ids) or [int(getattr(tok, "eos_token_id", 1) or 1)]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    return ids_t, torch.ones_like(ids_t, dtype=torch.bool)


class PolicyRunner:
    """
    Owns the model, the instruction tokens and — critically — the latent carry.
    `act()` never resets that carry: the hidden state flows from physics frame
    to physics frame exactly as it flowed from window to window in training.
    """

    def __init__(self, cfg: argparse.Namespace, accel, tok, trunk,
                 payload: Optional[Dict[str, Any]] = None):
        self.cfg, self.accel, self.tok = cfg, accel, tok

        # The JSON decision interface IS the policy now: there is no action
        # head, the motor command is decoded from the typed answers
        # (`tlr.json_action`).  The read-out has no per-question parameters
        # (LM head on the label tokens), so a checkpoint that predates the
        # interface still loads — its answers are simply untrained — and we say
        # so loudly instead of switching the policy off.
        if (payload is not None and "questions" not in (payload.get("args") or {})
                and "--questions" not in _flags_present(sys.argv)):
            print("  WARNING     : checkpoint predates the JSON interface — its "
                  "action head is dropped and the typed answers are UNTRAINED; "
                  "retrain before trusting this policy")
        self.bank = checkpoint_bank(cfg, payload)
        if self.bank is None or not tlr.has_json_action(self.bank):
            raise SystemExit(
                "control is always the JSON answers (the action head was "
                "removed): pass a question bank that contains at least one of "
                f"{tlr.JSON_ACTION_QIDS} (e.g. --questions auto), not "
                f"--questions {getattr(cfg, 'questions', None)!r}")
        # The questionnaire travels as PROMPT TEXT (Jev-faithful): the binding
        # renders the schema with Jev labels (A/B/C, 0/1/2) and gives markers
        # and labels their token ids, so a different schema is answered by the
        # same weights.
        self.qbind = tlr.QuestionBinding(self.bank, self.tok)
        # A checkpoint trained with --untie-lm-head carries `lm_head.*` tensors;
        # the architecture has to match before the state dict is loaded or the
        # load would report them as unexpected keys.
        self.untied = bool(payload is not None and any(
            str(k).startswith("lm_head.") for k in payload["trainable_state"]))
        # Same argument for LoRA: a checkpoint trained with --lora-rank carries
        # `<proj>.lora_a/.lora_b` tensors, and they only take effect if those
        # modules exist BEFORE the state dict is loaded — `strict=False` would
        # otherwise drop the entire adaptation without a word.  The rank is read
        # back from the factor's own shape, alpha from the saved args.
        _sd = (payload or {}).get("trainable_state", {})
        _keys = [k for k in _sd if ".lora_a" in str(k)]
        self.lora_rank = (int(_sd[_keys[0]].shape[0]) if _keys
                          else int(getattr(cfg, "lora_rank", 0)))
        _alpha = float(((payload or {}).get("args") or {}).get(
            "lora_alpha", getattr(cfg, "lora_alpha", 16.0)))
        if _keys:
            print(f"  lora        : checkpoint carries rank-{self.lora_rank} "
                  f"factors on {len(_keys)} projections (alpha={_alpha:g})")

        # The cycle tag is part of the architecture too: a `sinusoidal`
        # checkpoint has NO `cycle_emb` tensor, so rebuilding it as `learned`
        # would silently drop the tag (strict=False) and change every cycle.
        self.cycle_tag = str(getattr(cfg, "cycle_tag", "learned"))
        if self.cycle_tag != "learned":
            print(f"  cycle tag   : {self.cycle_tag} (unbounded loop — "
                  f"ceiling {cfg.loops} cycles)")
        model = tlr.LoopedACTTransformer(
            trunk, num_looped_layers=cfg.recurrent_layers, max_loops=cfg.loops,
            cycle_tag=self.cycle_tag,
            pond_tau=cfg.tau, fusion_heads=cfg.fusion_heads,
            grad_checkpoint=False,  # inference: never
            qbind=self.qbind, untie_lm_head=self.untied,
            lora_rank=self.lora_rank, lora_alpha=_alpha,
        ).to(accel.device)
        tlr.configure_trainable(model, cfg.trainable)

        if payload is not None:
            if int(payload.get("sensor_dim", SENSOR_DIM)) != SENSOR_DIM:
                raise SystemExit("checkpoint sensor_dim != SENSOR_DIM — wrong policy")
            names = payload.get("action_names")
            if names and tuple(names) != tuple(tlr.ACTION_NAMES):
                print(f"  WARNING     : checkpoint action order {tuple(names)} "
                      f"!= {tlr.ACTION_NAMES}; torque mapping may be permuted")
            # strict=False with the keys of removed modules (tlr.REMOVED_MODULES)
            # dropped and reported; warns when the saved bank or
            # Jev labels differ from the ones this run renders
            missing, _unexpected = tlr.load_trainable_state(
                model, payload, self.qbind, where="bridge")
            print(f"  adapter     : {len(payload['trainable_state'])} tensors "
                  f"in checkpoint ({len(missing)} model tensors left at init)")
        else:
            print("  adapter     : NOT loaded (random heads) — pass --checkpoint "
                  "for a trained policy")

        model.eval()
        self.model = model
        print(f"  json        : {len(self.bank)} question(s) "
              f"[{' '.join(s.qid for s in self.bank.specs)}] · "
              f"{self.bank.total} option logits · markers "
              f"<{' '.join(self.qbind.markers)}> · schema "
              f"{len(self.qbind.schema)} chars in the prompt")
        self.carry: Optional[torch.Tensor] = None
        self.instruction = "reach the red block"
        self.say: Optional[str] = None               # last COMPLETED utterance (None = silent)
        self.speak_max = int(getattr(cfg, "speak_max", 8))   # tokens incl. END, 0 = mute
        self.speak_tokens = max(1, int(getattr(cfg, "speak_tokens", 1)))   # K per frame
        self.spoken: List[int] = []                  # utterance streamed so far
        self.end_id = tlr.speech_end_id(tok)
        self._encode_prompt()
        self.frames = 0
        self.last_info: Dict[str, Any] = {}

    def _encode_prompt(self) -> None:
        """Prompt = instruction + "\\n" + SCHEMA — the same string `collate`
        builds during training, so the live prompt is byte-identical to the one
        the weights were fitted on.  Speech is NOT part of the prompt: it is
        generated after the state (see `act`)."""
        text = f"{self.instruction}\n{self.qbind.schema}"
        self.instr_ids, self.instr_mask = encode_instruction(
            self.tok, text, self.accel.device)

    def decode_speech(self, ids: List[int]) -> Optional[str]:
        """Token ids -> sentence; None for an empty utterance (silent)."""
        if not ids:
            return None
        dec = getattr(self.tok, "decode", None)
        if dec is None:                       # HashTokenizer has no inverse
            return " ".join(str(i) for i in ids)
        return str(dec(ids, skip_special_tokens=True)).strip() or None

    def set_instruction(self, text: str) -> None:
        """Re-point the policy at another task/colour. The latent carry survives —
        changing the goal is not a reset (training re-samples instructions per
        episode while the carry keeps flowing)."""
        self.instruction = text
        self._encode_prompt()

    def speech_input(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """The spoken prefix as the frame forward's speech input (None if empty)."""
        if self.speak_max <= 0:
            return None, None
        return tlr.speech_prefix(self.spoken, self.accel.device)

    @torch.no_grad()
    def speak_after(self, out, sids: torch.Tensor, apos: torch.Tensor,
                    carry_in: Optional[torch.Tensor]
                    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        STREAMED speech for this frame, from the forward `out` that produced
        the answers and was given `self.spoken` as its speech input.

        The first new token is the LM head at that forward's LAST position, so
        a silent frame costs nothing; with `--speak-tokens 2` and a non-END
        first token exactly ONE extra forward gives the second
        (`tlr.speak_frame`).  END — or reaching `--speak-max` tokens incl. END,
        which forces it — completes the utterance and resets `self.spoken`.
        Returns (full sentence on the completing frame else None, text so far,
        extra forwards ∈ {0,1}).  `--speak-max 0` mutes.
        """
        if self.speak_max <= 0:
            self.spoken = []
            return None, None, 0
        with torch.amp.autocast(device_type=self.accel.amp_device_type,
                                dtype=self.accel.dtype, enabled=self.accel.amp):
            new, done, n = tlr.speak_frame(self.model, out, self.instr_ids,
                                           self.instr_mask, sids, apos, carry_in,
                                           self.spoken, self.speak_tokens,
                                           self.end_id, self.speak_max)
        self.spoken = self.spoken + new
        partial = self.decode_speech(self.spoken)
        say = partial if done else None
        if done:
            self.spoken = []
        return say, partial, n

    @torch.no_grad()
    def act(self, sids: torch.Tensor,
            apos: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
        """One policy step: forward the state window (+ the utterance spoken so
        far, AFTER the state), read the LAST control step's typed answers,
        decode the action from them (`tlr.json_action`) and let the same
        forward emit this frame's speech token(s)."""
        self.frames += 1
        carry_in = self.carry if self.cfg.carry else None
        sp, spm = self.speech_input()
        with torch.amp.autocast(device_type=self.accel.amp_device_type,
                                dtype=self.accel.dtype, enabled=self.accel.amp):
            out = self.model(self.instr_ids, self.instr_mask, sids, apos, carry_in,
                             speech_ids=sp, speech_mask=spm)
        self.carry = out.carry.detach() if self.cfg.carry else None
        action = int(tlr.json_action(out.q_logits)[0, -1])
        # confidence of the decoded action = product of the arg-max
        # probabilities of the questions json_action reads
        p = 1.0
        for qid in tlr.JSON_ACTION_QIDS:
            if qid in out.q_logits:
                p *= float(torch.softmax(out.q_logits[qid][0, -1].float(), -1).max())
        info = {
            "action": action,
            "cycles": float(out.cycles[0, -1]),
            "ponder": float(out.ponder[0, -1]),
            "p": p,
            **{k: v for k, v in out.stats.items() if k != "cycle_usage"},
        }
        if out.task_logits is not None:
            # Free self-check: the head that names the VERB reads the same
            # read-out, so a mismatch against the instruction means the sentence
            # never made it into the latent (see `--task-weight`).
            tl = out.task_logits[0, -1].float()
            info["task_inferred"] = tlr.TASKS[int(tl.argmax())]
            info["task_p"] = float(torch.softmax(tl, dim=-1).max())
        say, partial, n_fwd = self.speak_after(out, sids, apos, carry_in)
        if say is not None:
            self.say = say
        info["say"] = say
        info["say_partial"] = partial
        info["speech_forwards"] = n_fwd
        # answers_json is a RENDERING of distributions the forward already
        # produced (Jev: typed answers); `say` is the only decoded text
        info["json"] = self.model.answers_json(out, say=say)
        self.last_info = info
        return action, info

    def warmup(self, sids: torch.Tensor, apos: torch.Tensor) -> float:
        """First forward pays kernel autotune (MIOpen) — keep it out of the log."""
        t0 = time.perf_counter()
        spoken, say = list(self.spoken), self.say
        self.act(sids, apos)
        self.spoken, self.say = spoken, say     # warmup must not start an utterance
        if self.accel.backend in ("rocm", "cuda"):
            torch.cuda.synchronize(self.accel.device)
        return time.perf_counter() - t0


# =============================================================================
# 5 · MINIMAL PNG WRITER (stdlib only — no Pillow/imageio dependency)
# =============================================================================
def write_png(path: str, rgb: np.ndarray) -> None:
    """8-bit RGB PNG from a uint8 (H,W,3) array, via zlib (filter type 0)."""
    h, w, _ = rgb.shape
    raw = bytearray()
    for row in np.ascontiguousarray(rgb).reshape(h, w * 3):
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(blob)


def free_camera(cfg: argparse.Namespace) -> "mujoco.MjvCamera":
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.30)
    cam.distance, cam.azimuth, cam.elevation = 1.75, 135.0, -20.0
    return cam


def write_decision(path: str, runner: "PolicyRunner", info: Dict[str, Any],
                   goal_m: float, frame: int, env: "MujocoArm",
                   cmd: Optional[Command] = None) -> None:
    """
    Write this frame's decision as JSON — the motor output, produced by the
    heads and rendered, never decoded token by token.  `info["json"]` is the
    model's own output (typed questions + the action derived from them +
    `say`); the wrapper adds only the
    telemetry the harness already owns (goal distance, ACT cycles) and, when the
    questionnaire is what drove the arm, the `Command` it was bound to.  That
    split is deliberate and is the one Jev-as-Policy describes: the model
    classifies, the harness keeps geometry, limits, timing and the e-stop.
    """
    doc: Dict[str, Any] = dict(info.get("json") or {})
    doc["frame"] = frame
    doc["instruction"] = runner.instruction
    doc["task"] = tlr.TASKS[env.task]
    if "task_inferred" in info:
        # what the model thinks it was asked to do, next to what it WAS asked
        doc["task_inferred"] = info["task_inferred"]
        doc["task_inferred_p"] = round(float(info.get("task_p", 0.0)), 3)
    # the model's own sentence this frame (null when it stayed silent) — kept
    # top-level even if a caller built `json` without it
    doc["say"] = info.get("say")
    if cmd is not None:
        doc["executed"] = {
            "action": tlr.ACTION_NAMES[cmd.action],
            "answers": cmd.note,
            "v_dir": [round(float(v), 3) for v in cmd.v_dir],
            "speed": round(float(cmd.speed), 3),
            "gripper": None if cmd.grip is None else round(float(cmd.grip), 3),
            "hold": bool(cmd.hold),
            "brake": bool(cmd.brake),
            "z_dir": round(float(cmd.z_dir), 3),
            "z_cmd_m": round(float(env.z_cmd), 4),
        }
    doc["telemetry"] = {
        "goal_m": round(float(goal_m), 4),
        "goal3d_m": round(float(getattr(env, "goal3d", goal_m)), 4),
        "reach_m": round(float(env.reach), 4),
        "in_reach": bool(goal_m < env.reach),
        "cycles": round(float(info["cycles"]), 3),
        "ponder": round(float(info["ponder"]), 3),
        "latent_norm": round(float(info["latent_norm"]), 3),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)          # a reader never sees a half-written frame


# =============================================================================
# 6 · MAIN BRIDGE LOOP
# =============================================================================
def run_bridge(cfg: argparse.Namespace, accel, tok, trunk,
               payload: Optional[Dict[str, Any]]) -> int:
    rng = np.random.default_rng(cfg.seed)
    env = MujocoArm(cfg)
    env.reset()
    runner = PolicyRunner(cfg, accel, tok, trunk, payload)
    guard = tlr.MemoryGuard(accel, gc_every=cfg.gc_every)
    win = StateWindow(cfg.window, tok, runner.qbind, accel.device)

    frames = int(cfg.duration * cfg.control_hz) if cfg.duration > 0 else int(cfg.frames)
    if frames <= 0:
        raise SystemExit("nothing to do: set --frames or --duration")

    print("=" * 78)
    print("MUJOCO BRIDGE")
    print("=" * 78)
    print(sensor_layout_table())
    print(f"  scene       : {NUM_OBJECTS} force-driven spheres + 3-DoF arm + tendon "
          f"gripper · {env.m.nv} dofs · {env.m.nu} actuators")
    print(f"  control     : {cfg.control_hz:g} Hz policy · {cfg.physics_hz:g} Hz "
          f"physics ({env.substeps} substeps/step) · {frames} frames "
          f"= {frames/cfg.control_hz:.1f}s sim time")
    print(f"  policy      : window S={cfg.window} steps · max_loops={cfg.loops} · "
          f"carry={'on' if cfg.carry else 'OFF (ablation)'} · device="
          f"{accel.name or accel.device}")
    print(f"  state       : {win.note()}")
    # Control is ALWAYS the questionnaire: typed answers -> Command -> torque.
    # The action head is gone; `--control action` is refused with a clear error
    # instead of silently driving the arm from something that no longer exists.
    ctrl = str(getattr(cfg, "control", "json"))
    if ctrl not in ("auto", "json"):
        raise SystemExit(f"--control {ctrl} is no longer supported: the action "
                         "head was removed and the arm is always driven by the "
                         "JSON answers (drop the flag or pass --control json)")
    print("  control     : json (typed answers -> Command -> torque; the "
          "`action` field is derived from them by tlr.json_action)")
    print(f"  speech      : "
          + (f"streamed, {runner.speak_tokens} token(s) per frame, ≤ "
             f"{runner.speak_max} per utterance incl. END, END first = silent"
             if runner.speak_max > 0 else "muted (--speak-max 0)"))
    print(f"  memory      : {accel.note or 'uncapped'} · amp={accel.amp} "
          f"({str(accel.dtype).split('.')[-1]}) · gc-every={cfg.gc_every or 'off'}")

    # optional visualisation -------------------------------------------------
    viewer = None
    if not cfg.headless:
        if _VIEWER_IMPORT_ERROR is not None:
            print(f"  viewer      : unavailable ({_VIEWER_IMPORT_ERROR}) — "
                  f"running headless")
        else:
            try:
                viewer = mujoco.viewer.launch_passive(env.m, env.d)
                with viewer.lock():
                    viewer.cam.lookat[:] = (0.0, 0.0, 0.30)
                    viewer.cam.distance, viewer.cam.azimuth = 1.75, 135.0
                    viewer.cam.elevation = -20.0
                print("  viewer      : passive window open — close it to stop")
            except Exception as exc:                          # pragma: no cover
                print(f"  viewer      : launch failed ({exc}) — running headless")
                viewer = None
    renderer = None
    if cfg.render_out:
        try:
            renderer = mujoco.Renderer(env.m, cfg.render_height, cfg.render_width)
            cam = free_camera(cfg)
            print(f"  render      : {cfg.render_out} "
                  f"({cfg.render_width}x{cfg.render_height}"
                  + (f", every {cfg.render_every} frames" if cfg.render_every else
                     ", final frame only") + ")")
        except Exception as exc:                              # pragma: no cover
            print(f"  render      : offscreen GL unavailable ({exc}) — "
                  f"continuing without PNG output")
            renderer = None

    # first frame: fill the window, then let the policy take over -------------
    obs, goal_m = env.observe()
    win.reset(state_sentence(obs))
    sids, apos = win.push(state_sentence(obs))
    dt_warm = runner.warmup(sids, apos)
    if accel.backend in ("rocm", "cuda"):
        torch.cuda.synchronize(accel.device)
    print(f"  warmup      : first forward {dt_warm*1e3:.0f} ms "
          f"(kernel autotune happens once)")
    runner.set_instruction(env.new_instruction(rng, cfg.goal, cfg.task)[0])
    print(f"  instruction : \"{runner.instruction}\" (task {tlr.TASKS[env.task]}"
          f", colour slot {env.target}"
          + (f", re-sampled every {cfg.retarget_every} frames)"
             if cfg.retarget_every else " for the whole run)"))

    frozen = tlr.freeze_gc()             # model/scene graph out of the GC scan
    print(f"  gc          : {frozen} long-lived objects frozen · explicit "
          f"collect {'every ' + str(cfg.gc_every) + ' frame(s)' if cfg.gc_every else 'off'}"
          f" (a full pass costs ~215-260 ms here)")

    t_start, in_reach, cycles_acc, frames_done = time.time(), 0, 0.0, 0
    peak_gb = 0.0
    cmd: Optional[Command] = None
    prev_goal: Optional[float] = None
    stuck, rescue, rescues = 0, 0, 0
    try:
        while frames_done < frames:
            if viewer is not None and not viewer.is_running():
                print("  viewer      : window closed by user")
                break
            t_loop = time.perf_counter()
            frames_done += 1

            # ── 1 · policy: obs -> pinned -> device -> ACT loop -> token ────
            obs, goal_m = env.observe()
            sids, apos = win.push(state_sentence(obs))
            action, info = runner.act(sids, apos)
            cycles_acc += info["cycles"]
            if info.get("say"):
                print(f"  [{frames_done:5d}] say \"{info['say']}\""
                      f"  ({info.get('speech_forwards', 0)} extra forward(s) "
                      f"this frame)")

            # ── 2 · decision -> torque -> physics ──────────────────────────
            # the questionnaire drives the arm: labels -> Command -> torques
            cmd = command_from_json(info.get("json") or {}, cfg)
            if cfg.unstuck:
                cmd, stuck, rescue = unstuck_guard(env, cmd, goal_m,
                                                   prev_goal, stuck, rescue, cfg)
                if cmd.note.startswith("harness"):
                    rescues += 1
                    if rescues == 1:
                        print(f"  unstuck     : the answers braked for "
                              f"{cfg.unstuck} frames with no progress — "
                              f"harness steering toward the target for "
                              f"{cfg.unstuck_rescue} frame(s)")
            env.set_command(cmd)
            env.note_action(cmd.action)
            env.step()
            prev_goal = goal_m

            # ── 3 · guards + visualisation ────────────────────────────────
            guard.flush_if_low()
            guard.reclaim(frames_done)
            if goal_m < env.reach:
                in_reach += 1
            if viewer is not None and viewer.is_running():
                with viewer.lock():
                    viewer.sync()
            if renderer is not None:
                renderer.update_scene(env.d, camera=cam)
                img = renderer.render()
                if cfg.render_every and frames_done % cfg.render_every == 0:
                    stem, ext = os.path.splitext(cfg.render_out)
                    write_png(f"{stem}_{frames_done:05d}{ext or '.png'}", img)
                write_png(cfg.render_out, img)

            # ── 4 · real-time pacing (optional) ───────────────────────────
            if cfg.realtime:
                slack = (1.0 / cfg.control_hz) - (time.perf_counter() - t_loop)
                if slack > 0:
                    time.sleep(slack)

            # ── 5 · retarget + JSON decision + log ────────────────────────
            if cfg.json_out and frames_done % max(cfg.json_every, 1) == 0:
                write_decision(cfg.json_out, runner, info, goal_m, frames_done, env,
                               cmd)
            if cfg.retarget_every and frames_done % cfg.retarget_every == 0:
                runner.set_instruction(env.new_instruction(rng, cfg.goal, cfg.task)[0])
                print(f"  [{frames_done:5d}] retarget -> \"{runner.instruction}\"")
            if frames_done % cfg.log_every == 0 or frames_done == 1:
                if accel.backend in ("rocm", "cuda"):
                    peak_gb = max(peak_gb,
                                  torch.cuda.max_memory_allocated(accel.device) / 2**30)
                    tail = f" peak={peak_gb:.2f}G"
                else:
                    tail = ""
                dec = cmd.note if cmd is not None else tlr.ACTION_NAMES[action]
                tag = "dec"
                print(f"  [{frames_done:5d}] t={frames_done/cfg.control_hz:6.2f}s "
                      f"{tag}={dec:>22} p={info['p']:.2f} "
                      f"goal={goal_m:5.3f}m{'*' if goal_m < env.reach else ' '} "
                      f"goal3d={env.goal3d:5.3f}m "
                      f"cycles={info['cycles']:.2f} ponder={info['ponder']:.2f} "
                      f"halt={info['halted_frac']:.2f} "
                      f"mass_err={info['mass_err']:.1e} "
                      f"|z|={info['latent_norm']:5.1f} "
                      f"{(time.perf_counter()-t_loop)*1e3:6.1f}ms/f{tail}", flush=True)
            if cfg.max_seconds and time.time() - t_start > cfg.max_seconds:
                print(f"  wall-clock budget --max-seconds={cfg.max_seconds:g} reached")
                break
    except KeyboardInterrupt:
        print("\n  interrupted (Ctrl-C)")
    finally:
        guard.reclaim(frames_done, force=True)
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    wall = time.time() - t_start
    print("-" * 78)
    print(f"  frames      : {frames_done} ({frames_done/cfg.control_hz:.1f}s sim, "
          f"{wall:.1f}s wall → {frames_done/max(wall,1e-9):.2f} Hz policy, "
          f"real-time factor {frames_done/cfg.control_hz/max(wall,1e-9):.2f})")
    print(f"  reaching    : {100.0*in_reach/max(frames_done,1):.1f}% of frames within "
          f"{env.reach*100:.1f} cm of the instructed sphere "
          f"(last goal {goal_m:.3f} m)")
    carry_note = ("threaded across all {} frames — never reset".format(frames_done)
                  if cfg.carry else
                  "DISABLED (--no-carry ablation resets it every frame)")
    print(f"  ACT         : mean hard cycles {cycles_acc/max(frames_done,1):.2f} · "
          f"latent carry {carry_note}")
    if cfg.unstuck:
        print(f"  unstuck     : {rescues} harness-steered frame(s) of {frames_done} "
              f"(the answers braked with the goal out of reach; the guard keeps "
              f"the arm from standing still forever)")
    print(f"  memory      : peak {peak_gb:.2f} GiB · OOM events={guard.oom_events} · "
          f"allocator flushes={guard.reclaims}")
    if renderer is not None:
        print(f"  frames on disk: {cfg.render_out}" +
              (f" + {cfg.render_width}x{cfg.render_height} snapshots every "
               f"{cfg.render_every}" if cfg.render_every else ""))
    return 0


# =============================================================================
# 7 · CLI  (reuse train_loop_robot's parser so the hardware guards stay in sync)
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = tlr.build_parser()
    p.description = ("MuJoCo <-> Looped-Transformer bridge: procedural MJCF arm, "
                     "text state + gated prompt fusion, ACT-driven torque control, "
                     "Jev-style typed answers read out of the LM head.")
    g = p.add_argument_group("mujoco bridge")
    g.add_argument("--checkpoint", default="",
                   help="adapter payload from train_loop_robot.py --out "
                        "(trainable_state only; the trunk is rebuilt from --model)")
    g.add_argument("--frames", type=int, default=500, help="policy steps to run")
    g.add_argument("--duration", type=float, default=0.0,
                   help="simulated seconds (overrides --frames)")
    g.add_argument("--control-hz", type=float, default=5.0,
                   help="policy/actuator rate — the physics runs faster")
    g.add_argument("--physics-hz", type=float, default=500.0)
    g.add_argument("--ee-speed", type=float, default=0.40,
                   help="task-space speed for the ±x/±y tokens (m/s). The expert "
                        "in training moved 0.12 units/step = 0.075 m/step at "
                        "--obs-scale 1.6, i.e. 0.375 m/s at 5 Hz — this is the "
                        "matching figure")
    g.add_argument("--obs-scale", type=float, default=1.6,
                   help="metres -> training world-units (arm reach ≈0.6 m, so 1.6 "
                        "puts the tool inside the trained pos range ±1.2)")
    g.add_argument("--ee-height", type=float, default=0.25,
                   help="tool height held by the Z term (m, absolute). 0.25 is "
                        "the arm's well-conditioned working plane — ∂x/∂q ≈ "
                        "0.26 there — and the ready pose starts on it; the task "
                        "itself is planar (the training world has no z axis), so "
                        "this is the ONLY thing that decides whether the jaws "
                        "meet the spheres: the balls sit at z≈0.035, so pass "
                        "--ee-height 0.05 to actually touch them (measured site "
                        "z 0.085). `goal=` is planar; `goal3d=` is the honest gap")
    g.add_argument("--obj-mass", type=float, default=0.05)
    g.add_argument("--obj-radius", type=float, default=0.035)
    g.add_argument("--retarget-every", type=int, default=200,
                   help="re-sample the instructed colour every N frames (0 = never)")
    g.add_argument("--goal", choices=("any",) + tlr.COLORS, default="any",
                   help="pin the instruction colour instead of re-drawing it "
                        "(default any).  Use it to replay a checkpoint fine-tuned "
                        "on one colour — per-goal difficulty is NOT uniform")
    g.add_argument("--task", choices=("any",) + tlr.TASKS, default="any",
                   help="pin the task VERB in the instruction (default any = "
                        "sampled per retarget).  The verb is the task interface: "
                        "reach brakes on the object, grasp holds it, throw carries "
                        "it off, push drives through — same geometry, same 28-D "
                        "observation, only the sentence changes")
    g.add_argument("--speak-max", type=int, default=8,
                   help="max utterance length in tokens INCLUDING the end token "
                        "(0 = mute); reaching it forces the end.  Speech is "
                        "STREAMED (--speak-tokens per frame): each frame's forward "
                        "carries the spoken prefix after the state and its last "
                        "position gives the next token (END = silent / done, no "
                        "extra cost).  The finished sentence lands in "
                        "info['say'] and the JSON's top-level `say` (null "
                        "otherwise).  With --model tiny the tokenizer is a hash, "
                        "so the 'words' print as ids")
    g.add_argument("--headless", action="store_true",
                   help="no viewer window (required on a headless session)")
    g.add_argument("--realtime", action="store_true",
                   help="sleep to hold --control-hz (only viable on a small trunk)")
    g.add_argument("--max-seconds", type=float, default=0.0,
                   help="wall-clock budget, 0 = unlimited")
    g.add_argument("--render-out", default="",
                   help="PNG path for offscreen frames (proof/screenshots)")
    g.add_argument("--render-every", type=int, default=0,
                   help="also write numbered PNGs every N frames (0 = final only)")
    g.add_argument("--render-width", type=int, default=1280)
    g.add_argument("--render-height", type=int, default=720)
    g.add_argument("--json-out", default="",
                   help="write this frame's decision as JSON (typed questions + "
                        "derived action + say + telemetry) — the motor output as "
                        "a document; atomic replace, so a reader never sees a "
                        "partial frame")
    g.add_argument("--json-every", type=int, default=1,
                   help="write the decision every N frames (1 = every frame)")
    g.add_argument("--control", default="json",
                   help="kept for compatibility: the arm is ALWAYS driven by the "
                        "typed answers bound to a Command (axis_x/axis_y -> "
                        "direction, speed -> scale, gripper/intent -> jaw+hold). "
                        "`json`/`auto` are accepted; `action` is refused with an "
                        "error (the action head was removed).  In the RL loop one "
                        "label is sampled per question, so logπ is the sum of the "
                        "per-question log-probs")
    g.add_argument("--unstuck", type=int, default=8,
                   help="HARNESS recovery for the JSON control: after N consecutive "
                        "frames where the answers brake and the goal distance "
                        "stops improving, the harness steers toward the target "
                        "itself for --unstuck-rescue frames (a policy answering "
                        "`stay` on both axes stops the arm, and a stopped arm "
                        "reads vel≈0 + prev=brake -> `stay` forever). 0 disables")
    g.add_argument("--unstuck-rescue", type=int, default=6,
                   help="frames of harness steering per recovery (see --unstuck)")
    return p


def main() -> int:
    cfg = build_parser().parse_args()
    accel = tlr.verify_hardware(cfg)
    if cfg.self_test:
        # `--self-test` is inherited from the trainer; honour it here instead of
        # silently running the bridge (the architectural invariants are the same
        # ones this script relies on: ACT convexity, no leak, carry, JSON heads)
        return tlr.run_self_test(cfg, accel)
    if str(getattr(cfg, "control", "json")) not in ("auto", "json"):
        raise SystemExit(f"--control {cfg.control} is no longer supported: the "
                         "action head was removed and the arm is always driven "
                         "by the JSON answers (drop the flag or pass --control json)")
    payload = load_checkpoint(cfg)
    tok, trunk = tlr.load_base(cfg.model, accel)
    bank = checkpoint_bank(cfg, payload)
    tlr.report_params(tlr.LoopedACTTransformer(
        trunk, num_looped_layers=cfg.recurrent_layers, max_loops=cfg.loops,
        fusion_heads=cfg.fusion_heads,
        qbind=tlr.QuestionBinding(bank, tok) if bank is not None else None,
        lora_rank=int(getattr(cfg, "lora_rank", 0))), accel)
    return run_bridge(cfg, accel, tok, trunk, payload)


if __name__ == "__main__":
    raise SystemExit(main())
