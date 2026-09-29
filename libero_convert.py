"""
libero_convert.py — turn LeRobot's LIBERO demos into this project's format.

Per frame it emits
    prompt   : the dataset's own task instruction (LIBERO has one per task)
    state    : a fixed-width state sentence (tool x y z, velocity in m/s, grip)
    answers  : the JSON questionnaire labels, derived from the recorded action
    say      : a speech target — an event sentence, progress narration, or null
               (null = silent: the model emits only its end token)

Source: https://huggingface.co/datasets/lerobot/libero (v3.0, 1693 episodes,
40 tasks, 10 fps).  observation.state is 8-d (eef xyz, axis-angle, 2 gripper
finger qpos); action is 7-d (eef delta xyz, delta rot, gripper -1 open/+1 close).
Camera frames live in separate AV1 videos and are NOT read here (vision comes
later); only the small per-frame parquet tables are downloaded.

v2 fixes (from the first 300-episode report):
  * episodes are sampled across ALL tasks (stratified), with a held-out split
  * each grasp is matched to the right object: the instruction is parsed into
    manipulated objects vs destinations, and the k-th real CARRY (held + moved)
    names the k-th object; regrasps keep the current object; missed grasps
    (fingers close on nothing) are silent
  * the state sentence reports velocity in m/s (per-frame deltas rounded to 0)
  * inverse-frequency class weights per question are written to meta.json
  * checks against the RECORDED MOTION (future eef displacement, finger width),
    not against the action the labels were computed from

Runs on a Kaggle CPU session.  Usage:
    python libero_convert.py --episodes 0 --out /kaggle/working/libero_conv   (0 = all)
    python libero_convert.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

REPO = "lerobot/libero"
FPS = 10.0
HELD_W = 0.004        # finger gap (m) above which a closed gripper holds something
CARRY_M = 0.05        # eef travel (m) between grasp and release that counts as a carry
LOOKAHEAD = 3         # frames of future motion used by the motion check

QUESTIONS = {                      # same ids/options as DEFAULT_QUESTIONS
    "axis_x": ("negative", "stay", "positive"),
    "axis_y": ("negative", "stay", "positive"),
    "gripper": ("open", "close", "stay"),
    "intent": ("approach", "grasp", "hold"),
    "height": ("down", "stay", "up"),
    "speed": ("slow", "medium", "fast"),
}

DEST_PREV = {"on", "in", "into", "onto", "to", "of", "at", "inside", "from", "under",
             "over", "between", "behind", "near", "beside", "next"}
STOP = DEST_PREV | {"it", "then", "with"}
VERBS = {"put", "pick", "place", "push", "open", "close", "turn", "stack", "move",
         "lift", "take", "grab", "set"}


# ─────────────────────────────────────────────────────────────────────────────
# instruction parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_objects(instruction: str):
    """-> (manipulated objects in order, destinations).

    'put both the alphabet soup and the cream cheese box in the basket'
        -> (['alphabet soup', 'cream cheese box'], ['basket'])
    'turn on the stove and put the moka pot on it' -> (['stove', 'moka pot'], [])
    """
    w = instruction.lower().strip().rstrip(".").replace(",", " ").split()
    manip, dest = [], []
    i = 0
    while i < len(w):
        if w[i] == "both" and i + 1 < len(w) and w[i + 1] == "the":
            i += 1
            continue
        if w[i] in ("the", "both"):
            j, phrase = i + 1, []
            while j < len(w):
                x = w[j]
                if x == "and":
                    nxt = w[j + 1] if j + 1 < len(w) else ""
                    if nxt in ("the", "it", "") or nxt in VERBS:
                        break
                    phrase.append(x)          # 'yellow and white mug'
                    j += 1
                    continue
                if x in STOP:
                    break
                phrase.append(x)
                j += 1
            if phrase:
                prev = w[i - 1] if i > 0 else ""
                particle = prev in ("on", "off") and i >= 2 and w[i - 2] == "turn"
                (dest if prev in DEST_PREV and not particle else manip).append(" ".join(phrase))
            i = max(j, i + 1)
        else:
            i += 1
    return manip, dest


ING = {"open": "opening", "close": "closing", "turn on": "turning on",
       "turn off": "turning off", "push": "pushing"}


def singular(p: str) -> str:
    w = p.split()
    if w and w[-1].endswith("s") and not w[-1].endswith("ss"):
        w[-1] = w[-1][:-1]
    return " ".join(w)


def parse_steps(instruction: str):
    """Instruction -> ordered steps {kind: carry|interact, verb, obj}.

    'turn on the stove and put the moka pot on it'
        -> [interact turn on stove, carry moka pot]
    'put both moka pots on the stove' -> [carry moka pot, carry moka pot]
    'pick up the black bowl between the plate and the ramekin and place it on the plate'
        -> [carry black bowl]          (only the first object of a single carry)
    'put the yellow and white mug in the microwave and close it'
        -> [carry yellow and white mug, interact close microwave]
    """
    w = instruction.lower().strip().rstrip(".").replace(",", " ").split()
    clauses, cur = [], []
    for i, x in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if x == "and" and nxt in VERBS and cur:
            clauses.append(cur)
            cur = []
            continue
        cur.append(x)
    if cur:
        clauses.append(cur)
    merged = []
    for c in clauses:                   # 'place it in ...' continues the previous carry
        if merged and len(c) > 1 and c[0] in ("place", "put") and c[1] == "it":
            merged[-1] = merged[-1] + ["and"] + c      # keep 'and' so the object phrase ends
        else:
            merged.append(c)
    steps, last_dest = [], None
    for c in merged:
        text = " ".join(c)
        manip, dest = parse_objects(text)
        v = c[0]
        if v == "turn" and len(c) > 1 and c[1] in ("on", "off"):
            steps.append({"kind": "interact", "verb": f"turn {c[1]}", "obj": (manip or ["switch"])[0]})
        elif v in ("open", "close", "push"):
            obj = (manip or [None])[0] if c[1:2] != ["it"] else last_dest
            steps.append({"kind": "interact", "verb": v, "obj": obj or "object"})
        else:
            if "both" in c and len(manip) == 1:
                objs = [singular(manip[0])] * 2
            elif "both" in c:
                objs = manip[:2]
            else:
                objs = manip[:1]
            for o in objs or ["object"]:
                steps.append({"kind": "carry", "verb": "put", "obj": o})
        if dest:
            last_dest = dest[0]
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# labelling (pure python, no I/O — the self-test drives these directly)
# ─────────────────────────────────────────────────────────────────────────────
def sign3(v: float, dead: float) -> str:
    return "negative" if v < -dead else "positive" if v > dead else "stay"


def width(s) -> float:
    return s[6] - s[7]


def grasp_segments(actions, states):
    """Closed-command intervals, each classified as carry / regrasp / missed."""
    segs, start = [], None
    for t, a in enumerate(actions):
        c = a[6] > 0.0
        if c and start is None:
            start = t
        if not c and start is not None:
            segs.append([start, t])
            start = None
    if start is not None:
        segs.append([start, None])
    out = []
    for s0, s1 in segs:
        last = (s1 if s1 is not None else len(states)) - 1
        held = width(states[last]) > HELD_W
        travel = math.dist(states[s0][:3], states[last][:3])
        kind = "carry" if held and travel > CARRY_M else "regrasp" if held else "missed"
        # closing phase: until the fingers stop moving (cap 15 frames)
        settle = s0
        for t in range(s0 + 1, min(last, s0 + 15) + 1):
            settle = t
            if t > s0 + 1 and abs(width(states[t]) - width(states[t - 1])) < 5e-4:
                break
        out.append({"start": s0, "end": s1, "last": last, "kind": kind,
                    "travel": travel, "settle": settle})
    return out


PAST = {"open": "opened", "close": "closed", "turn on": "turned on",
        "turn off": "turned off", "push": "pushed"}
PUSHABLE = ("open", "close", "push")    # can be done by pushing with an OPEN hand


def label_episode(actions, states, instruction: str, dead: float, speed_cuts):
    """-> (per-frame [(answers, say, event)], segments, steps, steps_done).

    Grasp segments are matched to the instruction's steps in order.  A carry step
    is finished by a real carry (held + moved); regrasps/missed grasps keep it.
    An interact step (open/close/turn on/push) is finished by any grasp that
    held something or moved the hand > 2 cm (knobs, handles), and its speech is
    'turning on the stove' / 'turned on the stove' instead of grasp/release.
    """
    steps = parse_steps(instruction) or [{"kind": "carry", "verb": "put", "obj": "object"}]
    segs = grasp_segments(actions, states)
    # Drawers, doors and plates are often pushed with an open hand: no grasp
    # segment exists for them (measured: 0.00 closes/episode on 'open the middle
    # drawer' and 'push the plate').  A pushable step only claims a grasp when
    # there are more usable grasps than the steps that NEED one; otherwise it is
    # done by pushing.  LIBERO demos are all successful, so such a step is
    # counted done — that part is assumed, not measured.
    pushable = [s["kind"] == "interact" and s["verb"] in PUSHABLE for s in steps]
    usable = (sum(sg["kind"] == "carry" for sg in segs)
              + min(sum(sg["kind"] == "regrasp" for sg in segs),
                    sum(s["kind"] == "interact" and not p for s, p in zip(steps, pushable))))
    surplus = usable - sum(not p for p in pushable)
    for s, p in zip(steps, pushable):
        s["by_push"] = p and surplus <= 0
        if p and surplus > 0:
            surplus -= 1
    pushes = []                                   # (step index, after segment index)
    k = 0

    def skip_pushes(after: int) -> None:
        nonlocal k
        while k < len(steps) and steps[k]["by_push"]:
            pushes.append((k, after))
            k += 1

    skip_pushes(-1)
    for i, sg in enumerate(segs):
        st = steps[min(k, len(steps) - 1)]
        sg["step"] = st
        sg["obj"] = st["obj"]
        if st["kind"] == "carry":
            done = sg["kind"] == "carry"
        else:
            done = sg["kind"] != "missed" or sg["travel"] > 0.02
        sg["done"] = done and k < len(steps)
        if sg["done"]:
            k += 1
        sg["next"] = steps[k]["obj"] if (sg["done"] and k < len(steps)) else None
        if sg["done"]:
            skip_pushes(i)
    steps_done = min(k, len(steps))
    # when a pushed step is SAID: the last frame if nothing follows it (demo
    # success), else the frame of the push interval where the hand is farthest
    # from where the interval started (heuristic: the end of the push stroke)
    push_say = {}
    for si, after in pushes:
        a = 0
        if after >= 0:
            e = segs[after]["end"]
            a = len(actions) - 1 if e is None else e + 1
        b = segs[after + 1]["start"] if after + 1 < len(segs) else len(actions)
        a = min(a, len(actions) - 1)
        if b >= len(actions):
            t_say = len(actions) - 1
        else:
            span = range(a, max(a + 1, b))
            t_say = max(span, key=lambda t: math.dist(states[t][:3], states[a][:3]))
        while t_say in push_say and t_say + 1 < len(actions):
            t_say += 1
        st = steps[si]
        push_say[t_say] = (f"{PAST[st['verb']]} the {st['obj']}",
                           steps[si + 1]["obj"] if si + 1 < len(steps) else None)
    by_t = {}
    for sg in segs:
        for t in range(sg["start"], sg["last"] + 1):
            by_t[t] = sg

    out, pending_move, lifted = [], steps[0]["obj"], set()
    for t, a in enumerate(actions):
        sg = by_t.get(t)
        closed = a[6] > 0.0
        prev_closed = t > 0 and actions[t - 1][6] > 0.0
        grip = ("close" if closed and not prev_closed else
                "open" if prev_closed and not closed else "stay")
        if sg is not None and (sg["kind"] == "missed" or t <= sg["settle"]):
            intent = "grasp"
        elif sg is not None:
            intent = "hold"
        else:
            intent = "approach"
        sp = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        ans = {"axis_x": sign3(a[0], dead), "axis_y": sign3(a[1], dead),
               "gripper": grip, "intent": intent,
               "height": {"negative": "down", "stay": "stay",
                          "positive": "up"}[sign3(a[2], dead)],
               "speed": ("slow" if sp < speed_cuts[0] else
                         "medium" if sp < speed_cuts[1] else "fast")}
        say, event = None, None
        if grip == "close" and sg is not None:
            event = sg["kind"]
            st = sg["step"]
            if st["kind"] == "interact":
                if sg["done"]:
                    say = f"{ING[st['verb']]} the {st['obj']}"
            elif sg["kind"] != "missed":
                say = f"grasped the {sg['obj']}"
        elif grip == "open":
            ps = by_t.get(t - 1)
            if ps is not None and ps["step"]["kind"] == "interact":
                if ps["done"]:
                    event = "release"
                    say = f"{PAST[ps['step']['verb']]} the {ps['obj']}"
                    pending_move = ps["next"]
            elif ps is not None and ps["kind"] != "missed":
                event = "release"
                say = f"released the {ps['obj']}"
                if ps["done"]:
                    pending_move = ps["next"]
        elif (sg is not None and sg["step"]["kind"] == "carry" and sg["kind"] != "missed"
              and ans["height"] == "up" and id(sg) not in lifted):
            say = f"lifting the {sg['obj']}"
            lifted.add(id(sg))
        elif (pending_move and sg is None and
              (t == 0 or ans["axis_x"] != "stay" or ans["axis_y"] != "stay"
               or ans["height"] != "stay")):
            say = f"moving to the {pending_move}"
            pending_move = None
        if say is None and t in push_say:
            say, nxt = push_say[t]
            event = "push"
            pending_move = nxt
        out.append((ans, say, event))
    return out, segs, steps, steps_done


def state_sentence(s, prev) -> str:
    """Fixed-width text, same style as render_state; velocity in m/s."""
    v = [(s[i] - prev[i]) * FPS if prev is not None else 0.0 for i in range(3)]
    f = lambda x: f"{x:+.2f}"
    return (f"tool {f(s[0])} {f(s[1])} {f(s[2])} vel {f(v[0])} {f(v[1])} {f(v[2])} "
            f"grip {width(s):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# download + convert
# ─────────────────────────────────────────────────────────────────────────────
def load_all():
    import pandas as pd
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(REPO, repo_type="dataset",
                                  allow_patterns=["data/*", "meta/*"], max_workers=16))
    tdf = pd.read_parquet(root / "meta/tasks.parquet")
    if "task" in tdf.columns:
        tasks = {int(r.task_index): str(r.task) for r in tdf.itertuples()}
    else:
        tasks = {int(ti): str(task) for task, ti in zip(tdf.index, tdf["task_index"])}
    files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"repo {REPO}: {len(files)} data files · {len(df)} frames · "
          f"{df['episode_index'].nunique()} episodes · {len(tasks)} tasks")
    return df.sort_values(["episode_index", "frame_index"]), tasks


def convert(n_episodes: int, out: Path, dead: float, eval_frac: float, seed: int) -> int:
    import numpy as np
    df, tasks = load_all()
    ep_task = df.groupby("episode_index")["task_index"].first()
    rng = random.Random(seed)
    by_task = collections.defaultdict(list)
    for ep, ti in ep_task.items():
        by_task[int(ti)].append(int(ep))
    per = (n_episodes // len(by_task)) if n_episodes else None
    chosen, split = [], {}
    for ti, eps in sorted(by_task.items()):
        eps = sorted(eps)
        rng.shuffle(eps)
        eps = eps[:per] if per else eps
        n_eval = max(1, round(len(eps) * eval_frac))
        for i, ep in enumerate(eps):
            split[ep] = "eval" if i < n_eval else "train"
        chosen += eps
    df = df[df["episode_index"].isin(set(chosen))]

    A = np.stack(df["action"].to_numpy())
    sp = np.linalg.norm(A[:, :3], axis=1)
    cuts = tuple(float(x) for x in np.quantile(sp, [1 / 3, 2 / 3]))
    print(f"chosen {len(chosen)} episodes over {len(by_task)} tasks · frames {len(df)} · "
          f"speed cuts {cuts[0]:.3f}/{cuts[1]:.3f} · deadband {dead}")

    out.mkdir(parents=True, exist_ok=True)
    counts = {k: collections.Counter() for k in QUESTIONS}
    events, speaking, n_frames = collections.Counter(), 0, 0
    task_done = collections.defaultdict(list)           # instr -> steps finished per episode
    task_steps = {}
    agree = {q: [0, 0] for q in ("axis_x", "axis_y", "height")}
    corr_a, corr_d = [], []
    grip_ok = {"close": [0, 0], "open": [0, 0]}
    examples = {}
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for ep, g in df.groupby("episode_index"):
            instr = tasks.get(int(g["task_index"].iloc[0]), "")
            acts = [list(map(float, a)) for a in g["action"]]
            sts = [list(map(float, s)) for s in g["observation.state"]]
            labs, segs, steps, n_done = label_episode(acts, sts, instr, dead, cuts)
            task_steps[instr] = steps
            task_done[instr].append(n_done)
            for s in segs:
                events[s["kind"]] += 1
            recs = []
            T = len(acts)
            for t, ((ans, say, ev), s) in enumerate(zip(labs, sts)):
                for k in QUESTIONS:
                    counts[k][ans[k]] += 1
                speaking += say is not None
                if ev == "release":
                    events["release"] += 1
                # motion check: does the labelled direction match where the hand went?
                if t + LOOKAHEAD < T:
                    d = [sts[t + LOOKAHEAD][i] - s[i] for i in range(3)]
                    corr_a.append(acts[t][:3])
                    corr_d.append(d)
                    for q, i in (("axis_x", 0), ("axis_y", 1), ("height", 2)):
                        lab = ans[q]
                        if lab not in ("stay",) and abs(d[i]) > 0.005:
                            want = lab in ("positive", "up")
                            agree[q][0] += (d[i] > 0) == want
                            agree[q][1] += 1
                    # gripper check: fingers move the commanded way within 5 frames
                    if ans["gripper"] in ("close", "open"):
                        w0, w5 = width(s), width(sts[min(t + 5, T - 1)])
                        good = (w5 < w0 - 1e-3) if ans["gripper"] == "close" else (w5 > w0 + 1e-3)
                        grip_ok[ans["gripper"]][0] += good
                        grip_ok[ans["gripper"]][1] += 1
                recs.append({"t": t, "state": state_sentence(s, sts[t - 1] if t else None),
                             "answers": ans, "say": say})
            n_frames += len(recs)
            fh.write(json.dumps({"episode": int(ep), "split": split[int(ep)],
                                 "prompt": instr,
                                 "steps": [f"{s['verb']} {s['obj']}" for s in steps],
                                 "frames": recs}) + "\n")
            if instr not in examples:
                examples[instr] = recs

    # inverse-frequency class weights (mean 1 over classes, capped at 20)
    weights = {}
    for k, c in counts.items():
        tot = sum(c.values())
        raw = {o: tot / (len(QUESTIONS[k]) * max(c[o], 1)) for o in QUESTIONS[k]}
        weights[k] = {o: round(min(v, 20.0), 3) for o, v in raw.items()}
    meta = {"source": REPO, "episodes": len(chosen), "frames": n_frames,
            "split": collections.Counter(split[e] for e in chosen),
            "speed_cuts": cuts, "deadband": dead, "fps": FPS,
            "questions": QUESTIONS, "class_weights": weights}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 78 + "\nCONVERSION REPORT\n" + "=" * 78)
    print(f"episodes {len(chosen)} ({dict(meta['split'])}) · frames {n_frames} · "
          f"tasks {len(task_steps)}")
    print(f"speaking frames {speaking} ({100 * speaking / max(n_frames, 1):.1f}%) · "
          f"grasp segments {dict(events)}")
    for k, c in counts.items():
        tot = sum(c.values())
        print(f"  {k:8s} " + "  ".join(f"{o}={100 * c[o] / tot:5.1f}%" for o in QUESTIONS[k])
              + f"   weights {weights[k]}")

    print("\nMOTION CHECKS (labels vs the recorded hand/finger motion)")
    ca, cd = np.array(corr_a), np.array(corr_d)
    print("  corr(action xyz, future eef displacement xyz)  rows=action, cols=motion")
    for i, n in enumerate("xyz"):
        row = [float(np.corrcoef(ca[:, i], cd[:, j])[0, 1]) for j in range(3)]
        print(f"    a_{n}: " + "  ".join(f"{v:+.2f}" for v in row))
    for q, (g, n) in agree.items():
        print(f"  {q:7s} direction agrees with motion: {100 * g / max(n, 1):5.1f}%  (n={n})")
    for q, (g, n) in grip_ok.items():
        print(f"  gripper '{q}' fingers follow within 5 frames: {100 * g / max(n, 1):5.1f}%  (n={n})")

    print("\nOBJECTS PER TASK (carries/episode should equal #objects)")
    mism = 0
    for instr in sorted(task_steps):
        c = task_done[instr]
        mean = sum(c) / len(c)
        full = 100 * sum(x == len(task_steps[instr]) for x in c) / len(c)
        flag = "" if abs(mean - len(task_steps[instr])) < 0.5 else "  <-- check"
        mism += bool(flag)
        st = [f"{s['verb']} {s['obj']}" for s in task_steps[instr]]
        print(f"  {len(c):3d} ep · steps done {mean:4.2f}/{len(st)} ({full:5.1f}% all) · {st} · {instr!r}{flag}")
    print(f"tasks whose completed-step count disagrees with the parsed steps: {mism}/{len(task_steps)}")

    print("\nEXAMPLES (every 15th frame + every speaking frame)")
    pick = [i for i in examples if any(w in i for w in ("moka", "both the", "microwave", "turn on"))][:4]
    for instr in pick:
        print(f"\n  PROMPT: {instr}")
        for r in examples[instr]:
            if r["t"] % 15 == 0 or r["say"]:
                a = r["answers"]
                print(f"   t={r['t']:3d} {r['state']} | x={a['axis_x'][:3]} y={a['axis_y'][:3]} "
                      f"h={a['height']:4s} g={a['gripper']:5s} i={a['intent']:8s} "
                      f"s={a['speed']:6s} | say={r['say']!r}")
    print(f"\nwrote {out / 'episodes.jsonl'} and {out / 'meta.json'}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
def self_test() -> int:
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {info}")

    cases = {
        "put both the alphabet soup and the cream cheese box in the basket":
            (["alphabet soup", "cream cheese box"], ["basket"]),
        "put the yellow and white mug in the microwave and close it":
            (["yellow and white mug"], ["microwave"]),
        "turn on the stove and put the moka pot on it": (["stove", "moka pot"], []),
        "put both moka pots on the stove": (["moka pots"], ["stove"]),
        "pick up the book and place it in the back compartment of the caddy":
            (["book"], ["back compartment", "caddy"]),
        "put the white mug on the plate and put the chocolate pudding to the right of the plate":
            (["white mug", "chocolate pudding"], ["plate", "right", "plate"]),
    }
    for instr, want in cases.items():
        got = parse_objects(instr)
        check(f"parse {instr[:40]!r}", got == want, got)

    # two objects: carry A (held, moved), a regrasp of B, then carry B
    open_, shut = -1.0, 1.0
    acts, sts = [], []
    def step(dx, dz, g, pos, w):
        acts.append([dx, 0.0, dz, 0, 0, 0, g]); sts.append(pos + [0, 0, 0, w / 2, -w / 2])
    p = [0.0, 0.0, 0.5]
    step(0.5, -0.5, open_, list(p), 0.08)                     # t0 approach
    step(0.0, 0.0, shut, list(p), 0.08)                       # t1 close on A
    step(0.0, 0.0, shut, list(p), 0.03)
    step(0.0, 0.6, shut, list(p), 0.03)                       # t3 lift
    step(0.6, 0.0, shut, [0.1, 0.0, 0.55], 0.03)
    step(0.0, 0.0, open_, [0.1, 0.0, 0.55], 0.03)             # t5 release A (carry 0.11 m)
    step(0.5, 0.0, open_, [0.1, 0.0, 0.55], 0.08)             # t6 move to B
    step(0.0, 0.0, shut, [0.1, 0.0, 0.55], 0.08)              # t7 close on B
    step(0.0, 0.0, shut, [0.1, 0.0, 0.55], 0.02)
    step(0.0, 0.0, open_, [0.1, 0.0, 0.55], 0.02)             # t9 release (regrasp, no travel)
    step(0.0, 0.0, shut, [0.1, 0.0, 0.55], 0.08)              # t10 close on B again
    step(0.0, 0.6, shut, [0.1, 0.0, 0.55], 0.02)              # t11 lift
    step(0.0, 0.0, shut, [0.2, 0.0, 0.60], 0.02)
    step(0.0, 0.0, open_, [0.2, 0.0, 0.60], 0.02)             # t13 release B (carry)
    step(0.0, 0.0, shut, [0.2, 0.0, 0.60], 0.08)              # t14 close on nothing
    step(0.0, 0.0, shut, [0.2, 0.0, 0.60], 0.0)
    step(0.0, 0.0, open_, [0.2, 0.0, 0.60], 0.0)              # t16 open (missed)
    labs, segs, _, _ = label_episode(acts, sts, "put both the red cup and the blue bowl in the basket",
                                     0.2, (0.3, 0.6))
    says = {t: s for t, (_, s, _) in enumerate(labs) if s}
    check("segment kinds carry/regrasp/carry/missed",
          [s["kind"] for s in segs] == ["carry", "regrasp", "carry", "missed"],
          [s["kind"] for s in segs])
    check("speech names the right object per grasp", says == {
        0: "moving to the red cup", 1: "grasped the red cup", 3: "lifting the red cup",
        5: "released the red cup", 6: "moving to the blue bowl", 7: "grasped the blue bowl",
        9: "released the blue bowl", 10: "grasped the blue bowl", 11: "lifting the blue bowl",
        13: "released the blue bowl"}, says)
    check("missed grasp is silent and labelled 'grasp'",
          14 not in says and 16 not in says and labs[15][0]["intent"] == "grasp")
    check("hold after the closing phase", labs[4][0]["intent"] == "hold")
    check("state sentence velocity in m/s",
          state_sentence([0.11, 0.0, 0.5, 0, 0, 0, 0.04, -0.04], [0.10, 0.0, 0.5, 0, 0, 0, 0.04, -0.04])
          == "tool +0.11 +0.00 +0.50 vel +0.10 +0.00 +0.00 grip 0.080",
          state_sentence([0.11, 0.0, 0.5, 0, 0, 0, 0.04, -0.04], [0.10, 0.0, 0.5, 0, 0, 0, 0.04, -0.04]))
    step_cases = {
        "turn on the stove and put the moka pot on it": ["turn on stove", "put moka pot"],
        "put both moka pots on the stove": ["put moka pot", "put moka pot"],
        "pick up the black bowl between the plate and the ramekin and place it on the plate":
            ["put black bowl"],
        "put the yellow and white mug in the microwave and close it":
            ["put yellow and white mug", "close microwave"],
        "open the top drawer and put the bowl inside": ["open top drawer", "put bowl"],
        "open the middle drawer of the cabinet": ["open middle drawer"],
        "push the plate to the front of the stove": ["push plate"],
        "turn on the stove": ["turn on stove"],
        "put both the alphabet soup and the cream cheese box in the basket":
            ["put alphabet soup", "put cream cheese box"],
        "put the black bowl in the bottom drawer of the cabinet and close it":
            ["put black bowl", "close bottom drawer"],
        "pick up the alphabet soup and place it in the basket": ["put alphabet soup"],
        "pick up the book and place it in the back compartment of the caddy": ["put book"],
    }

    for instr, want in step_cases.items():
        got = [f"{s['verb']} {s['obj']}" for s in parse_steps(instr)]
        check(f"steps {instr[:40]!r}", got == want, got)

    # interact then carry: knob turn (held, small travel) then pick+carry the pot
    acts.clear(); sts.clear()
    step(0.5, 0.0, open_, [0.0, 0.0, 0.5], 0.08)              # t0 move to knob
    step(0.0, 0.0, shut, [0.0, 0.0, 0.5], 0.08)               # t1 close on knob
    step(0.0, 0.0, shut, [0.03, 0.0, 0.5], 0.03)              # t2 twist (3 cm)
    step(0.0, 0.0, open_, [0.03, 0.0, 0.5], 0.03)             # t3 let go
    step(0.5, 0.0, open_, [0.03, 0.0, 0.5], 0.08)             # t4 move to pot
    step(0.0, 0.0, shut, [0.03, 0.0, 0.5], 0.08)              # t5 grasp pot
    step(0.0, 0.6, shut, [0.03, 0.0, 0.5], 0.03)              # t6 lift
    step(0.0, 0.0, shut, [0.2, 0.0, 0.6], 0.03)
    step(0.0, 0.0, open_, [0.2, 0.0, 0.6], 0.03)              # t8 release pot
    labs, segs, steps, n_done = label_episode(
        acts, sts, "turn on the stove and put the moka pot on it", 0.2, (0.3, 0.6))
    says = {t: s for t, (_, s, _) in enumerate(labs) if s}
    check("interaction speech + steps done", n_done == 2 and says == {
        0: "moving to the stove", 1: "turning on the stove", 3: "turned on the stove",
        4: "moving to the moka pot", 5: "grasped the moka pot", 6: "lifting the moka pot",
        8: "released the moka pot"}, (n_done, says))

    # open-hand push, nothing else: no grasp segment at all -> said at the end
    acts.clear(); sts.clear()
    step(0.5, 0.0, open_, [0.00, 0.0, 0.5], 0.08)             # t0
    step(0.5, 0.0, open_, [0.05, 0.0, 0.5], 0.08)
    step(0.5, 0.0, open_, [0.10, 0.0, 0.5], 0.08)
    step(0.0, 0.0, open_, [0.12, 0.0, 0.5], 0.08)             # t3 last frame
    labs, segs, steps, n_done = label_episode(
        acts, sts, "open the middle drawer of the cabinet", 0.2, (0.3, 0.6))
    says = {t: s for t, (_, s, _) in enumerate(labs) if s}
    check("open-hand push with no grasp: done, said on the last frame",
          n_done == 1 and not segs and says == {
              0: "moving to the middle drawer", 3: "opened the middle drawer"},
          (n_done, says))

    # push the drawer open, then carry the bowl: the push is said at the far
    # end of the stroke, and the single grasp goes to the bowl, not the drawer
    acts.clear(); sts.clear()
    step(0.5, 0.0, open_, [0.00, 0.0, 0.5], 0.08)             # t0 to the drawer
    step(0.5, 0.0, open_, [0.05, 0.0, 0.5], 0.08)
    step(0.0, 0.0, open_, [0.10, 0.0, 0.5], 0.08)             # t2 end of the stroke
    step(0.5, 0.0, open_, [0.08, 0.0, 0.5], 0.08)             # t3 to the bowl
    step(0.0, 0.0, shut, [0.08, 0.0, 0.5], 0.08)              # t4 grasp bowl
    step(0.0, 0.6, shut, [0.08, 0.0, 0.5], 0.03)              # t5 lift
    step(0.0, 0.0, shut, [0.20, 0.0, 0.6], 0.03)
    step(0.0, 0.0, open_, [0.20, 0.0, 0.6], 0.03)             # t7 release
    labs, segs, steps, n_done = label_episode(
        acts, sts, "open the top drawer and put the bowl inside", 0.2, (0.3, 0.6))
    says = {t: s for t, (_, s, _) in enumerate(labs) if s}
    check("push drawer then carry bowl: 2 steps, grasp assigned to the bowl",
          n_done == 2 and says == {
              0: "moving to the top drawer", 2: "opened the top drawer",
              3: "moving to the bowl", 4: "grasped the bowl", 5: "lifting the bowl",
              7: "released the bowl"}, (n_done, says))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="0 = all, else ~N stratified by task")
    ap.add_argument("--out", default="/kaggle/working/libero_conv")
    ap.add_argument("--deadband", type=float, default=0.2,
                    help="|action| below this counts as 'stay' (actions are in [-1,1])")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    return convert(a.episodes, Path(a.out), a.deadband, a.eval_frac, a.seed)


if __name__ == "__main__":
    sys.exit(main())
