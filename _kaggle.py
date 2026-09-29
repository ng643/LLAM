"""
_kaggle.py — push the LLMM training pipeline to Kaggle and pull the artifacts back.

Kaggle notebooks are the only *headless* compute this project can borrow: the local
machine is a Ryzen 880M iGPU on Windows (8 GB UMA carve-out, and it keeps taking the
whole laptop down under a training-sized process).  A pushed notebook ("Save & Run
All") runs without a browser, writes into /kaggle/working, and `kaggle kernels output`
downloads whatever it left there.

Design notes
------------
* The three project modules travel **base64-embedded in the notebook**.  A Kaggle
  dataset would be the tidier carrier, but that needs a second API surface and a
  dataset version bump on every code edit; a 500 KB cell is free and makes the
  kernel exactly reproducible from one file.
* The notebook shells out to the same `python train_loop_robot.py …` command lines
  used locally — no import-time magic, and the cell log is the script's own stdout.
* `enable_internet` is required for the HuggingFace download of the 0.5B trunk.

Usage
-----
    python _kaggle.py smoke            # tiny model, 20 steps: validates the round trip
    python _kaggle.py qwen             # the real multi-task + speech run
    python _kaggle.py status           # last run state
    python _kaggle.py pull             # download /kaggle/working
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER = "gotbot12345"
MODULES = ("train_loop_robot.py", "mujoco_env_bridge.py", "train_live_mujoco.py",
           "libero_convert.py")
STAGE = ROOT / "_kaggle_kernel"
OUT = ROOT / "_kaggle_out"


# ─────────────────────────────────────────────────────────────────────────────
# notebook construction
# ─────────────────────────────────────────────────────────────────────────────
def _cell(src: str, kind: str = "code", cid: str | None = None) -> dict:
    lines = src.splitlines(keepends=True)
    # nbformat warns (and will hard-error) on a missing cell id
    cid = cid or f"c{abs(hash(src)) % 10**8:08d}"
    if kind == "code":
        return {"cell_type": kind, "id": cid, "metadata": {}, "source": lines,
                "execution_count": None, "outputs": []}
    return {"cell_type": kind, "id": cid, "metadata": {}, "source": lines}


def _payload() -> str:
    blob = {}
    for name in MODULES:
        blob[name] = (ROOT / name).read_text(encoding="utf-8")
    raw = json.dumps(blob).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


CELL_UNPACK = '''
# ── the project sources, carried inline ────────────────────────────────────────
import base64, json, pathlib, sys, textwrap

SRC = pathlib.Path("/kaggle/working/src"); SRC.mkdir(parents=True, exist_ok=True)
_blob = json.loads(base64.b64decode(_PAYLOAD_B64).decode("utf-8"))
for _name, _text in _blob.items():
    (SRC / _name).write_text(_text, encoding="utf-8")
    print(f"wrote {_name:24s} {len(_text):7d} chars  {_text.count(chr(10))+1:5d} lines")
sys.path.insert(0, str(SRC))
import py_compile
for _name in _blob:
    py_compile.compile(str(SRC / _name), doraise=True)
print("all modules compile")
'''

CELL_ENV = '''
# ── what we got ────────────────────────────────────────────────────────────────
import os, shutil, subprocess, sys, time, torch
print("torch      :", torch.__version__, "| cuda", torch.version.cuda, "| hip", torch.version.hip)
print("python     :", sys.version.split()[0])
print("gpus       :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
print("cpu        :", os.cpu_count(), "cores | ram",
      f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.1f} GiB"
      if hasattr(os, "sysconf") else "")
print("disk free  :", shutil.disk_usage("/kaggle/working").free / 2**30, "GiB")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi : absent (CPU-only kernel)")
import transformers; print("transformers:", transformers.__version__)
print("mujoco     :", subprocess.run(["python", "-c", "import mujoco;print(mujoco.__version__)"],
                                     capture_output=True, text=True).stdout.strip() or "MISSING")
'''

CELL_STAGE_TMPL = '''
# ── @@TITLE@@ ────────────────────────────────────────────────────────────────────
import os, subprocess, sys, time
@@ENV@@t0 = time.time()
cmd = [sys.executable, "-u", str(SRC / "@@SCRIPT@@")] + @@ARGV@@
print("$ " + " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, cwd=str(SRC), stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"\\n[@@TITLE@@] exit={rc}  wall={time.time()-t0:.0f}s", flush=True)
@@ONFAIL@@
'''

CELL_DEPS = """
# ── optional deps ──────────────────────────────────────────────────────────────
import importlib, subprocess, sys
for mod, pip in (("mujoco", "mujoco"),):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', '?')} already present")
    except ImportError:
        print(f"installing {pip} …", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip], check=False)
        importlib.invalidate_caches()
        print(f"{mod}: {importlib.import_module(mod).__version__} installed")
"""

CELL_SUMMARY = '''
# ── what is in /kaggle/working ─────────────────────────────────────────────────
import os, pathlib
for p in sorted(pathlib.Path("/kaggle/working").rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/2**20:9.1f} MiB  {p}")
'''


def _stage_cell(title: str, script: str, argv: list[str], fatal: bool = False,
                env: dict | None = None) -> str:
    # token replacement, not str.format: the cell body is Python and full of braces
    on_fail = (f'if rc != 0 and {"True" if fatal else "False"}:\n'
               f'    raise SystemExit(f"{title} failed (rc={{rc}})")\n')
    envline = f"os.environ.update({env!r})\n" if env else ""
    return (CELL_STAGE_TMPL.replace("@@TITLE@@", title)
                          .replace("@@SCRIPT@@", script)
                          .replace("@@ARGV@@", repr(argv))
                          .replace("@@ENV@@", envline)
                          .replace("@@ONFAIL@@", on_fail))


def build(name: str, title: str, stages: list[tuple[str, str, list[str], bool]],
          notes: str, gpu: bool = True, sources: list[str] | None = None,
          pre: str = "", datasets: list[str] | None = None) -> Path:
    STAGE.mkdir(exist_ok=True)
    payload = _payload()
    cells = [
        _cell(f"# {title}\n\n{notes}", "markdown"),
        _cell("_PAYLOAD_B64 = " + repr(payload) + "\n" + CELL_UNPACK),
        _cell(CELL_ENV),
    ]
    cells.append(_cell(CELL_DEPS))
    if pre:
        cells.append(_cell(pre))
    for stage in stages:
        t, script, argv, fatal = stage[:4]
        env = stage[4] if len(stage) > 4 else None
        cells.append(_cell(_stage_cell(t, script, argv, fatal, env)))
    cells.append(_cell(CELL_SUMMARY))
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    (STAGE / "run.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {"id": f"{USER}/{name}", "title": title, "code_file": "run.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": list(datasets or []), "competition_sources": [],
            "kernel_sources": list(sources or []),
            "model_sources": []}
    (STAGE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size = (STAGE / "run.ipynb").stat().st_size
    print(f"notebook {STAGE/'run.ipynb'}  {size/2**20:.2f} MiB  ({len(cells)} cells)")
    return STAGE


# ─────────────────────────────────────────────────────────────────────────────
# recipes
# ─────────────────────────────────────────────────────────────────────────────
QWEN_ARGS = [
    "--model", "qwen", "--device", "cuda", "--dtype", "fp16",
    "--trainable", "lora", "--lora-rank", "16", "--lora-alpha", "32",
    "--loops", "6", "--recurrent-layers", "2",
    "--fusion-heads", "8", "--carry", "--questions", "auto",
    "--untie-lm-head", "--q-weight", "2.0", "--text-weight", "0.5",
    # batch 4 (not 8): measured on the probe, the full batch OOMs and the
    # guard's 2-way split costs ~15% MORE per control step than batch 4, which
    # fits with no retry at all (peak 10.1G of the 12G cap).
    "--task-weight", "0.5", "--batch", "4", "--window", "8",
    "--windows-per-episode", "4", "--steps", "4500", "--lr", "5e-4",
    "--trace", "2",
    "--log-every", "25", "--eval-every", "300", "--eval-batches", "8",
    "--vram-fraction", "0.85", "--vram-budget-gib", "12", "--gc-every", "0",
    "--grad-checkpoint",
    # interface variation (question subset / order / wording, option shuffle,
    # schema layout, a distractor question, state-sentence format) plus 30 % of
    # the batches drawn from the converted LIBERO demos (first stage below).
    # No episodes.jsonl (conversion failed or was skipped) is only a warning:
    # the trainer carries on synthetic-only.  VRAM (not measured yet): LIBERO
    # state rows carry no objects, so they are SHORTER than synthetic rows;
    # variation adds at most one distractor marker per state row and a longer
    # schema in the prompt.  Batch 4 / window 8 / grad checkpointing unchanged —
    # the headroom was ~1.9G at the 12G cap, and the VRAM guard splits a batch
    # instead of crashing if a longer interface ever does not fit.
    "--vary", "--libero", "/kaggle/working/libero_conv/episodes.jsonl",
    "--libero-frac", "0.3",
    "--out", "/kaggle/working/robot_kaggle.pt",
]


def live_argv(ckpt: str) -> list[str]:
    return (["--model", "qwen", "--device", "cuda", "--dtype", "fp16",
            "--checkpoint", ckpt,
            "--out", "/kaggle/working/robot_kaggle_live.pt",
            "--best-out", "/kaggle/working/robot_kaggle_live_best.pt",
            "--window", "8", "--loops", "6", "--questions", "auto",
            # llmm-live v1 trained the 136M untied LM head (--trainable adapters):
            # question CE fell to 0.05 while rollout agreement dropped 95 → 61-68 %.
            # Freeze the read-out, adapt through LoRA (rank taken from the
            # checkpoint) + the small modules, at a lower lr and larger batch.
            "--trainable", "lora", "--freeze-lm-head", "--carry",
            # 150 steps × 16 windows = 2400 windows/round (v1: 300 × 4 = 1200)
            "--imitate-rounds", "6", "--imitate-frames", "600", "--imitate-steps", "150",
            # batch 8 without grad checkpointing OOM'd in the DAgger fit (12.11 GiB
            # in use at the 12 GiB cap); keep micro-batch 4 and accumulate 4 → 16
            "--imitate-batch", "4", "--imitate-accum", "4", "--grad-checkpoint",
            "--imitate-lr", "5e-5", "--imitate-acc-windows", "256",
            "--frames", "2400", "--rollout", "256",
            # update-batch 8 OOM'd on the first A2C update in llmm-live v1
            # (12.08 GiB at the 12 GiB cap, LM head trainable then)
            "--update-batch", "4", "--speak-max", "8", "--duration", "0",
            "--save-every", "200", "--eval-frames", "240", "--headless",
            # the live defaults were tuned for the 880M's 8 GiB carve-out; the T4
            # has 14.56 GiB, so raise the cap the same way the trainer does
            "--vram-budget-gib", "12"])


def recipe_smoke() -> tuple[str, str, list]:
    argv = ["--model", "tiny", "--device", "cuda", "--steps", "20", "--batch", "4",
            "--window", "8", "--trainable", "adapters", "--questions", "auto",
            "--q-weight", "2.0", "--text-weight", "0.5", "--untie-lm-head",
            "--task-weight", "0.5", "--eval-every", "10", "--eval-batches", "2",
            # UNBOUNDED ACT: the sinusoidal cycle tag has no table, so the
            # ceiling is only a safety net — the halting head decides the
            # count and the ponder penalty just rewards finishing early.
            "--cycle-tag", "sinusoidal", "--loops", "16",
            "--out", "/kaggle/working/robot_smoke.pt", "--seed", "5"]
    return ("llmm-smoke", "llmm smoke", [
        # CUDA_LAUNCH_BLOCKING makes an illegal index fail AT its own kernel
        # instead of at the next sync, so the traceback names the real op.
        ("trainer (tiny, 20 steps)", "train_loop_robot.py", argv, True,
         {"CUDA_LAUNCH_BLOCKING": "1"}),
        ("bridge self-test", "mujoco_env_bridge.py",
         ["--self-test", "--model", "tiny", "--device", "cuda"], False),
        ("live-RL self-test", "train_live_mujoco.py",
         ["--self-test", "--model", "tiny", "--device", "cuda", "--window", "8"], False),
    ], "Round-trip validation: sources unpack, GPU is visible, the trainer runs, "
       "and both self-tests pass on CUDA/fp16 (T4 is Turing — no bf16 tensor cores).")


def recipe_qwen() -> tuple[str, str, list]:
    return ("llmm-qwen-multi-task", "llmm qwen multi-task", [
        # NOT fatal but checked: the trainer tests for episodes.jsonl itself and,
        # when it is missing or unreadable, warns and trains synthetic-only.
        # CPU work (download of LIBERO's per-frame tables + conversion) that runs
        # on the GPU session so the whole pipeline stays one kernel.
        ("convert all LIBERO episodes (40 tasks)", "libero_convert.py",
         ["--episodes", "0", "--out", "/kaggle/working/libero_conv"], False),
        ("supervised warm start (qwen 0.5B, multi-task + speech)",
         "train_loop_robot.py", QWEN_ARGS, True),
        ("live DAgger + RL on the MuJoCo bridge", "train_live_mujoco.py",
         live_argv("/kaggle/working/robot_kaggle.pt"), False,
         # NOT fatal: a raised cell marks the kernel ERROR and Kaggle then keeps
         # no output files — that is how robot_kaggle.pt was lost once.
         {"MUJOCO_GL": "egl"}),
    ], "The full pipeline in one session: LIBERO conversion (non-fatal), supervised "
       "multi-task + speech warm start with interface variation and 30 % LIBERO "
       "batches, then DAgger and A2C against the live MuJoCo arm.")


def recipe_diag() -> tuple[str, str, list]:
    """CPU-only (no GPU quota): localise an index fault where the error is exact.

    A CUDA assert reports at the *next* launch, so it names the wrong op; the
    same fault on CPU raises `IndexError: index N is out of bounds for dimension
    D with size M` and names the real one.  Stage A is the exact trainer path
    that failed on CUDA, stage B the trainer self-test, stage C the bridge.
    """
    argv = ["--model", "tiny", "--device", "cpu", "--steps", "3",
            "--batch", "2", "--window", "4", "--trainable", "adapters",
            "--questions", "auto", "--q-weight", "2.0", "--text-weight", "0.5",
            "--untie-lm-head", "--task-weight", "0.5", "--eval-every", "0",
            "--seed", "5", "--out", "/kaggle/working/robot_diag.pt"]
    return ("llmm-diag", "llmm diag", [
        ("trainer on CPU (3 steps — the CUDA-failing path)",
         "train_loop_robot.py", argv, False),
        ("trainer self-test on CPU", "train_loop_robot.py",
         ["--self-test", "--model", "tiny", "--device", "cpu"], False),
        ("trainer self-test on CPU with LoRA (rank 8)", "train_loop_robot.py",
         ["--self-test", "--model", "tiny", "--device", "cpu",
          "--lora-rank", "8", "--lora-alpha", "16"], False),
        ("trainer on CPU, LoRA trainable (3 steps)", "train_loop_robot.py",
         argv + ["--trainable", "lora", "--lora-rank", "8",
                 "--out", "/kaggle/working/robot_lora_diag.pt"], False),
        # the new flags: fine-tune from the LoRA checkpoint above with --vary on
        # (plain/shuf split in every log line), a held lr, a mid-run save
        ("--init-from + --vary 1.0 + plain/shuf log + --ckpt-every (4 steps)",
         "train_loop_robot.py",
         argv + ["--trainable", "lora", "--lora-rank", "8",
                 "--init-from", "/kaggle/working/robot_lora_diag.pt",
                 "--vary", "1.0", "--steps", "4", "--log-every", "1",
                 "--lr-hold", "0.5", "--lr-floor", "0.1", "--ckpt-every", "2",
                 "--out", "/kaggle/working/robot_init_diag.pt"], False),
        ("EXPECTED TO FAIL: --init-from with a different LoRA rank",
         "train_loop_robot.py",
         argv + ["--trainable", "lora", "--lora-rank", "16",
                 "--init-from", "/kaggle/working/robot_lora_diag.pt",
                 "--out", "/kaggle/working/robot_bad_init.pt"], False),
        ("bridge self-test on CPU", "mujoco_env_bridge.py",
         ["--self-test", "--model", "tiny", "--device", "cpu"], False),
        # the live stack on CPU: JSON-only control, speech after the state,
        # the imitation pool's speech targets and the speech CE in the fit
        ("live-RL self-test on CPU (speech CE on)", "train_live_mujoco.py",
         ["--self-test", "--model", "tiny", "--device", "cpu", "--window", "4",
          "--text-weight", "0.5", "--headless"], False),
        # the real trunk on CPU, BEFORE any GPU quota is spent: this is the only
        # place the Qwen tokenizer's fixed-width state sentence, the trunk build
        # under transformers 5.x and the LoRA injection on 24 real layers are
        # all exercised together (3 steps, batch 2 — minutes, not hours)
        ("trainer on CPU, Qwen 0.5B + LoRA (3 steps)", "train_loop_robot.py",
         ["--model", "qwen", "--device", "cpu", "--steps", "3", "--batch", "2",
          "--window", "4", "--trainable", "lora", "--lora-rank", "16",
          "--lora-alpha", "32", "--questions", "auto", "--q-weight", "2.0",
          "--text-weight", "0.5", "--untie-lm-head", "--task-weight", "0.5",
          "--eval-every", "0", "--seed", "5",
          "--out", "/kaggle/working/robot_qwen_diag.pt"], False),
    ], "CPU-only triage of the `index out of bounds` CUDA assert: the same code "
       "path raises a precise IndexError on CPU, naming the tensor and the "
       "offending index.  Plus the LoRA invariants (identity at init, frozen "
       "base) and a real LoRA training step.  No GPU quota is consumed.")


def recipe_qwenprobe() -> tuple[str, str, list]:
    """Scale probe: one real-shape step, traced, before the 3600-step run.

    The full run printed nothing for ~55 min.  Two things make that
    indistinguishable from a hang: the allocator cap (`--vram-budget-gib 8`
    protects the 880M's carve-out, but a T4 has 14.56 GiB) and the fact that
    nothing prints between the header and the step line.  `--trace` prints a
    synchronised line per phase, so the last line names whatever never
    returned.  Rung A is small enough to be known-good; rung B is the exact
    shape of the real run with the fixes (12 GiB budget, gradient
    checkpointing, chunked token CE); rung C is the fallback if B still OOMs.
    """
    base = ["--model", "qwen", "--device", "cuda", "--dtype", "fp16",
            "--trainable", "lora", "--lora-rank", "16", "--lora-alpha", "32",
            "--loops", "6", "--recurrent-layers", "2", "--fusion-heads", "8",
            "--carry", "--questions", "auto", "--untie-lm-head",
            "--q-weight", "2.0", "--text-weight", "0.5", "--task-weight", "0.5",
            "--vram-fraction", "0.85", "--gc-every", "0", "--seed", "5",
            "--eval-every", "0", "--log-every", "1", "--windows-per-episode", "4"]
    small = base + ["--steps", "2", "--batch", "2", "--window", "2",
                    "--windows-per-episode", "2", "--trace", "2",
                    "--out", "/kaggle/working/robot_probe_small.pt"]
    real = base + ["--steps", "2", "--batch", "8", "--window", "8",
                   "--trace", "2", "--grad-checkpoint",
                   "--vram-budget-gib", "12",
                   "--out", "/kaggle/working/robot_probe_real.pt"]
    fallback = base + ["--steps", "1", "--batch", "4", "--window", "8",
                       "--trace", "1", "--grad-checkpoint",
                       "--vram-budget-gib", "12",
                       "--out", "/kaggle/working/robot_probe_b4.pt"]
    return ("llmm-qwen-probe", "llmm qwen probe", [
        # cheap invariants first (tiny trunk, seconds): question read-out,
        # speech causality, Jev labels, json_action inverse, `say` in the JSON
        ("trainer self-test (tiny, CUDA)", "train_loop_robot.py",
         ["--self-test", "--model", "tiny", "--device", "cuda"], True),
        ("qwen @ batch 2 x window 2 (known-good baseline)", "train_loop_robot.py",
         small, True),
        ("qwen @ batch 8 x window 8 + fixes (the real shape)",
         "train_loop_robot.py", real, True),
        ("qwen @ batch 4 x window 8 + fixes (fallback shape)",
         "train_loop_robot.py", fallback, True),
    ], "Diagnose the wedged first step: three traced Qwen runs — baseline, the "
       "real shape with the fixes, and a fallback shape.  Every stage prints "
       "the phase it is in, so a hang is localised even when it never returns.")

def recipe_libero() -> tuple[str, str, list]:
    """CPU only: download LIBERO's per-frame tables (no videos) and convert them
    to prompt + state sentence + questionnaire labels + speech, with a report."""
    return ("llmm-libero-convert", "llmm libero convert", [
        ("converter self-test", "libero_convert.py", ["--self-test"], True),
        ("convert all LIBERO episodes (40 tasks)", "libero_convert.py",
         ["--episodes", "0", "--out", "/kaggle/working/libero_conv"], False),
    ], "Convert lerobot/libero demos into this project's text format and print "
       "label balance, speech rate, parsed objects and example frames.")

def recipe_live() -> tuple[str, str, list]:
    """Live stage only, from the retrain's checkpoint (the kernel output of
    llmm-qwen-multi-task is mounted read-only under /kaggle/input)."""
    return ("llmm-live", "llmm live", [
        ("live DAgger + RL on the MuJoCo bridge", "train_live_mujoco.py",
         live_argv("/tmp/robot_kaggle.pt"), False, {"MUJOCO_GL": "egl"}),
    ], "DAgger + A2C on the live MuJoCo arm from the supervised checkpoint.")


# llmm-qwen-multi-task's latest output is now v3 (the failed mixed retrain), so
# the live stage reads the good v2 checkpoint from a private dataset instead.
KERNEL_SOURCES: dict[str, list[str]] = {}
DATASET_SOURCES = {"live": [f"{USER}/llmm-ckpt-v2"]}


def _ablate_args(vary: bool, libero: bool, steps: int = 1500,
                 out: str = "/kaggle/working/robot_ablate.pt") -> list[str]:
    """QWEN_ARGS at `steps` steps with --vary and/or --libero switched off."""
    a, i = [], 0
    while i < len(QWEN_ARGS):
        t = QWEN_ARGS[i]
        if t == "--vary":
            i += 1
            continue
        if t in ("--libero", "--libero-frac", "--out", "--steps"):
            i += 2
            continue
        a.append(t)
        i += 1
    a += ["--steps", str(steps), "--out", out]
    if vary:
        a += ["--vary"]
    if libero:
        a += ["--libero", "/kaggle/working/libero_conv/episodes.jsonl",
              "--libero-frac", "0.3"]
    return a


def _recipe_ablate(tag: str, vary: bool, libero: bool):
    def r() -> tuple[str, str, list]:
        stages = []
        if libero:
            stages.append(("convert all LIBERO episodes (40 tasks)", "libero_convert.py",
                           ["--episodes", "0", "--out", "/kaggle/working/libero_conv"],
                           True))
        stages.append((f"ablation {tag}: vary={vary} libero={libero}, 1500 steps",
                       "train_loop_robot.py", _ablate_args(vary, libero), True))
        return (f"llmm-ablate-{tag}", f"llmm ablate {tag}", stages,
                "Which part of the v3 mix stops axis_x/axis_y/speed from being "
                "learned: 1500-step retrains with one feature at a time.")
    return r

def recipe_longvary() -> tuple[str, str, list, str]:
    """The user's test: plain --vary (no LIBERO) trained long enough to
    generalise.  12k steps (~5.7 h at the measured 1.70 s/it), lr held at the
    peak for half the run and ending at 0.1x instead of 0.05x, a checkpoint
    every 1500 steps, and the log splits axis_x/axis_y/speed CE + accuracy
    into shuffled-option rows vs plain rows."""
    a = _ablate_args(True, False, steps=12000,
                     out="/kaggle/working/robot_longvary.pt")
    i = a.index("--eval-every")
    a[i + 1] = "600"
    a += ["--lr-hold", "0.5", "--lr-floor", "0.1", "--ckpt-every", "1500"]
    return ("llmm-longvary", "llmm longvary",
            [("long plain --vary retrain, 12k steps", "train_loop_robot.py", a, True)],
            "Does interface variation generalise with a long run? 12k steps, "
            "--vary 0.75, no LIBERO, lr hold 0.5 / floor 0.1.")

CELL_FIND_CKPT = '''
# ── locate the supervised checkpoint in the mounted kernel output ─────────────
import pathlib, os
hits = sorted(pathlib.Path("/kaggle/input").rglob("robot_kaggle.pt"))
print("found:", [str(h) for h in hits])
assert hits, "robot_kaggle.pt not mounted under /kaggle/input"
if os.path.lexists("/tmp/robot_kaggle.pt"):
    os.remove("/tmp/robot_kaggle.pt")
os.symlink(hits[0], "/tmp/robot_kaggle.pt")
'''

# the follow-up fine-tune reads longvary's checkpoint from its kernel output
CELL_FIND_LONGVARY = CELL_FIND_CKPT.replace("robot_kaggle.pt", "robot_longvary.pt")
KERNEL_SOURCES["vary_libero_ft"] = [f"{USER}/llmm-longvary"]


def recipe_vary_libero_ft() -> tuple[str, str, list, str]:
    """After longvary: fine-tune its checkpoint with --vary AND 30 % LIBERO
    batches (LIBERO rows are varied by their own sampler).  2500 steps at
    lr 1e-4 (held 30 %, floor 0.1), ~1.2 h.  The log splits plain/shuf for
    synthetic rows and lib_plain/lib_shuf for LIBERO rows."""
    a = _ablate_args(True, True, steps=2500, out="/kaggle/working/robot_varylib.pt")
    a[a.index("--lr") + 1] = "1e-4"
    a[a.index("--eval-every") + 1] = "500"
    a += ["--init-from", "/tmp/robot_longvary.pt", "--lr-hold", "0.3",
          "--lr-floor", "0.1", "--ckpt-every", "1000"]
    return ("llmm-varylib-ft", "llmm varylib ft", [
        ("convert all LIBERO episodes (40 tasks)", "libero_convert.py",
         ["--episodes", "0", "--out", "/kaggle/working/libero_conv"], True),
        ("fine-tune longvary with --vary + LIBERO 0.3, 2500 steps",
         "train_loop_robot.py", a, True),
    ], "Follow-up to llmm-longvary: add LIBERO (varied) on top of the varied "
       "synthetic checkpoint.")


RECIPES = {"smoke": recipe_smoke, "qwen": recipe_qwen,
           "qwenprobe": recipe_qwenprobe, "diag": recipe_diag,
           "libero": recipe_libero, "live": recipe_live,
           "ablate_base": _recipe_ablate("base", False, False),
           "ablate_vary": _recipe_ablate("vary", True, False),
           "ablate_libero": _recipe_ablate("libero", False, True),
           "longvary": recipe_longvary, "vary_libero_ft": recipe_vary_libero_ft}

# ─────────────────────────────────────────────────────────────────────────────
# probe: does Kaggle's log stream deliver a RUNNING session's stdout live, and
# does an unflushed print reach it?  (The trainer's progress lines have no
# flush=True; if unflushed output is withheld until the cell ends, a perfectly
# healthy long run looks frozen.)
# ─────────────────────────────────────────────────────────────────────────────
CELL_PROBE_A = '''
import sys, time
print("probe A: 9 ticks, each with flush=True, 10 s apart", flush=True)
for i in range(9):
    print(f"  tick {i+1}/9  t={time.time():.1f}", flush=True)
    time.sleep(10)
print("probe A: done", flush=True)
'''

CELL_PROBE_B = '''
import sys, time
print("probe B: one UNFLUSHED line, then a 40 s sleep, then a flushed line")
time.sleep(40)
print("  B: this line is flushed AFTER the sleep", flush=True)
time.sleep(5)
'''


def push_probe() -> int:
    """Minimal two-cell kernel: no module payload, no deps, ~135 s of ticks."""
    STAGE.mkdir(exist_ok=True)
    cells = [
        _cell("# live-log probe\n\nDoes `kaggle kernels logs --follow` show a "
              "RUNNING session's stdout as it is written, and does an unflushed "
              "`print` reach it?  A is flushed per line, B is not.", "markdown"),
        _cell(CELL_PROBE_A),
        _cell(CELL_PROBE_B),
    ]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    (STAGE / "run.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {"id": f"{USER}/llmm-probe", "title": "llmm probe",
            "code_file": "run.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (STAGE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                                encoding="utf-8")
    print(f"notebook {STAGE/'run.ipynb'}  "
          f"{(STAGE/'run.ipynb').stat().st_size/2**20:.2f} MiB  ({len(cells)} cells)")
    r = _run(["kaggle", "kernels", "push", "-p", str(STAGE)], capture_output=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    return 0 if r.returncode == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action",
                    choices=list(RECIPES) + ["probe", "status", "pull", "log", "delete"])
    ap.add_argument("--name", default=None, help="kernel slug (default: the recipe's)")
    args = ap.parse_args()

    if args.action == "probe":
        return push_probe()

    if args.action in RECIPES:
        name, title, stages, notes = RECIPES[args.action]()
        name = args.name or name
        build(name, title, stages, notes,
              gpu=args.action not in ("diag", "libero"),
              sources=KERNEL_SOURCES.get(args.action),
              datasets=DATASET_SOURCES.get(args.action),
              pre=(CELL_FIND_CKPT if args.action == "live" else
                   CELL_FIND_LONGVARY if args.action == "vary_libero_ft" else ""))
        r = _run(["kaggle", "kernels", "push", "-p", str(STAGE)], capture_output=True)
        print(r.stdout[-3000:], r.stderr[-2000:])
        return 0 if r.returncode == 0 else 1

    name = args.name or RECIPES["qwen"]()[0]
    ref = f"{USER}/{name}"
    if args.action == "status":
        print(_run(["kaggle", "kernels", "status", ref], capture_output=True).stdout)
    elif args.action == "log":
        print(_run(["kaggle", "kernels", "output", ref, "--file-pattern", r".*\.log"],
                   capture_output=True).stdout[-4000:])
    elif args.action == "pull":
        OUT.mkdir(exist_ok=True)
        r = _run(["kaggle", "kernels", "output", ref, "-p", str(OUT)], capture_output=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                print(f"{p.stat().st_size/2**20:9.1f} MiB  {p}")
    elif args.action == "delete":
        r = _run(["kaggle", "kernels", "delete", ref, "-y"], capture_output=True)
        print(r.stdout, r.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
