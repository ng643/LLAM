# LLAM: a looped, text-native robot policy on Qwen2.5-0.5B

LLAM controls a simulated robot arm with a small language model, with no separate action decoder. Each control step, the robot's state is written as a text sentence. The model answers a fixed JSON questionnaire about what to do next, for example `axis_x: negative | stay | positive` or `gripper: open | close | stay`. Each answer is read from the LM head, limited to that question's option tokens. The motor command is built from those answers. After the state, the model can also say a short sentence, such as "grasped the red block", or stay silent.

This is research code and a work in progress. The numbers below are measured, and they include the failures.

## Architecture

- **Trunk:** Qwen2.5-0.5B, frozen, with LoRA (r=16, α=32) on q/k/v/o/gate/up/down.
- **Looped recurrent block:** layers 0–21 run once. The last 2 layers are re-applied up to 6 times under Adaptive Computation Time: a halting head, the Graves remainder, and a ponder cost. Each cycle gets a sinusoidal cycle tag.
- **PromptFusion:** gated cross-attention from the state tokens to the prompt (instruction plus questionnaire), applied every cycle. The gate starts at zero.
- **Carry token:** a latent vector carried from one control step to the next.
- **Read-out:** one answer marker per question after each state sentence. `softmax(h_marker · W_lm[option_ids]ᵀ)` gives one probability per option, and those form the JSON answers.
- **Speech:** generated after the state; an end token means "silent".
- **Extra heads:** a task head (reach / grasp / throw / push) and a ready head for early exit.

Every add-on starts inert: zero-init gates and LoRA B = 0. So at step 0 the model is exactly the pretrained trunk.

`architecture_viz.html` is an interactive walkthrough of the data flow, token layout, ACT, gates and the questionnaire read-out. Open it in a browser.

## Files

| File | What it does |
|---|---|
| `train_loop_robot.py` | Model, synthetic data, supervised multi-task trainer, eval, `--self-test` |
| `mujoco_env_bridge.py` | MuJoCo arm, state window, policy runner (JSON or action control) |
| `train_live_mujoco.py` | Live stage on the MuJoCo arm: DAgger imitation, then A2C |
| `libero_convert.py` | Converts LIBERO demos (`lerobot/libero`) to state sentences, questionnaire labels and speech |
| `_kaggle.py` | Builds and pushes Kaggle notebooks: `smoke`, `diag`, `qwen`, `live`, `ablate_*`, `status`, `pull` |
| `architecture_viz.html` | Interactive architecture visualisation |

## Running

```bash
pip install -r requirements.txt
python train_loop_robot.py --self-test --model tiny --device cpu      # fast checks
python train_loop_robot.py --model qwen --device cuda --dtype fp16 --trainable lora \
    --loops 6 --carry --questions auto --untie-lm-head --steps 4500 --out robot.pt
python train_live_mujoco.py --model qwen --checkpoint robot.pt --trainable lora \
    --freeze-lm-head --headless
python mujoco_env_bridge.py --model qwen --trainable lora --checkpoint robot.pt  # watch it
```

Full training runs were done on Kaggle (2× T4, fp16) through `_kaggle.py`. For the exact flags, see `QWEN_ARGS` and `live_argv` in that file.

## Results so far (measured)

**Supervised, synthetic only (v2, 4,500 steps, about 2.1 h on T4):**
- JSON-decoded action accuracy 0.898.
- axis_x 0.88, axis_y 0.90. Gripper, intent, height and speed about 0.99.
- ACT settles at about 1 cycle on both easy and hard windows, so the loop is not yet doing useful work.

**Supervised with interface variation + 30% LIBERO (v3, from scratch):**
- JSON action accuracy **0.219**; 0.160 on held-out wordings.
- axis_x/axis_y 0.54/0.45 and speed 0.45; the question loss stayed near ln 3.
- This is a regression. 1,500-step ablations (`_kaggle.py ablate_base|ablate_vary|ablate_libero`) are running to find its cause.

**Live MuJoCo stage, from the v2 checkpoint:**
- Baseline mean distance to goal 0.083 m with 61.7% reach; the expert's ceiling is 0.086 m and 62.1%.
- DAgger imitation makes it worse: 0.850 m, 0% reach, even though pool accuracy was 88–96%.
- The automatic revert did not restore the baseline. An eval straight after the revert has now been added to diagnose this.

## Notes

- Checkpoints, logs, Kaggle staging folders and converted datasets are git-ignored.
- `--self-test` in each script runs the correctness checks: bit-identity at init, ACT mass, option shuffle round-trip, LIBERO batching and others.
