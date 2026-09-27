# decision-model-rl

Train decision models through interaction with PufferLib games. The first result is two cooperating Overcooked chefs on one RTX 5080.

Read the [write-up and watch the video](https://rlafuente.com/posts/2026-9-26-training-a-small-decision-model-to-cook.html), or download the [trained weights](https://huggingface.co/rodneyslafuente/decision-model-rl-overcooked).

Each chef sees its own native observation expressed in text, game rules, and eight recent action outcomes. A shared OpenJev policy independently scores six buttons: `stay`, `up`, `down`, `left`, `right`, `interact`. Every selection executes exactly one native action. No pathfinding, scripted subgoals, action macros, or assigned roles.

Checkpoint 220 serves **six soups in 512 ticks**, greedy evaluation, seed 0. This is the best checkpoint on that evaluation seed, not a multi-seed average or a claim of generalization. A separate layout trial did not transfer successfully.

## Run

Tested on Linux x86_64, Python 3.12, and a 16 GB RTX 5080. Install `git`, `gcc`, `curl`, `libGL.so.1`, `libX11.so.6`, and DejaVu Sans fonts. The Python dependencies pin the versions used for the result.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
bash build_env.sh
python download.py
python evaluate.py
python render.py runs/evaluation.json 220 --folder videos
```

The renderer produces an MP4 at the original display cadence (16 game ticks per second) and an HTML replay with exact scoring prompts. It checks every recorded state against the native environment. Files go in `videos/`. For the original recorded evaluation, use `python render.py models/trained/evaluation.json 220 --folder videos`.

`python evaluate.py --adapter base --iteration 0` evaluates the pretrained policy with the same prompt. `--adapter runs/duo/checkpoint-000005 --iteration 5` selects another checkpoint. Evaluation is greedy and uses one seed by default.

## Train

```bash
python train.py --config configs/duo.yaml
```

This starts the cooperative stage from the released single-chef checkpoint 330, as the experiment did. To start the cooperative stage from checkpoint 220 instead, change `initial_adapter` to `models/trained` and use a new `run_dir`. Either starts a fresh optimizer. Existing runs resume automatically from their latest saved optimizer, weights, RNG state, and entropy coefficient. `Ctrl-C` finishes the current update and saves. `--max-updates 1` limits a run; `duration_hours` in YAML sets its time budget. Checkpoints retain the latest three, plus the best evaluation adapter.

To train the single-chef stage from pretrained OpenJev, run `python train_single.py --config configs/single.yaml`. To follow it with cooperation, point `initial_adapter` in the duo config at your single-chef `best-eval-adapter`. These are the final configurations. The original experiment changed prompts, horizons, and exploration settings during development, so this is a runnable continuation recipe, not a promise to reproduce the historical learning curve from one command.

The policy uses rank-16 LoRA on the text backbone and trains the existing three-class NLI head. It normalizes entailment probabilities across candidate actions. Training uses a clipped group-relative policy gradient with discounted reward-to-go, a leave-one-trajectory-out baseline, adaptive entropy, capped observation novelty, and a repeated-no-op penalty. There is no critic or teacher. Cooperative training collects 8,192 agent decisions across eight kitchens per update. The bridge supplies shaping rewards of 0.1 for adding an ingredient, 0.1 for starting cooking, 0.2 for plating, and 1.0 to both chefs for delivery.

Shared-prefix scoring, fused convolutions, and Liger kernels reduce repeated computation. The released checkpoint updated 10,825,728 parameters. Its recorded update peaked at 7.45 GiB allocated VRAM. See `results/` for training metrics and the evaluation summary. The model repository includes the complete 512-tick trace.

## Code

`policy.py` contains the native observation description and NLI policy. `efficient_nli.py` implements differentiable shared-prefix scoring. `exploration_objectives.py` supplies environment-independent exploration and credit assignment. `train.py` collects independent actions from both chefs; `train_single.py` contains the common optimizer and the single-chef training loop. `kitchen.py` and `overcooked_bridge.c` wrap the pinned native environment. `render.py` renders verified traces.

Overcooked is the only supported environment today. For another PufferLib game, supply its native observation/action interface and text description, then adapt the collector and renderer. The optimizer and exploration objectives can be reused. The current observation decoder assumes the 5×5 Cramped Room layout, and the six-action optimization is specific to this example.

```bash
python -m unittest discover -s tests -v
```

## Credits

[OpenJev](https://huggingface.co/AlexWortega/openjev/tree/552759daad712f1af6c4c13dabcb1e047886fc9c/qwen3.5-0.8b-nli-v5), an open implementation of the [Jev decision-model idea](https://typesafe.ai/blog/introducing-system-one-models-and-jev), supplies the pretrained Qwen3.5-0.8B NLI classifier. [PufferLib](https://github.com/PufferAI/PufferLib/tree/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/overcooked) supplies the environment and sprites. Training builds on [LoRA](https://arxiv.org/abs/2106.09685), [PPO](https://arxiv.org/abs/1707.06347), and [GRPO](https://arxiv.org/abs/2402.03300). The write-up contains full references and the experiment's limitations.

MIT license for this repository. Downloaded dependencies and base weights retain their respective licenses.
