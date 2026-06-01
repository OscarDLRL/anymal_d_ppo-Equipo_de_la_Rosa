from pathlib import Path
import re

p = Path("anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py")
text = p.read_text()
original = text

# Imports necesarios
if "import os" not in text:
    text = text.replace("import argparse", "import argparse\nimport os", 1)

if "import re" not in text:
    text = text.replace("import argparse", "import argparse\nimport re", 1)

# 1) Agregar resume_policy a train_or_sweep
if "resume_policy=None" not in text:
    pattern = r"def train_or_sweep\(\s*is_sweep=True,\s*live=False,\s*overrides=None,\s*run_name=None,\s*num_episodes=None\s*\):"
    repl = """def train_or_sweep(
    is_sweep=True,
    live=False,
    overrides=None,
    run_name=None,
    num_episodes=None,
    resume_policy=None,
):"""
    text, n = re.subn(pattern, lambda m: repl, text, count=1)
    print("train_or_sweep patch:", n)

# 2) Cambiar creación del policy
if "Resuming training from policy" not in text:
    pattern = r"(?m)^    policy = Agent\(N_OBS, N_ACT\)\n    optimizer = torch\.optim\.Adam\(policy\.parameters\(\), lr=hparams\[[\"']lr[\"']\]\)"
    repl = """    if resume_policy:
        if resume_policy.lower() in ("best", "auto"):
            resume_policy = pick_latest_checkpoint()

        print(f"Resuming training from policy: {resume_policy}")

        policy = torch.load(resume_policy, map_location="cpu", weights_only=False)
        policy.train()

        optimizer = torch.optim.Adam(policy.parameters(), lr=hparams["lr"])

        optim_candidate = resume_policy.replace("_policy.pt", "_optimizer.pt")
        if os.path.exists(optim_candidate):
            try:
                old_optimizer = torch.load(
                    optim_candidate,
                    map_location="cpu",
                    weights_only=False,
                )
                optimizer.load_state_dict(old_optimizer.state_dict())

                for group in optimizer.param_groups:
                    group["lr"] = hparams["lr"]

                print(f"Loaded optimizer state from: {optim_candidate}")
            except Exception as e:
                print(f"Could not load optimizer, using new Adam optimizer: {e}")

        m = re.search(
            r"Reward-([-+]?\\d+(?:\\.\\d+)?)_policy\\.pt$",
            os.path.basename(resume_policy),
        )
        start_reward = float(m.group(1)) if m else 100.0

    else:
        policy = Agent(N_OBS, N_ACT)
        optimizer = torch.optim.Adam(policy.parameters(), lr=hparams["lr"])
        start_reward = 0.0"""
    text, n = re.subn(pattern, lambda m: repl, text, count=1)
    print("policy loading patch:", n)

# 3) Inicializar reward desde el checkpoint
if "Initial running reward" not in text:
    pattern = r"(?m)^    running_reward = 0\.0\n    saving_reward = 100\.0"
    repl = """    running_reward = start_reward if resume_policy else 0.0
    saving_reward = start_reward if resume_policy else 100.0

    print(f"Initial running reward: {running_reward}")
    print(f"Initial saving reward: {saving_reward}")"""
    text, n = re.subn(pattern, lambda m: repl, text, count=1)
    print("reward init patch:", n)

# 4) Agregar argumentos al parser
if "--resume-policy" not in text:
    insert = '''    parser.add_argument(
        "--resume-policy",
        type=str,
        default=None,
        help="(train) path to *_policy.pt or 'best' to continue from best checkpoint.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="(train) override number of training episodes.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="(train) override learning rate.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="(train) name prefix for new checkpoints.",
    )

'''
    text = text.replace("    args = parser.parse_args()", insert + "    args = parser.parse_args()", 1)
    print("parser args patch: 1")

# 5) Cambiar llamada del modo train
if "resume_policy=args.resume_policy" not in text:
    pattern = r"(?m)^    if args\.mode == [\"']train[\"']:\n        train_or_sweep\(is_sweep=False, live=args\.live\)"
    repl = """    if args.mode == "train":
        train_overrides = {}

        if args.lr is not None:
            train_overrides["lr"] = args.lr

        train_or_sweep(
            is_sweep=False,
            live=args.live,
            overrides=train_overrides if train_overrides else None,
            run_name=args.run_name,
            num_episodes=args.episodes,
            resume_policy=args.resume_policy,
        )"""
    text, n = re.subn(pattern, lambda m: repl, text, count=1)
    print("train call patch:", n)

p.write_text(text)

if text == original:
    print("No se hicieron cambios. Puede que ya estuviera parcheado.")
else:
    print("Listo: script modificado.")
