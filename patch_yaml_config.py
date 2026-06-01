from pathlib import Path
import re

p = Path("anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py")
text = p.read_text()
original = text

# 1) Agregar import yaml
if "import yaml" not in text:
    text = text.replace("import argparse", "import argparse\nimport yaml", 1)
    print("import yaml: 1")

# 2) Agregar funciones para cargar YAML
if "def load_yaml_config" not in text:
    marker = "os.makedirs(VIDEO_DIR, exist_ok=True)"
    insert = r'''

def _resolve_project_path(path_value):
    """Resolve relative paths with respect to PROJECT_ROOT."""
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)


def load_yaml_config(config_path=None):
    """Load training configuration from a YAML file."""
    if config_path is None:
        return {}

    config_path = _resolve_project_path(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("YAML config must contain a dictionary at the top level.")

    print(f"Loaded YAML config from: {config_path}")
    return cfg


def apply_yaml_runtime_config(cfg):
    """Apply runtime settings that are not pure PPO hyperparameters."""
    global SAVE_DIR, VIDEO_DIR, device

    output_cfg = cfg.get("output", {})
    if output_cfg.get("save_dir"):
        SAVE_DIR = _resolve_project_path(output_cfg["save_dir"])

    if output_cfg.get("video_dir"):
        VIDEO_DIR = _resolve_project_path(output_cfg["video_dir"])
    elif output_cfg.get("save_dir"):
        VIDEO_DIR = os.path.join(SAVE_DIR, "videos")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    requested_device = cfg.get("device", "auto")

    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    print(f"Configured device: {device}")
    print(f"Configured checkpoint directory: {SAVE_DIR}")
    print(f"Configured video directory: {VIDEO_DIR}")


def merge_yaml_hparams(hparams, cfg):
    """Merge YAML sections into the existing hparams dictionary."""
    training_cfg = cfg.get("training", {})
    checkpoint_cfg = cfg.get("checkpointing", {})
    rendering_cfg = cfg.get("rendering", {})
    project_cfg = cfg.get("project", {})

    merged = dict(hparams)

    merged.update(training_cfg)
    merged.update(checkpoint_cfg)
    merged.update(rendering_cfg)

    if project_cfg.get("run_name") and not merged.get("run_name"):
        merged["run_name"] = project_cfg["run_name"]

    return merged
'''
    text = text.replace(marker, marker + insert, 1)
    print("YAML helper functions: 1")

# 3) Agregar config_path a train_or_sweep
if "config_path=None" not in text:
    text = text.replace(
        "resume_policy=None,\n):",
        "resume_policy=None,\n    config_path=None,\n):",
        1,
    )
    print("train_or_sweep config_path arg: 1")

# 4) Insertar lectura del YAML después del hparams dict y antes de overrides
if "cfg = load_yaml_config(config_path)" not in text:
    marker = "    if overrides:"
    insert = '''    cfg = load_yaml_config(config_path)
    if cfg:
        apply_yaml_runtime_config(cfg)
        hparams = merge_yaml_hparams(hparams, cfg)

    if hparams.get("run_name") and run_name is None:
        run_name = hparams["run_name"]

'''
    text = text.replace(marker, insert + marker, 1)
    print("YAML hparams merge: 1")

# 5) Hacer que Env use fall_threshold desde YAML
text = text.replace(
    "env = Env(fall_threshold=0.35)",
    'env = Env(fall_threshold=float(hparams.get("fall_threshold", 0.35)))',
    1,
)

# 6) Hacer que live también pueda venir de YAML
if "live = bool(live or hparams.get(\"live\", False))" not in text:
    text = text.replace(
        "run_name = getattr(run, \"name\", None) or run_name or \"run\"",
        'run_name = getattr(run, "name", None) or run_name or "run"\n    live = bool(live or hparams.get("live", False))',
        1,
    )
    print("YAML live setting: 1")

# 7) Agregar checkpoint periódico opcional
if "periodic checkpoint from YAML" not in text:
    marker = "            # --- Checkpoint save (with eval video) ----------------------------"
    insert = '''            # --- periodic checkpoint from YAML --------------------------------
            checkpoint_freq = int(hparams.get("checkpoint_freq", 0) or 0)
            if checkpoint_freq > 0 and i_episode > 0 and i_episode % checkpoint_freq == 0:
                periodic_policy_path = os.path.join(
                    SAVE_DIR,
                    f"{run_name}_{i_episode}_periodic_Reward-{running_reward}_policy.pt",
                )
                periodic_optim_path = os.path.join(
                    SAVE_DIR,
                    f"{run_name}_{i_episode}_periodic_Reward-{running_reward}_optimizer.pt",
                )
                torch.save(policy, periodic_policy_path)
                torch.save(optimizer, periodic_optim_path)
                print(f"Saved periodic checkpoint to {SAVE_DIR}")

'''
    text = text.replace(marker, insert + marker, 1)
    print("periodic checkpoint patch: 1")

# 8) Hacer render_on_checkpoint configurable
if 'hparams.get("render_on_checkpoint", True)' not in text:
    text = text.replace(
        "                render_episode(\n                    env,\n                    policy,\n                    i_episode,\n                    log_media=True,\n                    save_plot=True,\n                    prefix=\"checkpoint\",\n                )",
        "                if hparams.get(\"render_on_checkpoint\", True):\n                    render_episode(\n                        env,\n                        policy,\n                        i_episode,\n                        log_media=True,\n                        save_plot=True,\n                        prefix=\"checkpoint\",\n                    )",
        1,
    )
    print("render_on_checkpoint patch: 1")

# 9) Agregar argumento --config al parser
if "--config" not in text:
    insert = '''    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file for training parameters.",
    )

'''
    text = text.replace("    args = parser.parse_args()", insert + "    args = parser.parse_args()", 1)
    print("parser --config: 1")

# 10) Pasar config_path en modo train
if "config_path=args.config" not in text:
    text = text.replace(
        "            resume_policy=args.resume_policy,\n        )",
        "            resume_policy=args.resume_policy,\n            config_path=args.config,\n        )",
        1,
    )
    print("train config_path call: 1")

p.write_text(text)

if text == original:
    print("No changes made. It may already be patched.")
else:
    print("Done. YAML config support added.")
