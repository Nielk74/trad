#!/usr/bin/env python3
"""
One-time setup: create isolated venvs for voice clone workers.

Usage:
    python workers/setup_clone_venvs.py              # create both
    python workers/setup_clone_venvs.py --tier small
    python workers/setup_clone_venvs.py --tier medium
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VENV_CONFIGS = {
    "small": {
        "venv_dir": os.path.join(PROJECT_ROOT, "venvs", "clone_small"),
        "requirements": os.path.join(PROJECT_ROOT, "requirements", "requirements-clone-small.txt"),
        "description": "Qwen3-TTS 0.6B (CPU, voice cloning for small tier)",
    },
    "medium": {
        "venv_dir": os.path.join(PROJECT_ROOT, "venvs", "clone_medium"),
        "requirements": os.path.join(PROJECT_ROOT, "requirements", "requirements-clone-medium.txt"),
        "description": "Qwen3-TTS 1.7B (GPU, voice cloning for medium/high tier)",
    },
}


def setup_tier(tier: str, force: bool = False) -> None:
    cfg = VENV_CONFIGS[tier]
    venv_dir = cfg["venv_dir"]
    req_file = cfg["requirements"]
    python = os.path.join(venv_dir, "bin", "python")

    if os.path.exists(python) and not force:
        print(f"[{tier}] Venv already exists at {venv_dir} (use --force to recreate)")
        return

    print(f"\n[{tier}] {cfg['description']}")
    print(f"[{tier}] Creating venv at {venv_dir} ...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

    print(f"[{tier}] Upgrading pip ...")
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)

    print(f"[{tier}] Installing {os.path.basename(req_file)} ...")
    # Small tier uses CPU-only PyTorch — install from the CPU wheel index
    pip_cmd = [python, "-m", "pip", "install", "-r", req_file]
    if tier == "small":
        pip_cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    subprocess.run(pip_cmd, check=True)

    print(f"[{tier}] Done — worker Python: {python}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=["small", "medium", "all"], default="all",
                        help="Which venv to create (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Recreate venv even if it already exists")
    args = parser.parse_args()

    tiers = ["small", "medium"] if args.tier == "all" else [args.tier]
    for tier in tiers:
        setup_tier(tier, force=args.force)

    print("\nSetup complete.")
    print("Start the app normally — clone workers launch automatically on first use.")


if __name__ == "__main__":
    main()
