"""
main.py
-------
Command-line entry point for the UAP attack / defense pipeline.

Stages
------
  prepare   — generate CelebA split list files and validate paths
  train     — train the ResNet-50 backbone (delegates to the repo's main.py)
  attack    — generate a targeted Universal Adversarial Perturbation (UAP)
  defense   — train the Perturbation Rectifying Network (PRN)
  evaluate  — run full evaluation and print per-attribute accuracy tables

Run all stages end-to-end
-------------------------
    python main.py --all --target male

Run individual stages
---------------------
    python main.py prepare
    python main.py train
    python main.py attack  --target glasses --epochs 5 --xi 2
    python main.py defense --target glasses --epochs 5
    python main.py evaluate --target glasses

Supported targets
-----------------
    glasses  (attribute index 15, xi=2  by default)
    male     (attribute index 20, xi=10 by default)

Saved artifacts
---------------
    checkpoints/model_best.pth.tar         — backbone weights (after train)
    targeted_uap_<target>_<tag>.pt         — UAP noise tensor
    defense_<target>_<tag>.pt             — trained PRN weights
"""

import argparse
import os
import subprocess
import sys

import torch

# ---------------------------------------------------------------------------
# Attribute configuration
# ---------------------------------------------------------------------------

ATTRIBUTE_CONFIG = {
    "glasses": {
        "index": 15,
        "label": "Eyeglasses",
        "class_names": ["No Glasses", "Glasses"],
        "default_xi": 2.0,
        "default_batch_size_defense": 16,
    },
    "male": {
        "index": 20,
        "label": "Male",
        "class_names": ["Female", "Male"],
        "default_xi": 10.0,
        "default_batch_size_defense": 64,
    },
}


def _artifact_tag(args) -> str:
    """Build a short filename tag from the key hyperparameters."""
    return f"{args.target}_{args.attack_epochs}epochs_{int(args.xi)}xi"


def _uap_path(args) -> str:
    return f"targeted_uap_{_artifact_tag(args)}.pt"


def _defense_path(args) -> str:
    return f"defense_{_artifact_tag(args)}.pt"


# ---------------------------------------------------------------------------
# Stage: prepare
# ---------------------------------------------------------------------------

def stage_prepare(args) -> None:
    """Generate the three CelebA split list files required by the repo."""
    print("\n=== Stage: prepare ===")
    os.chdir("face-attribute-prediction")
    from network_init import prepare_data_lists
    prepare_data_lists()
    print("Preparation complete.\n")


# ---------------------------------------------------------------------------
# Stage: train backbone
# ---------------------------------------------------------------------------

def stage_train(args) -> None:
    """Delegate backbone training to the upstream repo's main.py."""
    print("\n=== Stage: train backbone ===")
    cmd = [
        sys.executable, "main.py",
        "-d", "./data/celeba",
        "--arch", "resnet50",
        "--epochs", str(args.train_epochs),
        "--train-batch", str(args.train_batch),
        "--test-batch", str(args.train_batch),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd="face-attribute-prediction")
    if result.returncode != 0:
        sys.exit(f"Backbone training failed (exit code {result.returncode}).")
    print("Backbone training complete.\n")


# ---------------------------------------------------------------------------
# Stage: attack
# ---------------------------------------------------------------------------

def stage_attack(args) -> None:
    """Generate and save the targeted UAP noise tensor."""
    print(f"\n=== Stage: attack (target={args.target}) ===")

    cfg = ATTRIBUTE_CONFIG[args.target]
    xi = args.xi if args.xi is not None else cfg["default_xi"]

    from network_init import build_model, build_dataloaders
    from train_attack import generate_targeted_uap

    model = build_model()
    train_loader, _, _ = build_dataloaders(batch_size=args.attack_batch)

    uap_noise = generate_targeted_uap(
        model=model,
        train_loader=train_loader,
        target_idx=cfg["index"],
        epochs=args.attack_epochs,
        max_iter=args.max_iter,
        overshoot=args.overshoot,
        xi=xi,
    )

    out_path = _uap_path(args)
    torch.save(uap_noise, out_path)
    print(f"UAP saved → {out_path}\n")


# ---------------------------------------------------------------------------
# Stage: defense
# ---------------------------------------------------------------------------

def stage_defense(args) -> None:
    """Train the PRN denoiser against the pre-computed UAP."""
    print(f"\n=== Stage: defense (target={args.target}) ===")

    cfg = ATTRIBUTE_CONFIG[args.target]
    uap_path = _uap_path(args)

    if not os.path.exists(uap_path):
        sys.exit(
            f"UAP file not found: {uap_path}\n"
            "Run the 'attack' stage first, or pass --uap-path explicitly."
        )

    from network_init import build_model, build_dataloaders
    from train_defense import train_defense

    uap_noise = torch.load(uap_path)
    model = build_model()

    defense_batch = (
        args.defense_batch
        if args.defense_batch is not None
        else cfg["default_batch_size_defense"]
    )
    train_loader, _, _ = build_dataloaders(batch_size=defense_batch)

    defense_net = train_defense(
        model=model,
        train_loader=train_loader,
        uap_noise=uap_noise,
        target_idx=cfg["index"],
        epochs=args.defense_epochs,
    )

    out_path = _defense_path(args)
    torch.save(defense_net, out_path)
    print(f"Defense network saved → {out_path}\n")


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------

def stage_evaluate(args) -> None:
    """Print per-attribute accuracy (clean / attacked / defended) on the test set."""
    print(f"\n=== Stage: evaluate (target={args.target}) ===")

    cfg = ATTRIBUTE_CONFIG[args.target]
    uap_path = _uap_path(args)
    def_path = _defense_path(args)

    for path in (uap_path, def_path):
        if not os.path.exists(path):
            sys.exit(
                f"Required artifact not found: {path}\n"
                "Run attack and defense stages first."
            )

    from network_init import build_model, build_dataloaders, get_attributes
    import numpy as np
    import pandas as pd

    uap_noise  = torch.load(uap_path)
    defense_net = torch.load(def_path).cuda().eval()

    model = build_model()
    _, _, test_loader = build_dataloaders(batch_size=64)
    attributes = get_attributes()

    main_device = torch.device("cuda:0")
    import torch.nn as nn

    if isinstance(model, nn.DataParallel):
        model_inner = model.module
    else:
        model_inner = model
    model_inner = model_inner.to(main_device)
    model = nn.DataParallel(model_inner).cuda()
    model.eval()

    n_attrs = len(attributes)
    correct_orig = torch.zeros(n_attrs)
    correct_adv  = torch.zeros(n_attrs)
    correct_den  = torch.zeros(n_attrs)
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images  = images.to(main_device)
            targets = targets.to(main_device)

            out_orig = torch.stack(model(images)).argmax(dim=2)
            correct_orig += (out_orig.t() == targets).sum(dim=0).cpu()

            adv = torch.clamp(images + uap_noise, 0, 1)
            out_adv = torch.stack(model(adv)).argmax(dim=2)
            correct_adv += (out_adv.t() == targets).sum(dim=0).cpu()

            den = defense_net(adv)
            out_den = torch.stack(model(den)).argmax(dim=2)
            correct_den += (out_den.t() == targets).sum(dim=0).cpu()

            total += images.size(0)
            del images, targets
            torch.cuda.empty_cache()

    acc_orig = correct_orig / total * 100
    acc_adv  = correct_adv  / total * 100
    acc_den  = correct_den  / total * 100

    df = pd.DataFrame(
        np.array([attributes, acc_orig, acc_adv, acc_den]).T,
        columns=["Attribute", "Original (%)", "Attacked (%)", "Defended (%)"],
    ).sort_values("Original (%)", ascending=False)

    print("\n── Per-attribute accuracy ──────────────────────────────")
    print(df.to_string(index=False))
    print()
    print(f"Mean original : {acc_orig.mean():.2f}%")
    print(f"Mean attacked : {acc_adv.mean():.2f}%")
    print(f"Mean defended : {acc_den.mean():.2f}%")

    target_row = df[df["Attribute"] == cfg["label"]]
    if not target_row.empty:
        print(f"\nTarget attribute ({cfg['label']}):")
        print(target_row.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UAP Attack & Defense pipeline for CelebA binary classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Global options ───────────────────────────────────────────────────
    parser.add_argument(
        "--target",
        choices=list(ATTRIBUTE_CONFIG.keys()),
        default="male",
        help="Facial attribute to attack / defend (default: male).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all stages sequentially: prepare → train → attack → defense → evaluate.",
    )

    # ── Backbone training ────────────────────────────────────────────────
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-batch",  type=int, default=32)

    # ── Attack ───────────────────────────────────────────────────────────
    parser.add_argument("--attack-epochs", type=int, default=5)
    parser.add_argument("--attack-batch",  type=int, default=64)
    parser.add_argument("--max-iter",      type=int, default=50,
                        help="DeepFool iterations per batch.")
    parser.add_argument("--overshoot",     type=float, default=0.02,
                        help="DeepFool overshoot coefficient.")
    parser.add_argument("--xi",            type=float, default=None,
                        help="L2 norm budget for the UAP (default: per-target).")

    # ── Defense ──────────────────────────────────────────────────────────
    parser.add_argument("--defense-epochs", type=int, default=5)
    parser.add_argument("--defense-batch",  type=int, default=None,
                        help="Batch size for defense training (default: per-target).")

    # ── Positional stage selector ─────────────────────────────────────────
    parser.add_argument(
        "stage",
        nargs="?",
        choices=["prepare", "train", "attack", "defense", "evaluate"],
        help="Single stage to run (omit when using --all).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Default xi from target config if not specified
    if args.xi is None:
        args.xi = ATTRIBUTE_CONFIG[args.target]["default_xi"]

    # Ensure we are inside the repo for all import-dependent stages
    repo_dir = "face-attribute-prediction"

    stages = {
        "prepare":  stage_prepare,
        "train":    stage_train,
        "attack":   stage_attack,
        "defense":  stage_defense,
        "evaluate": stage_evaluate,
    }

    if args.all:
        for name, fn in stages.items():
            if name != "train" or os.path.exists(
                os.path.join(repo_dir, "checkpoints", "model_best.pth.tar")
            ):
                fn(args)
            else:
                fn(args)
    elif args.stage:
        if args.stage not in ("prepare", "train"):
            # All stages after prepare need to be inside the repo
            if os.path.isdir(repo_dir):
                os.chdir(repo_dir)
        stages[args.stage](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
