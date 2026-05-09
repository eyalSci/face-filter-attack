"""
network_init.py
---------------
Handles all one-time setup:
  - CelebA data-list generation
  - Model loading (ResNet-50 + pixel-space NormalizeLayer wrapper)
  - DataLoader construction
  - Attribute name extraction

Usage
-----
    from network_init import build_model, build_dataloaders, get_attributes

Assumes
-------
  * The face-attribute-prediction repo has been cloned and you are running
    from inside it (i.e. `os.chdir('face-attribute-prediction')` already done).
  * CelebA data lives at ./data/celeba  (img_align_celeba, list_attr_celeba.csv,
    list_eval_partition.csv).
  * A trained checkpoint exists at checkpoints/model_best.pth.tar.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import models          # provided by the face-attribute-prediction repo
from celeba import CelebA  # idem

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = "./data/celeba"
IMG_DIR = os.path.join(ROOT, "img_align_celeba/img_align_celeba")
ATTR_FILE = os.path.join(ROOT, "list_attr_celeba.csv")
PARTITION_FILE = os.path.join(ROOT, "list_eval_partition.csv")
CHECKPOINT_PATH = "checkpoints/model_best.pth.tar"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def create_symlink() -> None:
    """Create the symlink that the repo's CelebA loader expects."""
    link_target = os.path.join(ROOT, "img_align_celeba_png")
    if not os.path.exists(link_target):
        os.symlink(os.path.abspath(IMG_DIR), link_target)
        print(f"Symlink created: {link_target}")
    else:
        print("Symlink already exists, skipping.")


def generate_list_file(partition_id: int, output_name: str) -> None:
    """Write a space-separated attribute list for one CelebA split.

    Args:
        partition_id: 0 = train, 1 = val, 2 = test.
        output_name:  Filename to write inside ROOT.
    """
    print(f"Processing {output_name}...")

    with open(ATTR_FILE, "r") as f_attr, open(PARTITION_FILE, "r") as f_part:
        attr_lines = f_attr.readlines()
        part_lines = f_part.readlines()

    # Skip header rows (CSV has 1 header; original TXT has 2)
    attr_lines = attr_lines[1:] if "image_id" in attr_lines[0].lower() else attr_lines[2:]
    if "image_id" in part_lines[0].lower() or "partition" in part_lines[0].lower():
        part_lines = part_lines[1:]

    with open(os.path.join(ROOT, output_name), "w") as f_out:
        count = 0
        for i, (attr_line, part_line) in enumerate(zip(attr_lines, part_lines)):
            attr_line = attr_line.strip()
            part_line = part_line.strip()

            sep = "," if "," in attr_line else None
            attr_parts = attr_line.split(sep)
            part_parts = part_line.split(sep)

            if not attr_parts or not part_parts or len(part_parts) < 2:
                continue

            fname = attr_parts[0].replace('"', "").replace("'", "")
            p_fname = part_parts[0].replace('"', "").replace("'", "")
            p_id = part_parts[1]

            if fname != p_fname:
                print(f"Mismatch at line {i}: {fname} vs {p_fname}")
                break

            if int(p_id) == partition_id:
                attrs = [x.strip() for x in attr_parts[1:]]
                attrs = ["0" if x == "-1" else "1" for x in attrs]
                f_out.write(f"{fname} {' '.join(attrs)}\n")
                count += 1

    print(f"Saved {count} images to {output_name}")


def prepare_data_lists() -> None:
    """Generate train / val / test attribute list files (idempotent)."""
    create_symlink()
    generate_list_file(0, "train_40_att_list.txt")
    generate_list_file(1, "val_40_att_list.txt")
    generate_list_file(2, "test_40_att_list.txt")
    print("Data preparation complete.")


# ---------------------------------------------------------------------------
# Attribute names
# ---------------------------------------------------------------------------

def get_attributes() -> list[str]:
    """Return the ordered list of CelebA attribute names."""
    with open(ATTR_FILE, "r") as f:
        header = f.readline().strip()
    columns = header.split(",")
    return columns[1:]  # drop image_id column


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NormalizeLayer(nn.Module):
    """Prepend a normalisation step so the model can accept raw [0,1] images.

    This lets attack / defence code operate in pixel-space rather than the
    normalised space the backbone was trained in.
    """

    def __init__(
        self,
        mean: list[float] = (0.485, 0.456, 0.406),
        std: list[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x - self.mean) / self.std


def build_model() -> nn.Module:
    """Load the pretrained ResNet-50 wrapped in a pixel-space normaliser.

    Returns:
        A DataParallel model on CUDA in eval mode.
    """
    # 1. Initialise backbone
    resnet = models.resnet50()
    resnet = nn.DataParallel(resnet).cuda()

    # 2. Load weights
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        resnet.load_state_dict(checkpoint["state_dict"])
        print(f"=> Loaded weights from {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: checkpoint not found at {CHECKPOINT_PATH}")

    # 3. Wrap: NormalizeLayer → ResNet core
    model = nn.Sequential(NormalizeLayer(), resnet.module).cuda()
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size: int = 64,
    num_workers: int = 16,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for CelebA.

    Args:
        batch_size:   Batch size for all three loaders.
        num_workers:  Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    to_tensor = transforms.Compose([transforms.ToTensor()])

    train_dataset = CelebA(ROOT, "train_40_att_list.txt", to_tensor)
    val_dataset = CelebA(ROOT, "val_40_att_list.txt", to_tensor)
    test_dataset = CelebA(ROOT, "test_40_att_list.txt", to_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick-start helper
# ---------------------------------------------------------------------------

def setup(
    batch_size: int = 64,
    num_workers: int = 16,
) -> tuple[nn.Module, DataLoader, DataLoader, DataLoader, list[str]]:
    """One-call convenience wrapper.

    Returns:
        (model, train_loader, val_loader, test_loader, attributes)
    """
    model = build_model()
    train_loader, val_loader, test_loader = build_dataloaders(batch_size, num_workers)
    attributes = get_attributes()
    return model, train_loader, val_loader, test_loader, attributes
