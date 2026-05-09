"""
train_defense.py
----------------
Perturbation Rectifying Network (PRN) — a lightweight U-Net style denoiser
trained to undo a specific UAP while preserving classifier accuracy.

Public API
----------
    PerturbationRectifyingNetwork
        The denoiser architecture.

    train_defense(model, train_loader, uap_noise, target_idx, ...)
        Training loop; returns the trained rectifier.

Example
-------
    from network_init import setup
    from train_attack import generate_targeted_uap
    from train_defense import train_defense
    import torch

    model, train_loader, _, _, _ = setup()

    TARGET_IDX = 20
    uap_noise = torch.load("targeted_uap_noise_gender_5epochs_10xi.pt")

    defense_net = train_defense(model, train_loader, uap_noise, TARGET_IDX, epochs=5)
    torch.save(defense_net, "defense_gender_5epochs_10xi.pt")
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class PerturbationRectifyingNetwork(nn.Module):
    """Lightweight encoder-decoder denoiser with a single skip connection.

    Architecture summary
    --------------------
    Encoder:
        Conv(3→64) → Conv(64→64) → MaxPool   [feature map c2 saved for skip]
        Conv(64→128) → Conv(128→128)

    Decoder:
        Upsample → Cat(c2) → Conv(192→64) → Conv(64→3) → Sigmoid

    The Sigmoid output guarantees predictions stay in [0, 1].
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Skip connection: upsampled c4 (128) concatenated with c2 (64) → 192
        self.conv5 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        p1 = self.pool(c2)
        c3 = self.relu(self.conv3(p1))
        c4 = self.relu(self.conv4(c3))

        # Decoder
        u1 = self.up(c4)
        # Align spatial dimensions (handles odd input sizes after pooling)
        if u1.shape != c2.shape:
            u1 = nn.functional.interpolate(u1, size=c2.shape[2:])

        merged = torch.cat([u1, c2], dim=1)
        c5 = self.relu(self.conv5(merged))
        return self.sigmoid(self.conv6(c5))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_defense(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    uap_noise: torch.Tensor,
    target_idx: int,
    epochs: int = 5,
    lr: float = 1e-3,
    pixel_weight: float = 1.0,
    cls_weight: float = 0.1,
) -> PerturbationRectifyingNetwork:
    """Train the PRN to remove *uap_noise* and restore correct predictions.

    Loss = pixel_weight * MSE(rectified, clean)
         + cls_weight   * CrossEntropy(model(rectified)[:, target_idx], true_label)

    The backbone *model* is frozen throughout.

    Args:
        model:         Frozen pixel-space classification model.
        train_loader:  DataLoader over the training set.
        uap_noise:     The pre-computed UAP tensor, shape (1, C, H, W).
        target_idx:    Attribute index that was attacked.
        epochs:        Number of training epochs.
        lr:            Adam learning rate.
        pixel_weight:  Weight for the pixel reconstruction loss.
        cls_weight:    Weight for the classification (semantic) loss.

    Returns:
        Trained PerturbationRectifyingNetwork on CUDA.
    """
    device_ids = list(range(torch.cuda.device_count()))
    main_device = torch.device("cuda:0")

    # ---- Freeze backbone ----
    for param in model.parameters():
        param.requires_grad = False

    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.to(main_device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=main_device)
    model.eval()

    # ---- Build rectifier ----
    rectifier = PerturbationRectifyingNetwork().to(main_device)
    rectifier = nn.DataParallel(rectifier, device_ids=device_ids, output_device=main_device)
    rectifier.train()

    optimizer = optim.Adam(rectifier.parameters(), lr=lr)
    criterion_pixel = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    print("Starting Defense Training...")
    t0 = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        correct_recovered = 0
        total_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(main_device)
            labels = labels.to(main_device)
            target_labels = labels[:, target_idx].long()

            # Apply attack (no grad needed here)
            with torch.no_grad():
                images_adv = torch.clamp(images + uap_noise, 0, 1)

            # Rectify
            images_rectified = rectifier(images_adv)

            # Pixel loss
            loss_pixel = criterion_pixel(images_rectified, images)

            # Semantic / classification loss
            outputs_rect = model(images_rectified)
            outputs_rect = torch.stack(outputs_rect).permute(1, 0, 2)   # (B, attrs, 2)
            loss_cls = criterion_cls(outputs_rect[:, target_idx, :], target_labels)

            loss = pixel_weight * loss_pixel + cls_weight * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Book-keeping
            total_loss += loss.item()
            preds = torch.argmax(outputs_rect[:, target_idx, :], dim=1)
            correct_recovered += (preds == target_labels).sum().item()
            total_samples += images.size(0)

            if i % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | "
                    f"Loss={loss.item():.4f} | elapsed={elapsed:.1f}s"
                )

        acc = 100.0 * correct_recovered / total_samples
        avg_loss = total_loss / len(train_loader)
        print(
            f"==> Epoch {epoch+1} complete — "
            f"Avg Loss={avg_loss:.4f} | Defense Accuracy={acc:.2f}%"
        )

    # Return the unwrapped module so callers don't need DataParallel boilerplate
    return rectifier.module if isinstance(rectifier, nn.DataParallel) else rectifier
