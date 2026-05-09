"""
train_attack.py
---------------
Targeted Universal Adversarial Perturbation (UAP) via batched DeepFool.

Public API
----------
    deepfool_batch(x, model, target_idx, target_labels, ...)
        Single-step perturbation for a batch of images.

    generate_targeted_uap(model, train_loader, target_idx, ...)
        Full UAP training loop; returns the saved noise tensor.

Example
-------
    from network_init import setup
    from train_attack import generate_targeted_uap
    import torch

    model, train_loader, _, _, attributes = setup()

    TARGET_IDX = 20  # 'Male'  |  15 = 'Eyeglasses'
    uap_noise = generate_targeted_uap(
        model, train_loader, TARGET_IDX,
        epochs=5, max_iter=50, overshoot=0.02, xi=10,
    )
    torch.save(uap_noise, "targeted_uap_noise_gender_5epochs_10xi.pt")
"""

import time
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DeepFool (batched)
# ---------------------------------------------------------------------------

def deepfool_batch(
    x: torch.Tensor,
    model: nn.Module,
    target_idx: int,
    target_labels: torch.Tensor,
    max_iter: int = 50,
    overshoot: float = 0.02,
) -> torch.Tensor:
    """Compute a minimal perturbation that flips *target_idx* for each image.

    This is a batched implementation of DeepFool for binary classifiers.
    Images that are already misclassified are skipped via an *active_mask*.

    Args:
        x:             Clean images, shape (B, C, H, W), in [0, 1].
        model:         The frozen classification model.
        target_idx:    Which attribute head (0-indexed column) to attack.
        target_labels: The *desired* (flipped) label for each image, shape (B,).
        max_iter:      Maximum number of DeepFool iterations.
        overshoot:     Overshoot coefficient applied to each update step.

    Returns:
        Perturbation delta of the same shape as *x* (detached).
    """
    device = x.device
    B = x.size(0)

    x_adv = x.clone().detach().requires_grad_(True)
    active_mask = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        if not active_mask.any():
            break

        # Forward pass
        outputs = model(x_adv)
        outputs = torch.stack(outputs).permute(1, 0, 2)   # (B, num_attrs, 2)
        logits = outputs[:, target_idx, :]                 # (B, 2)
        preds = torch.argmax(logits, dim=1)

        # Stop updating images that are already fooled
        active_mask = active_mask & (~(preds == target_labels))
        if not active_mask.any():
            break

        # Loss: encourage target class score > other class score
        target_scores = torch.gather(logits, 1, target_labels.unsqueeze(1)).squeeze()
        other_scores = torch.gather(logits, 1, (1 - target_labels).unsqueeze(1)).squeeze()
        diff = target_scores - other_scores

        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss = (diff * active_mask.float()).sum()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            grad_norm_sq = grad.view(B, -1).norm(dim=1) ** 2 + 1e-8
            scale = (torch.abs(diff) + 1e-4) / grad_norm_sq
            scale = scale.view(B, 1, 1, 1)

            perturbation = scale * grad
            mask_view = active_mask.view(B, 1, 1, 1).float()

            x_adv.data += (1 + overshoot) * perturbation * mask_view
            x_adv.grad.zero_()

    return (x_adv - x).detach()


# ---------------------------------------------------------------------------
# UAP training loop
# ---------------------------------------------------------------------------

def generate_targeted_uap(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    target_idx: int,
    epochs: int = 10,
    max_iter: int = 50,
    overshoot: float = 0.1,
    xi: float = 10.0,
    img_shape: tuple[int, int, int] = (3, 218, 178),
) -> torch.Tensor:
    """Train a Universal Adversarial Perturbation (UAP) to flip *target_idx*.

    The algorithm:
      1. For each batch, apply the current delta and check the fooling rate.
      2. Run batched DeepFool on the perturbed batch to get new perturbations.
      3. Aggregate via the batch mean and add to delta.
      4. Project delta onto the L2 ball of radius *xi*.

    Args:
        model:        The (frozen) pixel-space classification model.
        train_loader: DataLoader over the training set.
        target_idx:   Attribute index to flip.
        epochs:       Number of full passes over the training set.
        max_iter:     DeepFool iterations per batch.
        overshoot:    DeepFool overshoot coefficient.
        xi:           L2 norm budget for the UAP.
        img_shape:    (C, H, W) of a single image, used to initialise delta.

    Returns:
        Learned perturbation delta, shape (1, C, H, W), on CUDA.
    """
    device_ids = list(range(torch.cuda.device_count()))
    main_device = torch.device("cuda:0")

    # Unwrap and re-wrap with explicit output_device
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.to(main_device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=main_device)
    model.eval()

    delta = torch.zeros((1, *img_shape), device=main_device)

    print(f"Starting UAP generation (batch_size={train_loader.batch_size}) ...")
    t0 = time.time()

    for epoch in range(epochs):
        epoch_fooled = 0
        epoch_total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(main_device)
            labels = labels.long().to(main_device)
            current_targets = 1 - labels[:, target_idx]  # flip the label

            with torch.no_grad():
                x_check = torch.clamp(images + delta, 0, 1)
                outputs_check = model(x_check)
                outputs_check = torch.stack(outputs_check).permute(1, 0, 2)
                preds_check = torch.argmax(outputs_check[:, target_idx, :], dim=1)
                fooled = (preds_check == current_targets)
                batch_fooled = fooled.sum().item()
                epoch_fooled += batch_fooled
                epoch_total += images.size(0)

            # Generate new perturbations from the current (perturbed) inputs
            x_start = torch.clamp(images + delta, 0, 1)
            batch_perturbations = deepfool_batch(
                x_start, model, target_idx, current_targets, max_iter, overshoot
            )

            # Average batch perturbations and accumulate
            avg_perturbation = batch_perturbations.mean(dim=0, keepdim=True)
            delta = delta + avg_perturbation

            # Project onto L2 ball
            norm = torch.norm(delta)
            if norm > xi:
                delta = delta * (xi / (norm + 1e-8))

            if i % 200 == 0:
                elapsed = time.time() - t0
                batch_fr = 100.0 * batch_fooled / images.size(0)
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | "
                    f"||delta||={norm.item():.2f} | Batch FR={batch_fr:.2f}% | "
                    f"elapsed={elapsed:.1f}s"
                )

        epoch_fr = 100.0 * epoch_fooled / epoch_total
        print(f"===> Epoch {epoch+1} done — Average Fooling Rate = {epoch_fr:.2f}% <===")

    return delta.detach()
