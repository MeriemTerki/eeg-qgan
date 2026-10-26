"""Training the autoencoder, with its acceptance gates enforced.

Audit items E3 and F10.

E3: the gate outcome must decide something. Previously the reconstruction test was
computed, stored, and then excluded from the pass flag, so ten folds with a normalised
error above one were all recorded as passing and the pipeline continued on a decoder
that explained none of the signal variance. Here :func:`train_autoencoder` returns a
verdict and the caller must act on it; the policy is a configuration choice between
stopping and recording an exclusion, and there is no path that merely prints.

F10: the autoencoder must not see the outer test fold. It is trained on training-fold
windows only, so the codes the generator learns and the windows the detector is tested
on come from disjoint patients. Training it on all available windows would leak test
information into the representation and inflate the downstream scores of every arm
equally, which is subtle because it would not change the between-arm comparison while
still invalidating the absolute numbers.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from eeg_qgan.evaluation.gates import AEGateFailure, GateReport, evaluate_ae_gates
from eeg_qgan.models.autoencoder import MultichannelAE
from eeg_qgan.seeding import spawn_rng
from eeg_qgan.training.stft import multi_resolution_stft_loss


def build_autoencoder(cfg: dict, n_channels: int) -> MultichannelAE:
    ae_cfg = cfg.get("autoencoder", {}) or {}
    return MultichannelAE(
        n_channels=int(n_channels),
        sample_length=int(cfg["sample_length"]),
        latent_dim=int(cfg["latent_dim"]),
        bottleneck=str(ae_cfg.get("bottleneck", "temporal")),
        widths=tuple(ae_cfg.get("widths", (64, 128, 256))),
        denoise_std=float(ae_cfg.get("denoise_std", 0.01)),
        arch=str(ae_cfg.get("arch", "v2")),
        blocks_per_stage=int(ae_cfg.get("blocks_per_stage", 1)),
        stem_kernel=int(ae_cfg.get("stem_kernel", 7)),
        attention=bool(ae_cfg.get("attention", True)),
    )


def _reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    stft_weight: float,
    l1_weight: float,
) -> torch.Tensor:
    """MSE for energy, L1 for sharpness, multi-resolution STFT for band structure."""
    loss = nn.functional.mse_loss(recon, target)
    if l1_weight > 0:
        loss = loss + float(l1_weight) * nn.functional.l1_loss(recon, target)
    if stft_weight > 0:
        loss = loss + float(stft_weight) * multi_resolution_stft_loss(recon, target)
    return loss


def train_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: dict,
    device: str = "cpu",
    fold_label: str = "",
) -> tuple[MultichannelAE, dict]:
    """Fit the autoencoder on training-fold windows and score it against the gates.

    Args:
        x_train: training windows, shape ``(N, C, T)``. Must exclude the outer test fold.
        x_val: held-out windows for the gates, from the same training fold.
        cfg: resolved configuration.
        device: torch device.
        fold_label: identifier used in messages.

    Returns:
        The trained autoencoder and a metadata dict whose ``gates`` entry carries the
        verdict. When a gate fails the behaviour follows
        ``autoencoder.on_gate_failure``: ``raise`` stops the run, ``exclude`` marks the
        fold so downstream analysis can drop it.
    """
    ae_cfg = cfg.get("autoencoder", {}) or {}
    t0 = time.perf_counter()

    ae = build_autoencoder(cfg, n_channels=int(x_train.shape[1])).to(device)
    epochs = int(ae_cfg.get("epochs", cfg.get("ae_epochs", 300)))
    batch_size = int(ae_cfg.get("batch_size", 128))
    lr = float(ae_cfg.get("lr", 1e-3))
    weight_decay = float(ae_cfg.get("weight_decay", 1e-5))
    stft_weight = float(ae_cfg.get("stft_weight", 0.05))
    l1_weight = float(ae_cfg.get("l1_weight", 0.05))
    patience = int(ae_cfg.get("patience", 40))
    grad_clip = float(ae_cfg.get("grad_clip", 1.0))

    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    # ae_v5c probe evidence: with ReduceLROnPlateau the error was still falling when the
    # 300-epoch budget ran out (0.4759 at epoch 165 down to 0.4492 at epoch 300), so the
    # reported reconstruction was a budget limit rather than a capacity limit. A cosine
    # decay to a small floor spends the last epochs refining instead of waiting for a
    # plateau that never arrived.
    schedule_mode = str(ae_cfg.get("lr_schedule", "cosine")).lower()
    warmup_epochs = int(ae_cfg.get("warmup_epochs", 5))
    min_lr_factor = float(ae_cfg.get("min_lr_factor", 0.02))

    def _lr_factor(epoch_index: int) -> float:
        """Multiplicative factor for epoch ``epoch_index`` counted from zero."""
        if schedule_mode == "plateau":
            return 1.0
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        span = max(1, epochs - warmup_epochs)
        progress = min(1.0, (epoch_index - warmup_epochs) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    plateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
        if schedule_mode == "plateau"
        else None
    )
    cosine_schedule = (
        torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor) if plateau is None else None
    )

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_va_np = np.asarray(x_val, dtype=np.float32)
    # Optional cap so CPU/GPU AE training finishes in hours, not days, on TUH-scale folds.
    # Do not seed this draw with Python's ``hash()``: CPython randomises string hashing
    # per process, so a 80k-window subsample would change between SLURM jobs and can
    # push a previously passing fold over the NMSE gate (job 1184337: 0.558 vs 0.55).
    max_train = ae_cfg.get("max_train_windows", None)
    if max_train is not None and len(x_train_np) > int(max_train):
        rng = spawn_rng(int(cfg.get("seed", 42)), fold_label, "ae_subsample")
        keep = rng.choice(len(x_train_np), size=int(max_train), replace=False)
        x_train_np = x_train_np[keep]
        print(
            f"    AE subsample {fold_label}: using {len(x_train_np)} / "
            f"{len(x_train)} train windows (max_train_windows={int(max_train)})",
            flush=True,
        )

    x_tr = torch.as_tensor(x_train_np)
    x_va = torch.as_tensor(x_va_np, device=device)

    best_val = float("inf")
    best_nmse = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    history: list[dict] = []
    log_every = max(1, int(ae_cfg.get("log_every_epochs", 5)))

    print(
        f"    AE train start {fold_label}: n_train={len(x_tr)} n_val={len(x_va_np)} "
        f"epochs≤{epochs} patience={patience} batch={batch_size} device={device}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        ae.train()
        perm_seed = int(
            spawn_rng(int(cfg.get("seed", 42)), fold_label, "ae_epoch", epoch).integers(0, 2**31)
        )
        perm = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(perm_seed))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(x_tr), batch_size):
            batch = x_tr[perm[start : start + batch_size]].to(device)
            recon, _ = ae(batch, denoise_decode=True)
            loss = _reconstruction_loss(
                recon, batch, stft_weight=stft_weight, l1_weight=l1_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        ae.eval()
        with torch.no_grad():
            recon_va, _ = ae(x_va, denoise_decode=False)
            val_loss = float(nn.functional.mse_loss(recon_va, x_va).item())
            val_nmse = float(
                np.mean((recon_va.cpu().numpy() - x_va_np) ** 2)
                / (float(np.var(x_va_np)) + 1e-8)
            )
        if plateau is not None:
            plateau.step(val_nmse)
        else:
            assert cosine_schedule is not None
            cosine_schedule.step()
        train_loss = epoch_loss / max(1, n_batches)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_nmse": val_nmse,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )

        # Select the checkpoint the gate will score: NMSE, not raw MSE.
        if val_nmse < best_nmse - 1e-6:
            best_nmse = val_nmse
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1
            improved = False
            if epochs_without_improvement >= patience:
                print(
                    f"    AE early stop {fold_label}: epoch={epoch}/{epochs} "
                    f"best_val_nmse={best_nmse:.4f} patience={patience}",
                    flush=True,
                )
                break

        if epoch == 1 or epoch % log_every == 0 or improved:
            print(
                f"    AE epoch {epoch}/{epochs} {fold_label}: "
                f"train_loss={train_loss:.4f} val_nmse={val_nmse:.4f} "
                f"best_nmse={best_nmse:.4f} "
                f"no_improve={epochs_without_improvement}/{patience}",
                flush=True,
            )

    if best_state is not None:
        ae.load_state_dict(best_state)

    print(
        f"    AE train done {fold_label}: epochs={len(history)} "
        f"best_val_nmse={best_nmse:.4f} wall={time.perf_counter() - t0:.0f}s "
        f"gate_threshold={float(ae_cfg.get('max_nmse', 0.5))}",
        flush=True,
    )

    report: GateReport = evaluate_ae_gates(
        ae,
        np.asarray(x_val, dtype=np.float32),
        device=device,
        max_nmse=float(ae_cfg.get("max_nmse", 0.5)),
        max_saturation=ae_cfg.get("max_saturation", None),
        min_acf_preservation=ae_cfg.get("min_acf_preservation"),
    )

    policy = str(ae_cfg.get("on_gate_failure", "raise")).lower()
    excluded = False
    if not report.passed:
        if policy == "raise":
            raise AEGateFailure(report, fold_label)
        if policy == "exclude":
            excluded = True
            print(
                f"    EXCLUDING {fold_label}: {report.explain()}. "
                "The fold is recorded as excluded and must be dropped from every "
                "aggregate that depends on the decoder.",
                flush=True,
            )
        else:
            raise ValueError(
                f"autoencoder.on_gate_failure must be 'raise' or 'exclude', got '{policy}'"
            )

    meta = {
        "gates": report.as_dict(),
        "excluded": excluded,
        "exclusion_reason": report.explain() if excluded else None,
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "best_val_nmse": best_nmse,
        "wall_seconds": time.perf_counter() - t0,
        "lr_schedule": schedule_mode,
        "architecture": ae.describe(),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "history": _trim_training_history(history, cfg),
    }
    return ae, meta


def _trim_training_history(history: list[dict], cfg: dict) -> list[dict]:
    save_cfg = cfg.get("save", {}) or {}
    if bool(save_cfg.get("full_training_history", False)):
        return history
    keep = int(save_cfg.get("training_history_epochs", 20))
    return history[-keep:] if keep > 0 else history


@torch.no_grad()
def encode_windows(
    ae: MultichannelAE, x: np.ndarray, device: str = "cpu", batch_size: int = 256
) -> np.ndarray:
    """Encode windows to latent codes."""
    ae.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        chunk = torch.as_tensor(
            np.asarray(x[start : start + int(batch_size)], dtype=np.float32), device=device
        )
        out.append(ae.encode(chunk).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def reconstruct(
    ae: MultichannelAE, x: np.ndarray, device: str = "cpu", batch_size: int = 256
) -> np.ndarray:
    """Round-trip windows through the autoencoder."""
    ae.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        chunk = torch.as_tensor(
            np.asarray(x[start : start + int(batch_size)], dtype=np.float32), device=device
        )
        recon, _ = ae(chunk, denoise_decode=False)
        out.append(recon.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
