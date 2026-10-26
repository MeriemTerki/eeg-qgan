"""Adversarial latent training with disjoint selection and reporting splits.

Audit item E4 is the reason this module looks different from its predecessor. The old
loop evaluated a validation distance every 250 steps, kept the best checkpoint by that
distance, and then reported the same quantity on the same split::

    val_mmd = float(mmd_rbf_multi(z_va[:n].to(device), g(...)).item())
    ...
    if score < best_score:
        best_g = {...}

The reported number was therefore a minimum over six evaluations rather than an
estimate of held-out distance, so every arm's value was optimistic. The bias does not
cancel across arms, because the number of effective draws differs with how noisy each
arm's trajectory is: the quantum arm's trajectory is the smoothest, so it benefits
least, which biases the comparison against it. The same split was also the
quality-gate reference, so the augmentation selection saw it too. Independently, the
selection distance and the reported distance disagreed by a factor of five to twenty in
the saved records, which is itself evidence the two were measuring different things.

Here ``splits.select`` drives checkpoint selection and ``splits.report`` is untouched
until training has finished. Both are grouped, so neither shares a patient with the
training codes.

Two further fixes:

F12, divergence. The oversized control diverged in every run at the matched learning
rate. The loop now records a convergence verdict from the trajectory, so a diverged arm
is reported as diverged rather than silently entered into a capacity comparison.

F3, estimator. Selection uses the biased differentiable form for speed, which is
harmless because a constant offset does not change an argmin over checkpoints, while
everything reported uses the unbiased estimator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from eeg_qgan.evaluation.mmd import mmd2_unbiased, mmd_rbf_multi_biased
from eeg_qgan.models.critic import build_critic, gradient_penalty
from eeg_qgan.models.registry import build_generator, is_quantum
from eeg_qgan.training.latent_space import LatentTransform


@dataclass
class TrainingReport:
    """Everything about one arm's training run that a reader needs to judge it."""

    arm: str
    n_params: int
    steps: int
    batch_size: int
    lr_generator: float
    lr_critic: float
    loss_mode: str
    wall_seconds: float
    selection_distance: float
    selection_step: int
    history: list[dict] = field(default_factory=list)
    converged: bool = True
    convergence_note: str = ""
    generator_description: dict = field(default_factory=dict)
    critic_description: dict = field(default_factory=dict)
    transform_mode: str = "identity"
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "n_params": self.n_params,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "lr_generator": self.lr_generator,
            "lr_critic": self.lr_critic,
            "loss_mode": self.loss_mode,
            "wall_seconds": self.wall_seconds,
            "selection_distance": self.selection_distance,
            "selection_step": self.selection_step,
            "history": self.history,
            "converged": self.converged,
            "convergence_note": self.convergence_note,
            "generator": self.generator_description,
            "critic": self.critic_description,
            "transform_mode": self.transform_mode,
            **self.extras,
        }


def _training_budget(arm: str, cfg: dict) -> tuple[int, int, int]:
    """Steps, batch size and critic steps for one arm.

    Under an equal budget every arm gets identical values, which is the condition the
    comparison rests on. Only when the budget is explicitly unequal do the per-family
    overrides apply.
    """
    train_cfg = cfg.get("training", {}) or {}
    equal = bool(train_cfg.get("equal_budget", cfg.get("fair_training_budget", True)))
    if equal:
        steps = int(train_cfg.get("generator_steps", cfg.get("generator_update_steps", 1500)))
        batch = int(train_cfg.get("batch_size", cfg.get("adv_batch_size", 128)))
    elif is_quantum(arm):
        steps = int(train_cfg.get("quantum_steps", 5000))
        batch = int(train_cfg.get("quantum_batch_size", 64))
    else:
        steps = int(train_cfg.get("classical_steps", 5000))
        batch = int(train_cfg.get("batch_size", 128))
    n_critic = int(train_cfg.get("n_critic", cfg.get("n_critic", 5)))
    return steps, batch, n_critic


def _learning_rates(arm: str, cfg: dict) -> tuple[float, float]:
    """Generator and critic learning rates.

    ``match_lr`` gives every arm the same generator rate, which is required for a
    capacity comparison to be about capacity. It is also what exposed the divergence
    of the oversized control, since that arm needs a lower rate to be stable.
    """
    train_cfg = cfg.get("training", {}) or {}
    lr_critic = float(train_cfg.get("critic_lr", cfg.get("adv_d_lr", 1e-4)))
    if bool(train_cfg.get("match_lr", cfg.get("fair_match_lr", True))):
        return float(train_cfg.get("generator_lr", 1e-3)), lr_critic
    per_arm = (train_cfg.get("generator_lr_by_arm", {}) or {})
    if arm in per_arm:
        return float(per_arm[arm]), lr_critic
    default = 0.01 if is_quantum(arm) else 1e-3
    return float(train_cfg.get("generator_lr", default)), lr_critic


def _assess_convergence(history: list[dict], threshold: float) -> tuple[bool, str]:
    """Decide whether a trajectory converged, from the selection distances.

    A distance an order of magnitude above the others, or one that grows over the run,
    means the optimiser failed rather than that the model lacks capacity. Saying so is
    the difference between reporting a null result and reporting an artefact.
    """
    values = [
        float(h["selection_distance"])
        for h in history
        if "selection_distance" in h and np.isfinite(h["selection_distance"])
    ]
    if len(values) < 2:
        return True, ""
    best, final, first = min(values), values[-1], values[0]

    if not np.isfinite(best):
        return False, "selection distance became non-finite"
    if best > threshold:
        return False, (
            f"best selection distance {best:.4f} exceeds the divergence threshold "
            f"{threshold:.4f}; treat this arm as failed to train rather than as "
            "evidence about capacity"
        )
    if final > 3.0 * first and final > threshold / 2:
        return False, (
            f"selection distance grew from {first:.4f} to {final:.4f} over training, "
            "which indicates optimiser divergence"
        )
    return True, ""


def train_adversarial(
    z_train: np.ndarray,
    z_select: np.ndarray,
    arm: str,
    cfg: dict,
    device: str = "cpu",
    transform: LatentTransform | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[nn.Module, nn.Module, TrainingReport]:
    """Train one generator arm against a shared critic.

    Args:
        z_train: encoder codes for training, in code space.
        z_select: codes used only to choose the checkpoint. Must be disjoint from
            ``z_train`` and from the reporting split by patient.
        arm: which generator to build.
        cfg: resolved configuration.
        device: torch device for the critic and classical generators.
        transform: the shared latent coordinate system. Applied to both the training
            and selection codes so the critic and the selection metric operate in the
            same space the reported metrics will use.
        rng: randomness for batch sampling.

    Returns:
        The generator with its selected checkpoint loaded, the critic, and the report.
    """
    t0 = time.perf_counter()
    rng = rng if rng is not None else np.random.default_rng(int(cfg.get("seed", 42)))
    train_cfg = cfg.get("training", {}) or {}

    output_scale = float(transform.output_scale) if transform is not None else 1.0
    generator = build_generator(arm, cfg, output_scale=output_scale)
    # PennyLane's analytic simulator runs on CPU; classical arms follow the device.
    generator = generator.cpu() if is_quantum(arm) else generator.to(device)

    loss_mode = str(train_cfg.get("loss", cfg.get("adv_loss", "wgan_gp"))).lower()
    critic = build_critic(int(cfg["latent_dim"]), cfg, loss_mode).to(device)

    steps, batch, n_critic = _training_budget(arm, cfg)
    lr_g, lr_d = _learning_rates(arm, cfg)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d, betas=(0.5, 0.9))
    gp_lambda = float(train_cfg.get("gp_lambda", cfg.get("wgan_gp_lambda", 10.0)))
    eval_every = int(train_cfg.get("eval_every", 250))
    log_every = int(train_cfg.get("log_every", 250))
    divergence_threshold = float(train_cfg.get("divergence_threshold", 0.25))

    # Move both splits into the shared coordinate system exactly once.
    if transform is not None:
        z_tr_np = np.asarray(transform.to_generator_space(z_train), dtype=np.float32)
        z_sel_np = np.asarray(transform.to_generator_space(z_select), dtype=np.float32)
    else:
        z_tr_np = np.asarray(z_train, dtype=np.float32)
        z_sel_np = np.asarray(z_select, dtype=np.float32)

    z_tr = torch.as_tensor(z_tr_np)
    z_sel = torch.as_tensor(z_sel_np)

    best_distance = float("inf")
    best_state: dict | None = None
    best_step = -1
    history: list[dict] = []

    for step in range(1, steps + 1):
        # ---- critic ----
        for _ in range(n_critic):
            idx = torch.as_tensor(
                rng.integers(0, len(z_tr), size=min(batch, len(z_tr))), dtype=torch.long
            )
            real = z_tr[idx].to(device)
            with torch.no_grad():
                fake = generator(generator.sample_noise(real.size(0), device)).to(device)
            critic.zero_grad(set_to_none=True)
            if loss_mode == "wgan_gp":
                loss_d = (
                    critic(fake).mean()
                    - critic(real).mean()
                    + gradient_penalty(critic, real, fake, gp_lambda)
                )
            else:
                loss_d = (
                    torch.nn.functional.relu(1.0 - critic(real)).mean()
                    + torch.nn.functional.relu(1.0 + critic(fake)).mean()
                )
            loss_d.backward()
            opt_d.step()

        # ---- generator ----
        n_batch = min(batch, len(z_tr))
        fake = generator(generator.sample_noise(n_batch, device)).to(device)
        generator.zero_grad(set_to_none=True)
        loss_g = -critic(fake).mean()
        loss_g.backward()
        opt_g.step()

        # ---- checkpoint selection, on the selection split only ----
        log_step = (step % log_every == 0) or step == 1 or step == steps
        eval_step = (step % eval_every == 0) or step == steps
        selection_distance = None
        if eval_step:
            with torch.no_grad():
                n_eval = min(len(z_sel), int(train_cfg.get("eval_samples", 512)))
                sel_idx = torch.as_tensor(
                    rng.choice(len(z_sel), size=n_eval, replace=False), dtype=torch.long
                )
                real_eval = z_sel[sel_idx].to(device)
                fake_eval = generator(generator.sample_noise(n_eval, device)).to(device)
                # Biased form is fine for an argmin over checkpoints; every reported
                # number uses the unbiased estimator.
                selection_distance = float(mmd_rbf_multi_biased(real_eval, fake_eval).item())

            if selection_distance < best_distance:
                best_distance = selection_distance
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}

        if log_step or eval_step:
            entry: dict = {
                "step": step,
                "critic_loss": float(loss_d.item()),
                "generator_loss": float(loss_g.item()),
            }
            if selection_distance is not None:
                entry["selection_distance"] = selection_distance
            history.append(entry)

        if step % log_every == 0 or step == 1 or step == steps:
            msg = f"      {arm} step {step}/{steps}"
            if history and "selection_distance" in history[-1]:
                msg += f" selection_distance={history[-1]['selection_distance']:.4f}"
            print(msg, flush=True)

    if best_state is not None:
        generator.load_state_dict(best_state)

    converged, note = _assess_convergence(history, divergence_threshold)
    if not converged:
        print(f"      WARNING: {arm} did not converge: {note}", flush=True)

    report = TrainingReport(
        arm=arm,
        n_params=int(generator.count_params()),
        steps=steps,
        batch_size=batch,
        lr_generator=lr_g,
        lr_critic=lr_d,
        loss_mode=loss_mode,
        wall_seconds=time.perf_counter() - t0,
        selection_distance=best_distance,
        selection_step=best_step,
        history=history,
        converged=converged,
        convergence_note=note,
        generator_description=generator.describe(),
        critic_description=critic.describe(),
        transform_mode=transform.mode if transform is not None else "identity",
    )

    if is_quantum(arm):
        quantum_cfg = cfg.get("quantum", {}) or {}
        if bool(quantum_cfg.get("measure_entanglement", True)):
            try:
                # Measured after the selected checkpoint is loaded, so the number
                # describes the circuit that is actually evaluated.
                report.extras["mean_entanglement"] = generator.mean_entanglement(
                    n_samples=int(quantum_cfg.get("entanglement_samples", 256))
                )
            except Exception as exc:
                print(f"      WARNING: entanglement measurement failed: {exc}", flush=True)
        if bool(quantum_cfg.get("measure_coupling", True)):
            try:
                report.extras["coordinate_coupling"] = generator.coordinate_coupling(
                    n_probes=int(quantum_cfg.get("coupling_probes", 32))
                )
            except Exception as exc:
                print(f"      WARNING: coupling measurement failed: {exc}", flush=True)

    return generator, critic, report


@torch.no_grad()
def sample_latent_codes(
    generator: nn.Module,
    n: int,
    device: str = "cpu",
    transform: LatentTransform | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Draw ``n`` codes and return them in encoder-code space, ready for decoding.

    The inverse transform is applied exactly once and only when a transform is in
    force. Applying it unconditionally is what rescaled the non-adversarial arm's
    samples a second time, since that arm was never transformed to begin with.
    """
    generator.eval()
    chunks: list[np.ndarray] = []
    remaining = int(n)
    while remaining > 0:
        take = min(remaining, int(batch_size))
        z = generator(generator.sample_noise(take, device))
        chunks.append(z.detach().cpu().numpy())
        remaining -= take

    codes = np.concatenate(chunks, axis=0)[: int(n)]
    if transform is not None:
        codes = np.asarray(transform.to_code_space(codes), dtype=np.float32)
    return codes.astype(np.float32)


@torch.no_grad()
def decode_codes(
    ae: nn.Module,
    codes: np.ndarray,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """Decode latent codes into windows."""
    ae.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(codes), int(batch_size)):
        chunk = torch.as_tensor(codes[start : start + int(batch_size)], dtype=torch.float32, device=device)
        out.append(ae.decode(chunk, denoise=False).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def report_distance(
    z_report: np.ndarray,
    codes: np.ndarray,
    transform: LatentTransform | None = None,
    rng: np.random.Generator | None = None,
    n_max: int = 1024,
) -> dict:
    """Unbiased held-out distance, measured in the shared coordinate system.

    Both sides are mapped through the same transform, so the mixture arm and the
    adversarial arms are finally scored in one coordinate system rather than two.
    """
    if transform is not None:
        ref = np.asarray(transform.to_generator_space(z_report), dtype=np.float32)
        cand = np.asarray(transform.to_generator_space(codes), dtype=np.float32)
    else:
        ref = np.asarray(z_report, dtype=np.float32)
        cand = np.asarray(codes, dtype=np.float32)

    from eeg_qgan.evaluation.mmd import mmd_report

    rep = mmd_report(ref, cand, n_max=n_max, rng=rng)
    out = rep.as_dict()
    out["space"] = transform.mode if transform is not None else "identity"
    out["mmd2_raw_code_space"] = mmd2_unbiased(
        np.asarray(z_report, dtype=np.float32), np.asarray(codes, dtype=np.float32)
    )
    return out
