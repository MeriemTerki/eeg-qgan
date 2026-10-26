"""Style-qGAN quantum latent generator.

The circuit itself is unchanged from the audited implementation, because the audit
confirmed it does what the manuscript says: the encoding is scalar so each noise
coordinate reaches exactly one qubit, the four entanglement topologies differ only
in their controlled-Z wiring while all carrying exactly 128 parameters, and the
Meyer-Wallach measure is the standard pure-state form averaged over the noise
distribution.

What changed is the reporting and the honesty of the comparison.

F5: when the latent width exceeds the qubit count a dense linear layer expands the
readouts, and in the ablation configuration that layer held 288 of 416 parameters.
:meth:`parameter_breakdown` now separates circuit parameters from classical ones, and
:attr:`is_pure_quantum` states plainly whether the arm is all-quantum.

F7: without entanglement each readout is a deterministic one-dimensional function of
a single noise coordinate, so the product topology cannot represent any dependence at
all. :meth:`coordinate_coupling` measures this directly, which is what turns an
untestable claim into a reported number.

F6: the noise width is no longer implicitly the qubit count. An optional fixed random
projection lets the circuit accept wider noise, so the quantum arm can be given the
same input dimension as its twin.
"""
from __future__ import annotations

from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn

from eeg_qgan.models.base import LatentGenerator

EntanglementName = Literal["none", "linear", "ring", "full"]
AnsatzName = Literal["style_qgan", "efficient_su2"]


def entangler_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    """Controlled-Z wiring for one topology.

    All topologies keep the single-qubit rotation layers identical, so the parameter
    count is invariant and only connectivity varies:

      * ``none``: no entangling gates, a product circuit;
      * ``linear``: nearest-neighbour chain without wrap-around;
      * ``ring``: nearest-neighbour chain with wrap-around, the default;
      * ``full``: all-to-all.
    """
    if n_qubits < 2 or entanglement == "none":
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if entanglement == "ring":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    raise ValueError(f"Unknown entanglement topology '{entanglement}'")


def _resolve_device(n_qubits: int, cfg: dict) -> tuple[str, str]:
    """Pick a simulator and a differentiation method.

    ``default.qubit`` with backpropagation gives analytic expectation values, which
    is what the principal tables use. Finite-shot evaluation is a separate,
    explicitly labelled check.
    """
    if bool(cfg.get("quantum_prefer_lightning", False)):
        for name in ("lightning.qubit", "default.qubit"):
            try:
                qml.device(name, wires=n_qubits)
                return name, "adjoint" if "lightning" in name else "backprop"
            except Exception:
                continue
    return "default.qubit", "backprop"


def _style_layer(n_qubits: int, entanglement: str):
    """One re-uploading block.

    Per qubit: a data-dependent ``RY(w * u + b)``, then trainable ``RY`` and ``RZ``,
    then the topology's controlled-Z gates. Four trainable scalars per qubit per
    layer, hence ``4 * n_layers * n_qubits`` parameters.
    """
    pairs = entangler_pairs(n_qubits, entanglement)

    def layer(inputs, w_style, b_style, theta, phi, idx):
        for i in range(n_qubits):
            qml.RY(w_style[idx, i] * inputs[..., i] + b_style[idx, i], wires=i)
        for i in range(n_qubits):
            qml.RY(theta[idx, i], wires=i)
            qml.RZ(phi[idx, i], wires=i)
        for a, b in pairs:
            qml.CZ(wires=[a, b])

    return layer


def build_style_layer(
    n_qubits: int,
    n_layers: int,
    cfg: dict | None = None,
    entanglement: str = "ring",
    shots: int | None = None,
) -> qml.qnn.TorchLayer:
    cfg = cfg or {}
    dev_name, diff = _resolve_device(n_qubits, cfg)
    if shots is not None:
        # Finite-shot evaluation needs parameter-shift gradients.
        dev = qml.device(dev_name, wires=n_qubits, shots=int(shots))
        diff = "parameter-shift"
    else:
        dev = qml.device(dev_name, wires=n_qubits)
    layer_fn = _style_layer(n_qubits, entanglement)

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs, w_style, b_style, theta, phi):
        for idx in range(n_layers):
            layer_fn(inputs, w_style, b_style, theta, phi, idx)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    shapes = {
        "w_style": (n_layers, n_qubits),
        "b_style": (n_layers, n_qubits),
        "theta": (n_layers, n_qubits),
        "phi": (n_layers, n_qubits),
    }
    return qml.qnn.TorchLayer(circuit, shapes)


def build_efficient_su2_layer(
    n_qubits: int, n_layers: int, cfg: dict | None = None
) -> qml.qnn.TorchLayer:
    """Alternative ansatz retained for the depth ablation."""
    cfg = cfg or {}
    dev_name, diff = _resolve_device(n_qubits, cfg)
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs, layer_weights):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(inputs[..., i] * 0.5 + layer_weights[layer, i, 0], wires=i)
                qml.RZ(layer_weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CZ(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"layer_weights": (n_layers, n_qubits, 2)})


class StyleQuantumGenerator(LatentGenerator):
    """Parameterised quantum circuit generating latent codes.

    Args:
        latent_dim: width of the latent code.
        n_qubits: circuit width. Must not exceed ``latent_dim``.
        n_layers: number of re-uploading blocks.
        entanglement: controlled-Z topology.
        ansatz: circuit family.
        noise_dim: input width. Defaults to ``n_qubits``. When larger, a fixed
            untrained projection maps noise onto the qubits, which exists so the
            quantum arm can be given the same noise width as its classical twin
            without adding trainable capacity.
        shots: finite-shot evaluation when set; analytic when None.
        output_scale: multiplier applied to the bounded readout.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        n_qubits: int | None = None,
        n_layers: int = 4,
        entanglement: EntanglementName = "ring",
        ansatz: AnsatzName = "style_qgan",
        noise_dim: int | None = None,
        shots: int | None = None,
        cfg: dict | None = None,
        output_scale: float = 1.0,
    ) -> None:
        n_qubits = int(n_qubits) if n_qubits else int(latent_dim)
        if n_qubits > latent_dim:
            raise ValueError(
                f"n_qubits ({n_qubits}) cannot exceed latent_dim ({latent_dim}). "
                "The previous implementation clamped this silently."
            )
        super().__init__(
            latent_dim=latent_dim,
            noise_dim=int(noise_dim) if noise_dim else n_qubits,
            output_scale=output_scale,
        )
        self.n_qubits = n_qubits
        self.n_layers = int(n_layers)
        self.entanglement = str(entanglement)
        self.ansatz = str(ansatz)
        self.shots = shots
        self._cfg = cfg or {}

        if self.ansatz == "efficient_su2":
            self.pqc = build_efficient_su2_layer(self.n_qubits, self.n_layers, self._cfg)
        else:
            self.pqc = build_style_layer(
                self.n_qubits, self.n_layers, self._cfg, self.entanglement, shots
            )

        # Fixed, untrained noise projection: only present when the caller asks for a
        # wider noise input than the circuit has qubits. Registered as a buffer so it
        # is saved with the checkpoint but contributes no trainable parameters.
        if self.noise_dim != self.n_qubits:
            gen = torch.Generator().manual_seed(int(self._cfg.get("seed", 42)))
            proj = torch.randn(self.noise_dim, self.n_qubits, generator=gen)
            proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.register_buffer("noise_projection", proj, persistent=True)
        else:
            # Registered as None so the attribute exists uniformly and state_dict
            # handling does not depend on which branch was taken.
            self.register_buffer("noise_projection", None)

        # Classical style expansion when the latent is wider than the circuit.
        self.expand = (
            nn.Sequential(nn.Linear(self.n_qubits, self.latent_dim), nn.Tanh())
            if self.n_qubits != self.latent_dim
            else None
        )
        self._init_params()

    def _init_params(self) -> None:
        with torch.no_grad():
            for name, p in self.pqc.named_parameters():
                if "w_style" in name:
                    p.fill_(1.0)
                elif "b_style" in name:
                    p.zero_()
                else:
                    p.normal_(0.0, 0.1)

    # ------------------------------------------------------------- generation
    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        u = noise
        if self.noise_projection is not None:
            u = u @ self.noise_projection.to(u.device, u.dtype)
        batch = u.shape[0]
        # PennyLane's analytic simulator runs on CPU; the result returns to the
        # caller's device.
        out = self._stack(self.pqc(u.detach().cpu().float()), batch)
        if self.expand is not None:
            out = self.expand(out.to(self.expand[0].weight.device))
        return out.to(noise.device)

    def _stack(self, out, batch: int) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            out = torch.stack(tuple(out), dim=-1 if out[0].dim() > 0 else 0)
            if out.dim() == 1:
                out = out.unsqueeze(0)
        elif out.dim() == 1:
            out = out.unsqueeze(0) if batch == 1 else out.reshape(batch, self.n_qubits)
        return out.reshape(batch, self.n_qubits)

    # -------------------------------------------------------------- reporting
    @property
    def is_pure_quantum(self) -> bool:
        """True when no classical layer sits between the readout and the output."""
        return self.expand is None

    def parameter_breakdown(self) -> dict:
        quantum = sum(p.numel() for p in self.pqc.parameters())
        classical = self.count_params() - quantum
        total = max(1, self.count_params())
        return {
            "total": self.count_params(),
            "quantum": int(quantum),
            "classical": int(classical),
            "classical_fraction": float(classical) / total,
            "is_pure_quantum": self.is_pure_quantum,
            "expected_circuit_params": 4 * self.n_layers * self.n_qubits,
        }

    def describe(self) -> dict:
        base = super().describe()
        base.update(
            {
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
                "entanglement": self.entanglement,
                "ansatz": self.ansatz,
                "shots": self.shots,
                "gate_count": self.gate_count(),
            }
        )
        return base

    def gate_count(self) -> int:
        n, d = self.n_qubits, self.n_layers
        n_ent = len(entangler_pairs(n, self.entanglement))
        if self.ansatz == "efficient_su2":
            return d * (2 * n + n_ent)
        return d * (3 * n + n_ent)

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def mean_entanglement(self, n_samples: int = 256) -> float:
        """Meyer-Wallach Q averaged over the noise distribution.

        ``Q = 2 (1 - mean_k Tr rho_k^2)``, zero for a product state. Averaging over
        noise rather than evaluating at one point matters, and this must be called
        after the selected checkpoint is loaded so it describes the circuit actually
        evaluated.
        """
        if self.ansatz != "style_qgan":
            return float("nan")
        params = {n.split(".")[-1]: p.detach().cpu() for n, p in self.pqc.named_parameters()}
        try:
            w, b, th, ph = params["w_style"], params["b_style"], params["theta"], params["phi"]
        except KeyError:
            return float("nan")

        dev = qml.device("default.qubit", wires=self.n_qubits)
        layer_fn = _style_layer(self.n_qubits, self.entanglement)

        @qml.qnode(dev, interface="torch")
        def purity(inputs):
            for idx in range(self.n_layers):
                layer_fn(inputs, w, b, th, ph, idx)
            return [qml.purity(wires=[i]) for i in range(self.n_qubits)]

        noise = self.sample_noise(int(n_samples), "cpu")
        if self.noise_projection is not None:
            noise = noise @ self.noise_projection
        values = []
        for k in range(noise.shape[0]):
            pur = torch.stack([torch.as_tensor(p) for p in purity(noise[k])]).float()
            values.append(float(2.0 * (1.0 - pur.mean().item())))
        return float(sum(values) / len(values)) if values else float("nan")

    @torch.no_grad()
    def coordinate_coupling(
        self, n_probes: int = 32, eps: float = 0.5, separable_tol: float = 1e-5
    ) -> dict:
        """Measure how far perturbing one noise coordinate moves the others.

        Without entanglement each readout depends on exactly one noise coordinate, so
        the off-diagonal response is identically zero and the circuit provably cannot
        represent dependence. With entanglement it is nonzero. Reporting this makes
        the product control's meaning verifiable rather than asserted, and it shows
        that the entanglers do work even when the distance metric cannot see it.
        """
        base_noise = self.sample_noise(int(n_probes), "cpu")
        base_out = self.generate(base_noise).cpu()

        diag, offdiag = [], []
        for coord in range(self.noise_dim):
            perturbed = base_noise.clone()
            perturbed[:, coord] += eps
            delta = (self.generate(perturbed).cpu() - base_out).abs()
            target = min(coord, self.latent_dim - 1)
            diag.append(float(delta[:, target].mean()))
            others = [j for j in range(self.latent_dim) if j != target]
            if others:
                offdiag.append(float(delta[:, others].mean()))

        mean_diag = sum(diag) / len(diag) if diag else 0.0
        mean_off = sum(offdiag) / len(offdiag) if offdiag else 0.0
        ratio = mean_off / mean_diag if mean_diag > 0 else 0.0
        return {
            "mean_diagonal_response": mean_diag,
            "mean_offdiagonal_response": mean_off,
            "coupling_ratio": ratio,
            # Judged on the ratio against a tolerance rather than against exact zero:
            # a product circuit is analytically separable, but the simulator's
            # floating-point arithmetic leaves a residue around 1e-7.
            "is_separable": ratio < separable_tol,
            "separable_tol": separable_tol,
            "entanglement": self.entanglement,
            "note": (
                "A product topology is analytically separable, so its off-diagonal "
                "response is zero up to simulator arithmetic and it cannot represent "
                "cross-coordinate dependence by construction."
            ),
        }
