"""CRESSO-v4: Contact-Refractory Hash-Spectral Optimizer.

A standalone PyTorch optimizer prototype that prioritizes harder convergence
while keeping persistent state far below dense-momentum and dense-second-history
optimizers.

CRESSO-v4 treats each parameter tensor as a field and evolves a compact internal
contact-dissipative state composed of:
  * deterministic multiscale spectral carriers,
  * transient axial current geometry,
  * fixed-capacity hash-cell refractory reservoirs,
  * contact action and surprise regulation.

It does not store dense momentum, dense second moments, row/column histories,
Newton-Schulz matrices, polar factors, or matrix orthogonalization buffers.

Compared with CRESSO-v3, v4 adds a performance-oriented hash-cell synaptic field
and transient basis caching. The hash cells are fixed-capacity reservoirs over a
deterministic coordinate hash; they provide local geometry and direction memory
without preserving a historical value for every weight coordinate.

Persistent state for large tensors is O(rank * ndim + rank + hash_bins),
independent of the number of tensor elements. Dense gradients and temporary dense
fields are still used during step(), because PyTorch backpropagation and the
actual parameter update are dense.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "CRESSO4",
    "CRESSO",
    "count_optimizer_state_elements",
    "theoretical_state_elements",
]


def _validate_scalar(name: str, value: float, lower: float | None = None, upper: float | None = None) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be >= {lower}, got {value}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}, got {value}")


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _state_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _hash_unit(n: int, salt: int = 0) -> float:
    # Deterministic low-cost hash in [0, 1). No randomness is stored in the optimizer state.
    x = (n + 1) * 0x9E3779B1 + (salt + 11) * 0x85EBCA77
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return float(x & 0xFFFFFF) / float(1 << 24)


def _normalize_basis(x: Tensor, zero_mean: bool = False) -> Tensor:
    if zero_mean:
        x = x - x.mean()
    rms = torch.sqrt(x.pow(2).mean() + 1e-12)
    return x / rms.clamp_min(1e-6)


def _make_multiscale_frequencies(rank: int, ndim: int, max_frequency: int, device: torch.device) -> Tensor:
    """Deterministic frequency pack with low, diagonal, sparse mixed and high chirp modes."""
    rank = int(rank)
    ndim = int(ndim)
    max_frequency = max(0, int(max_frequency))
    if rank <= 0:
        return torch.empty((0, ndim), dtype=torch.long, device=device)
    modes: list[tuple[int, ...]] = [tuple([0] * ndim)]
    seen = set(modes)

    if max_frequency <= 0:
        while len(modes) < rank:
            modes.append(tuple([0] * ndim))
        return torch.tensor(modes[:rank], dtype=torch.long, device=device)

    # Axis-local low-frequency modes.
    for f in range(1, max_frequency + 1):
        for axis in range(ndim):
            v = [0] * ndim
            v[axis] = f
            t = tuple(v)
            if t not in seen:
                modes.append(t); seen.add(t)
            if len(modes) >= rank:
                return torch.tensor(modes[:rank], dtype=torch.long, device=device)

    # Diagonal and anti-diagonal carriers.
    for f in range(1, max_frequency + 1):
        v = [f] * ndim
        t = tuple(v)
        if t not in seen:
            modes.append(t); seen.add(t)
        if len(modes) >= rank:
            return torch.tensor(modes[:rank], dtype=torch.long, device=device)
        if ndim > 1:
            v = [f if i % 2 == 0 else max(1, max_frequency + 1 - f) for i in range(ndim)]
            t = tuple(v)
            if t not in seen:
                modes.append(t); seen.add(t)
            if len(modes) >= rank:
                return torch.tensor(modes[:rank], dtype=torch.long, device=device)

    # Pairwise sparse mixed modes.
    for f1 in range(1, max_frequency + 1):
        for f2 in range(1, max_frequency + 1):
            for a in range(ndim):
                if ndim == 1:
                    continue
                b = (a + 1) % ndim
                v = [0] * ndim
                v[a] = f1
                v[b] = f2
                t = tuple(v)
                if t not in seen:
                    modes.append(t); seen.add(t)
                if len(modes) >= rank:
                    return torch.tensor(modes[:rank], dtype=torch.long, device=device)

    # Hashed mixed modes for remaining capacity. Duplicates are allowed only after
    # an exhaustion guard; duplicate carriers still carry independent contact state.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    n = 1
    attempts = 0
    max_unique_attempts = max(64, 10 * rank * max(1, ndim))
    while len(modes) < rank:
        v: list[int] = []
        active = 0
        for axis in range(ndim):
            raw = (n * primes[axis % len(primes)] + axis * axis + 3 * axis + 1) % (max_frequency + 1)
            f = int(raw)
            if f != 0:
                active += 1
            v.append(f)
        if active == 0:
            v[n % ndim] = 1 + (n % max_frequency)
        elif active == 1 and ndim > 1:
            v[(n + 1) % ndim] = 1 + ((n // 2) % max_frequency)
        t = tuple(v)
        if t not in seen or attempts >= max_unique_attempts:
            modes.append(t); seen.add(t)
        n += 1
        attempts += 1
    return torch.tensor(modes[:rank], dtype=torch.long, device=device)


def _omega_from_freqs(freqs: Tensor, base: float = 1.0) -> Tensor:
    if freqs.numel() == 0:
        return torch.empty((0,), device=freqs.device, dtype=torch.float32)
    f = freqs.to(torch.float32)
    return base * torch.sqrt(1.0 + (f * f).sum(dim=1))


def _axis_coord(dim: int, device: torch.device, dtype: torch.dtype, centered: bool = False) -> Tensor:
    u = (torch.arange(dim, device=device, dtype=dtype) + 0.5) / float(dim)
    return u - 0.5 if centered else u



def _instant_axis_metric(rel: Tensor, power: float, coupling: float, eps: float) -> Tensor:
    """Stateless axial current metric built only from the current gradient field.

    This transient metric damps coordinates whose current-force density is high
    along multiple tensor axes, without storing row/column or per-weight history.
    """
    if coupling == 0.0 or rel.ndim == 0 or rel.numel() <= 1:
        return torch.ones_like(rel)
    ndim = rel.ndim
    power = max(float(power), 0.25)
    density = rel.clamp_min(eps).pow(power).clamp(max=1.0e6)
    log_metric: Tensor | None = None
    for axis in range(ndim):
        reduce_dims = tuple(i for i in range(ndim) if i != axis)
        ax = density.mean(dim=reduce_dims, keepdim=True) if reduce_dims else density
        ax = ax / ax.mean().clamp_min(eps)
        term = torch.log(ax.clamp_min(eps))
        log_metric = term if log_metric is None else log_metric + term
    assert log_metric is not None
    return torch.exp((float(coupling) / (ndim * power)) * log_metric).clamp(0.08, 12.0)


def _rms_normalize(x: Tensor, target_rms: Tensor | float, eps: float) -> Tensor:
    rms = torch.sqrt(x.pow(2).mean() + eps)
    if torch.is_tensor(target_rms):
        return x * (target_rms.to(dtype=x.dtype, device=x.device) / rms.clamp_min(eps))
    return x * (float(target_rms) / rms.clamp_min(eps))


def _additive_phase(shape: Sequence[int], freqs: Tensor, mode_index: int, dtype: torch.dtype, chirp: bool = False) -> Tensor:
    device = freqs.device
    ndim = len(shape)
    phase: Tensor | None = None
    for axis, dim in enumerate(shape):
        f = int(freqs[mode_index, axis].item()) if freqs.numel() else 0
        if f == 0 and not chirp:
            continue
        u = _axis_coord(dim, device, dtype, centered=False)
        comp = math.pi * float(f) * u
        if chirp:
            c = 0.25 + 1.75 * _hash_unit(mode_index, 17 + axis)
            # A centered quadratic term gives non-stationary wave packets without storing extra coefficients.
            comp = comp + math.pi * c * float(max(1, f)) * (u - 0.5).pow(2)
        view = [1] * ndim
        view[axis] = dim
        comp = comp.view(view)
        phase = comp if phase is None else phase + comp
    if phase is None:
        return torch.zeros(shape, device=device, dtype=dtype)
    return phase.expand(shape)


def _basis_pair(shape: Sequence[int], freqs: Tensor, mode_index: int, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """Return normalized cosine/sine basis tensors for one deterministic multiwave carrier."""
    freq_sum = int(freqs[mode_index].abs().sum().item()) if freqs.numel() else 0
    device = freqs.device
    if freq_sum == 0:
        return torch.ones(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    kind = mode_index % 4
    ndim = len(shape)

    if kind == 0:
        phase = _additive_phase(shape, freqs, mode_index, dtype, chirp=False)
        return _normalize_basis(torch.cos(phase), zero_mean=True), _normalize_basis(torch.sin(phase), zero_mean=True)

    if kind == 1:
        cos_b: Tensor | None = None
        sin_b: Tensor | None = None
        active = 0
        for axis, dim in enumerate(shape):
            f = int(freqs[mode_index, axis].item())
            if f == 0:
                continue
            active += 1
            u = _axis_coord(dim, device, dtype, centered=False)
            phase = math.pi * float(f) * u
            view = [1] * ndim
            view[axis] = dim
            c = torch.cos(phase).view(view)
            s = torch.sin(phase).view(view)
            cos_b = c if cos_b is None else cos_b * c
            sin_b = s if sin_b is None else sin_b * s
        if active == 0:
            return torch.ones(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
        return _normalize_basis(cos_b.expand(shape), zero_mean=True), _normalize_basis(sin_b.expand(shape), zero_mean=True)

    if kind == 2:
        phase = _additive_phase(shape, freqs, mode_index, dtype, chirp=True)
        return _normalize_basis(torch.cos(phase), zero_mean=True), _normalize_basis(torch.sin(phase), zero_mean=True)

    # Localized wave packet: deterministic center/width, additive phase, Gaussian envelope.
    phase = _additive_phase(shape, freqs, mode_index, dtype, chirp=True)
    window: Tensor | None = None
    for axis, dim in enumerate(shape):
        u = _axis_coord(dim, device, dtype, centered=False)
        center = 0.15 + 0.70 * _hash_unit(mode_index, 101 + axis)
        width = 0.18 + 0.20 * _hash_unit(mode_index, 151 + axis)
        w = torch.exp(-0.5 * ((u - center) / width).pow(2))
        view = [1] * ndim
        view[axis] = dim
        w = w.view(view)
        window = w if window is None else window * w
    assert window is not None
    window = window.expand(shape)
    return _normalize_basis(window * torch.cos(phase), zero_mean=True), _normalize_basis(window * torch.sin(phase), zero_mean=True)


def _project(x: Tensor, freqs: Tensor, rank: int) -> Tuple[Tensor, Tensor]:
    cos_coeff = torch.empty(rank, device=x.device, dtype=x.dtype)
    sin_coeff = torch.empty(rank, device=x.device, dtype=x.dtype)
    shape = tuple(x.shape)
    for r in range(rank):
        bc, bs = _basis_pair(shape, freqs, r, x.dtype)
        cos_coeff[r] = (x * bc).mean()
        sin_coeff[r] = (x * bs).mean()
    return cos_coeff, sin_coeff


def _reconstruct(cos_coeff: Tensor, sin_coeff: Tensor, freqs: Tensor, shape: Sequence[int], dtype: torch.dtype, gates: Tensor | None = None) -> Tensor:
    rank = int(cos_coeff.numel())
    out = torch.zeros(shape, device=cos_coeff.device, dtype=dtype)
    if rank == 0:
        return out
    # Multiwave carriers are intentionally redundant; shrinkage limits coherent over-summation.
    shrink = 1.0 / math.sqrt(max(1.0, 0.7 * math.log2(rank + 1.0)))
    for r in range(rank):
        gate = 1.0 if gates is None else float(gates[r].item())
        if gate == 0.0:
            continue
        bc, bs = _basis_pair(shape, freqs, r, dtype)
        out.add_(bc, alpha=float(cos_coeff[r].item()) * shrink * gate)
        out.add_(bs, alpha=float(sin_coeff[r].item()) * shrink * gate)
    return out



def _basis_pairs_for_step(
    shape: Sequence[int],
    freqs: Tensor,
    rank: int,
    dtype: torch.dtype,
    cache_limit_elements: int,
) -> list[Tuple[Tensor, Tensor]] | None:
    """Build a transient per-step basis cache when it is small enough."""
    n = 1
    for d in shape:
        n *= int(d)
    if int(cache_limit_elements) <= 0 or 2 * int(rank) * n > int(cache_limit_elements):
        return None
    return [_basis_pair(shape, freqs, r, dtype) for r in range(rank)]


def _project_step(x: Tensor, freqs: Tensor, rank: int, basis_pairs: list[Tuple[Tensor, Tensor]] | None) -> Tuple[Tensor, Tensor]:
    if basis_pairs is None:
        return _project(x, freqs, rank)
    cos_coeff = torch.empty(rank, device=x.device, dtype=x.dtype)
    sin_coeff = torch.empty(rank, device=x.device, dtype=x.dtype)
    for r, (bc, bs) in enumerate(basis_pairs):
        cos_coeff[r] = (x * bc).mean()
        sin_coeff[r] = (x * bs).mean()
    return cos_coeff, sin_coeff


def _reconstruct_step(
    cos_coeff: Tensor,
    sin_coeff: Tensor,
    freqs: Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    gates: Tensor | None = None,
    basis_pairs: list[Tuple[Tensor, Tensor]] | None = None,
) -> Tensor:
    if basis_pairs is None:
        return _reconstruct(cos_coeff, sin_coeff, freqs, shape, dtype, gates)
    rank = int(cos_coeff.numel())
    out = torch.zeros(shape, device=cos_coeff.device, dtype=dtype)
    if rank == 0:
        return out
    shrink = 1.0 / math.sqrt(max(1.0, 0.7 * math.log2(rank + 1.0)))
    for r, (bc, bs) in enumerate(basis_pairs):
        gate = 1.0 if gates is None else float(gates[r].item())
        if gate == 0.0:
            continue
        out.add_(bc, alpha=float(cos_coeff[r].item()) * shrink * gate)
        out.add_(bs, alpha=float(sin_coeff[r].item()) * shrink * gate)
    return out



def _hash_index_tensor(shape: Sequence[int], bins: int, device: torch.device) -> Tensor:
    """Transient coordinate hash. No dense index tensor is stored in optimizer state."""
    bins = int(bins)
    n = 1
    for d in shape:
        n *= int(d)
    if bins <= 0:
        return torch.empty(shape, device=device, dtype=torch.long)
    i = torch.arange(n, device=device, dtype=torch.long)
    # Two multiplicative-mix rounds; deterministic and cheap.
    h = i * 1103515245 + 12345
    h = h ^ (h >> 16)
    h = h * 2246822519
    h = h ^ (h >> 13)
    return torch.remainder(h, bins).view(shape)


def _hash_reduce_mean(x: Tensor, idx: Tensor, bins: int) -> Tensor:
    flat_idx = idx.reshape(-1)
    flat_x = x.reshape(-1)
    out = torch.zeros(int(bins), device=x.device, dtype=x.dtype)
    out.scatter_add_(0, flat_idx, flat_x)
    counts = torch.zeros(int(bins), device=x.device, dtype=x.dtype)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_x, dtype=x.dtype))
    return out / counts.clamp_min(1.0)


def count_optimizer_state_elements(optimizer: Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel()
    return int(total)


def theoretical_state_elements(
    shape: Sequence[int],
    rank: int,
    spectral: bool = True,
    hash_bins: int = 64,
    hard_channel_min_size: int = 16,
    thin_matrix_hard_cutoff: int = 8,
) -> int:
    dims = tuple(int(d) for d in shape)
    ndim = len(dims)
    rank = int(rank)
    if spectral:
        n = 1
        for d in dims:
            n *= d
        thin_matrix = ndim == 2 and int(thin_matrix_hard_cutoff) > 0 and min(dims) <= int(thin_matrix_hard_cutoff)
        hard_enabled = n >= int(hard_channel_min_size) and not thin_matrix
        hash_state = 3 * int(hash_bins) if hard_enabled else 0
        # freqs: R*k; omega R; q/p and metric q/p each have cos+sin => 8R;
        # calcium cos+sin => 2R; mode_confidence => R; four scalar reservoirs;
        # hash energy/impulse/refractory => 3B when the hard field is enabled.
        return int(rank * ndim + 12 * rank + hash_state + 4)
    # Scalar fallback: q, impulse, metric_q, metric_impulse, force_rms, drive_energy, surprise, action.
    return 8


class CRESSO4(Optimizer):
    """Contact-Refractory Axial-Spectral Optimizer.

    CRESSO4 is a low-persistent-state optimizer prototype. Large tensors are
    driven by an internal contact-dissipative multiwave field, while small
    tensors use a scalar contact fallback. The optimizer intentionally avoids
    dense historical state and does not use Muon-style matrix orthogonalization.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 2.0e-3,
        rank: int = 8,
        max_frequency: int = 5,
        weight_decay: float = 0.0,
        energy_power: float = 1.20,
        dt: float = 0.15,
        impulse_friction: float = 0.18,
        contact_gain: float = 0.16,
        restoring: float = 0.62,
        cubic: float = 0.006,
        metric_dt: float = 0.09,
        metric_friction: float = 0.22,
        metric_coupling: float = 0.36,
        local_sharpness: float = 0.0,
        instant_metric_coupling: float = 0.55,
        instant_metric_power: float = 1.0,
        root_channel_mix: float = 0.16,
        echo_mix: float = 0.06,
        transport_mix: float = 0.18,
        hash_bins: int = 64,
        hash_decay: float = 0.92,
        hash_impulse_decay: float = 0.78,
        hash_metric_coupling: float = 0.42,
        hash_drive_mix: float = 0.18,
        reservoir_decay: float = 0.95,
        refractory_decay: float = 0.90,
        refractory_gain: float = 0.55,
        confidence_decay: float = 0.92,
        confidence_gain: float = 0.50,
        surprise_gain: float = 0.18,
        surprise_decay: float = 0.95,
        prediction_mix: float = 0.16,
        novelty_mix: float = 0.04,
        warmup_steps: int = 10,
        surprise_brake: float = 0.18,
        target_update_rms: float = 1.0,
        min_gain: float = 0.04,
        max_gain: float = 6.0,
        drive_clip: float = 4.0,
        spike_mix: float = 0.25,
        min_spectral_size: int = 1024,
        basis_cache_limit_elements: int = 2_000_000,
        hard_channel_min_size: int = 16,
        thin_matrix_hard_cutoff: int = 8,
        maximize: bool = False,
        eps: float = 1e-8,
    ) -> None:
        for name, value, lower, upper in [
            ("lr", lr, 0.0, None),
            ("weight_decay", weight_decay, 0.0, None),
            ("energy_power", energy_power, 0.25, None),
            ("dt", dt, 0.0, 1.0),
            ("impulse_friction", impulse_friction, 0.0, None),
            ("contact_gain", contact_gain, 0.0, None),
            ("restoring", restoring, 0.0, None),
            ("cubic", cubic, 0.0, None),
            ("metric_dt", metric_dt, 0.0, 1.0),
            ("metric_friction", metric_friction, 0.0, None),
            ("metric_coupling", metric_coupling, 0.0, None),
            ("local_sharpness", local_sharpness, 0.0, None),
            ("instant_metric_coupling", instant_metric_coupling, 0.0, None),
            ("instant_metric_power", instant_metric_power, 0.25, None),
            ("root_channel_mix", root_channel_mix, 0.0, 1.0),
            ("echo_mix", echo_mix, 0.0, 1.0),
            ("transport_mix", transport_mix, 0.0, 1.0),
            ("hash_decay", hash_decay, 0.0, 1.0),
            ("hash_impulse_decay", hash_impulse_decay, 0.0, 1.0),
            ("hash_metric_coupling", hash_metric_coupling, 0.0, None),
            ("hash_drive_mix", hash_drive_mix, 0.0, 1.0),
            ("reservoir_decay", reservoir_decay, 0.0, 1.0),
            ("refractory_decay", refractory_decay, 0.0, 1.0),
            ("refractory_gain", refractory_gain, 0.0, None),
            ("confidence_decay", confidence_decay, 0.0, 1.0),
            ("confidence_gain", confidence_gain, 0.0, None),
            ("surprise_gain", surprise_gain, 0.0, None),
            ("surprise_decay", surprise_decay, 0.0, 1.0),
            ("prediction_mix", prediction_mix, 0.0, 1.0),
            ("novelty_mix", novelty_mix, 0.0, 1.0),
            ("surprise_brake", surprise_brake, 0.0, None),
            ("target_update_rms", target_update_rms, 0.0, None),
            ("min_gain", min_gain, 0.0, None),
            ("max_gain", max_gain, 0.0, None),
            ("drive_clip", drive_clip, 0.1, None),
            ("spike_mix", spike_mix, 0.0, 1.0),
            ("eps", eps, 0.0, None),
        ]:
            _validate_scalar(name, value, lower, upper)
        if int(rank) < 0:
            raise ValueError("rank must be non-negative")
        if int(max_frequency) < 0:
            raise ValueError("max_frequency must be non-negative")
        if int(hash_bins) < 0:
            raise ValueError("hash_bins must be non-negative")
        if int(min_spectral_size) < 1:
            raise ValueError("min_spectral_size must be >= 1")
        if int(basis_cache_limit_elements) < 0:
            raise ValueError("basis_cache_limit_elements must be >= 0")
        if int(hard_channel_min_size) < 0:
            raise ValueError("hard_channel_min_size must be >= 0")
        if int(thin_matrix_hard_cutoff) < 0:
            raise ValueError("thin_matrix_hard_cutoff must be >= 0")
        if int(warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if min_gain > max_gain:
            raise ValueError("min_gain must be <= max_gain")

        defaults = dict(
            lr=float(lr),
            rank=int(rank),
            max_frequency=int(max_frequency),
            weight_decay=float(weight_decay),
            energy_power=float(energy_power),
            dt=float(dt),
            impulse_friction=float(impulse_friction),
            contact_gain=float(contact_gain),
            restoring=float(restoring),
            cubic=float(cubic),
            metric_dt=float(metric_dt),
            metric_friction=float(metric_friction),
            metric_coupling=float(metric_coupling),
            local_sharpness=float(local_sharpness),
            instant_metric_coupling=float(instant_metric_coupling),
            instant_metric_power=float(instant_metric_power),
            root_channel_mix=float(root_channel_mix),
            echo_mix=float(echo_mix),
            transport_mix=float(transport_mix),
            hash_bins=int(hash_bins),
            hash_decay=float(hash_decay),
            hash_impulse_decay=float(hash_impulse_decay),
            hash_metric_coupling=float(hash_metric_coupling),
            hash_drive_mix=float(hash_drive_mix),
            reservoir_decay=float(reservoir_decay),
            refractory_decay=float(refractory_decay),
            refractory_gain=float(refractory_gain),
            confidence_decay=float(confidence_decay),
            confidence_gain=float(confidence_gain),
            surprise_gain=float(surprise_gain),
            surprise_decay=float(surprise_decay),
            prediction_mix=float(prediction_mix),
            novelty_mix=float(novelty_mix),
            warmup_steps=int(warmup_steps),
            surprise_brake=float(surprise_brake),
            target_update_rms=float(target_update_rms),
            min_gain=float(min_gain),
            max_gain=float(max_gain),
            drive_clip=float(drive_clip),
            spike_mix=float(spike_mix),
            min_spectral_size=int(min_spectral_size),
            basis_cache_limit_elements=int(basis_cache_limit_elements),
            hard_channel_min_size=int(hard_channel_min_size),
            thin_matrix_hard_cutoff=int(thin_matrix_hard_cutoff),
            maximize=bool(maximize),
            eps=float(eps),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None or not torch.is_floating_point(param):
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("CRESSO4 does not support sparse gradients")
                if group["maximize"]:
                    grad = -grad
                if param.numel() >= group["min_spectral_size"] and group["rank"] > 0 and param.ndim > 0:
                    self._spectral_step(param, grad, group)
                else:
                    self._scalar_step(param, grad, group)
        return loss

    def _init_common(self, state: dict, device: torch.device, dtype: torch.dtype) -> None:
        state["step"] = 0
        state["force_rms"] = torch.zeros((), device=device, dtype=dtype)
        state["drive_energy"] = torch.ones((), device=device, dtype=dtype)
        state["surprise"] = torch.zeros((), device=device, dtype=dtype)
        state["action"] = torch.zeros((), device=device, dtype=dtype)

    def _init_spectral(self, param: Tensor, group: dict, dtype: torch.dtype) -> None:
        state = self.state[param]
        self._init_common(state, param.device, dtype)
        rank = int(group["rank"])
        freqs = _make_multiscale_frequencies(rank, param.ndim, int(group["max_frequency"]), param.device)
        state["freqs"] = freqs
        state["omega"] = _omega_from_freqs(freqs).to(device=param.device, dtype=dtype)
        for name in ("q_cos", "q_sin", "p_cos", "p_sin", "metric_cos", "metric_sin", "metric_p_cos", "metric_p_sin"):
            state[name] = torch.zeros(rank, device=param.device, dtype=dtype)
        state["calcium_cos"] = torch.zeros(rank, device=param.device, dtype=dtype)
        state["calcium_sin"] = torch.zeros(rank, device=param.device, dtype=dtype)
        state["mode_confidence"] = torch.zeros(rank, device=param.device, dtype=dtype)
        bins = int(group.get("hash_bins", 0))
        thin_cutoff = int(group.get("thin_matrix_hard_cutoff", 0))
        thin_matrix = param.ndim == 2 and thin_cutoff > 0 and min(param.shape) <= thin_cutoff
        if bins > 0 and param.numel() >= int(group.get("hard_channel_min_size", 0)) and not thin_matrix:
            state["hash_energy"] = torch.ones(bins, device=param.device, dtype=dtype)
            state["hash_impulse"] = torch.zeros(bins, device=param.device, dtype=dtype)
            state["hash_refractory"] = torch.zeros(bins, device=param.device, dtype=dtype)

    def _init_scalar(self, param: Tensor, dtype: torch.dtype) -> None:
        state = self.state[param]
        self._init_common(state, param.device, dtype)
        state["q"] = torch.zeros((), device=param.device, dtype=dtype)
        state["impulse"] = torch.zeros((), device=param.device, dtype=dtype)
        state["metric_q"] = torch.zeros((), device=param.device, dtype=dtype)
        state["metric_impulse"] = torch.zeros((), device=param.device, dtype=dtype)

    def _update_reservoir(self, old: Tensor, current: Tensor, decay: float, step: int) -> Tensor:
        if step <= 1:
            return current.detach().clone().to(old.dtype)
        return old.mul(float(decay)).add(current.detach().to(old.dtype), alpha=1.0 - float(decay))

    def _base_force(self, param: Tensor, grad: Tensor, group: dict, dtype: torch.dtype) -> Tensor:
        force = grad.to(dtype=dtype)
        wd = float(group["weight_decay"])
        if wd != 0.0:
            force = force + wd * param.detach().to(dtype=dtype)
        return force

    def _contact_update_pair(
        self,
        q: Tensor,
        p: Tensor,
        port: Tensor,
        omega: Tensor,
        calcium: Tensor,
        action: Tensor,
        group: dict,
        rank: int,
    ) -> Tensor:
        dt = float(group["dt"])
        refractory_decay = float(group["refractory_decay"])
        calcium.mul_(refractory_decay).add_(port.abs(), alpha=1.0 - refractory_decay)
        refractory_gate = 1.0 / (1.0 + float(group["refractory_gain"]) * calcium)
        surprise = self._current_surprise.to(q.dtype) if hasattr(self, "_current_surprise") else torch.zeros((), device=q.device, dtype=q.dtype)
        plasticity = 1.0 + float(group["surprise_gain"]) * surprise / (1.0 + surprise)
        brake = 1.0 / (1.0 + float(group["surprise_brake"]) * surprise)
        gated_port = port * refractory_gate * plasticity * brake

        energy = 0.5 * (p.pow(2).mean() + ((omega * q).pow(2)).mean()) if rank > 0 else torch.zeros_like(action)
        port_power = (gated_port * p).mean() if rank > 0 else torch.zeros_like(action)
        action.mul_(0.985).add_((port_power - energy).to(action.dtype), alpha=dt)
        friction = float(group["impulse_friction"]) + float(group["contact_gain"]) * torch.tanh(action.abs()).item()

        p.mul_(max(0.0, 1.0 - dt * friction))
        p.add_(gated_port - float(group["restoring"]) * (omega * omega) * q - float(group["cubic"]) * q.pow(3), alpha=dt)
        q.add_(p, alpha=dt)
        q.clamp_(-8.0, 8.0)
        return action

    def _spectral_step(self, param: Tensor, grad: Tensor, group: dict) -> None:
        dtype = _work_dtype(param.dtype)
        state = self.state[param]
        if len(state) == 0:
            self._init_spectral(param, group, _state_dtype(dtype))
            state = self.state[param]
        state["step"] += 1
        step = int(state["step"])
        eps = float(group["eps"])

        force = self._base_force(param, grad, group, dtype)
        force_rms_now = torch.sqrt(force.pow(2).mean() + eps)
        state["force_rms"] = self._update_reservoir(state["force_rms"], force_rms_now, group["reservoir_decay"], step)
        scale = state["force_rms"].to(dtype).clamp_min(eps)

        freqs: Tensor = state["freqs"]
        rank = int(state["omega"].numel())
        omega = state["omega"].to(dtype=state["q_cos"].dtype)
        basis_pairs = _basis_pairs_for_step(
            tuple(param.shape), freqs, rank, dtype, int(group["basis_cache_limit_elements"])
        )
        thin_cutoff = int(group["thin_matrix_hard_cutoff"])
        thin_matrix = param.ndim == 2 and thin_cutoff > 0 and min(param.shape) <= thin_cutoff
        hard_enabled = param.numel() >= int(group["hard_channel_min_size"]) and not thin_matrix

        # Spectral metric-density field from current force magnitude. This is a compact field,
        # not a dense historical accumulator.
        rel = (force.abs() / scale + eps).clamp(max=1.0e4)
        hash_idx = None
        hash_metric = 1.0
        hash_pred = 0.0
        bins = int(group["hash_bins"])
        if hard_enabled and bins > 0 and "hash_energy" in state:
            hash_idx = _hash_index_tensor(tuple(param.shape), bins, param.device)
            cell_density = _hash_reduce_mean(rel.pow(float(group["energy_power"])), hash_idx, bins)
            h_energy: Tensor = state["hash_energy"]
            if step <= 1:
                h_energy.copy_(cell_density.to(h_energy.dtype))
            else:
                h_energy.mul_(float(group["hash_decay"])).add_(cell_density.to(h_energy.dtype), alpha=1.0 - float(group["hash_decay"]))
            hm = (h_energy.to(dtype) / h_energy.to(dtype).mean().clamp_min(eps)).clamp_min(eps)
            hash_metric = hm.pow(float(group["hash_metric_coupling"]) / max(0.25, float(group["energy_power"])))
            hash_metric = hash_metric[hash_idx].clamp(0.08, 12.0)
        density = torch.log1p(rel.pow(float(group["energy_power"])))
        density = density / density.mean().clamp_min(eps) - 1.0
        mcos_port, msin_port = _project_step(density, freqs, rank, basis_pairs)
        mcos_port = mcos_port.to(state["metric_cos"].dtype)
        msin_port = msin_port.to(state["metric_sin"].dtype)
        mdt = float(group["metric_dt"])
        mf = float(group["metric_friction"])
        for qname, pname, port in [
            ("metric_cos", "metric_p_cos", mcos_port),
            ("metric_sin", "metric_p_sin", msin_port),
        ]:
            mq: Tensor = state[qname]
            mp: Tensor = state[pname]
            mp.mul_(max(0.0, 1.0 - mdt * mf))
            mp.add_(port - (omega * omega) * mq, alpha=mdt)
            mq.add_(mp, alpha=mdt)
            mq.clamp_(-3.0, 3.0)

        log_metric = _reconstruct_step(
            state["metric_cos"].to(dtype), state["metric_sin"].to(dtype), freqs, tuple(param.shape), dtype, basis_pairs=basis_pairs
        ).clamp(-3.0, 3.0)
        metric = torch.exp(float(group["metric_coupling"]) * log_metric).clamp(0.06, 16.0)

        # Current-step local thermodynamic sharpness and axial current metric:
        # both are transient and no dense state is retained.
        local_sharp = float(group["local_sharpness"])
        if local_sharp != 0.0:
            local_metric = (1.0 + local_sharp * torch.sqrt(rel.clamp(max=100.0))).clamp(1.0, 8.0)
        else:
            local_metric = 1.0
        instant_metric = _instant_axis_metric(
            rel,
            float(group["instant_metric_power"]),
            float(group["instant_metric_coupling"]) if hard_enabled else 0.0,
            eps,
        )
        tangent_force = (force / (scale * metric * local_metric * instant_metric * hash_metric + eps)).clamp(-1.0e4, 1.0e4)

        qcos: Tensor = state["q_cos"]
        qsin: Tensor = state["q_sin"]
        conf: Tensor = state["mode_confidence"]
        mode_gates = torch.sigmoid(float(group["confidence_gain"]) * conf).to(dtype)
        pred = _reconstruct_step(qcos.to(dtype), qsin.to(dtype), freqs, tuple(param.shape), dtype, gates=mode_gates, basis_pairs=basis_pairs)
        error = tangent_force - pred
        surprise_now = torch.sqrt(error.pow(2).mean() + eps)
        state["surprise"] = self._update_reservoir(state["surprise"], surprise_now, group["surprise_decay"], step)
        self._current_surprise = state["surprise"]

        port_source = error + float(group["transport_mix"]) * tangent_force
        pcos_port, psin_port = _project_step(port_source, freqs, rank, basis_pairs)
        pcos_port = pcos_port.to(qcos.dtype)
        psin_port = psin_port.to(qsin.dtype)

        if hash_idx is not None and "hash_impulse" in state:
            cell_port = _hash_reduce_mean(port_source, hash_idx, bins).to(state["hash_impulse"].dtype)
            h_ref: Tensor = state["hash_refractory"]
            h_imp: Tensor = state["hash_impulse"]
            h_ref.mul_(float(group["hash_decay"])).add_(cell_port.abs(), alpha=1.0 - float(group["hash_decay"]))
            h_gate = 1.0 / (1.0 + float(group["refractory_gain"]) * h_ref)
            h_imp.mul_(float(group["hash_impulse_decay"])).add_(cell_port * h_gate, alpha=1.0 - float(group["hash_impulse_decay"]))
            hash_pred = h_imp.to(dtype)[hash_idx]

        # Mode confidence is a refractory reliability trace: modes receiving consistent nonzero
        # error become trusted; modes with noisy sign-inconsistent excitation are damped.
        port_energy = torch.sqrt(pcos_port.pow(2) + psin_port.pow(2) + eps)
        signed_support = (pcos_port * qcos + psin_port * qsin) / (port_energy * torch.sqrt(qcos.pow(2) + qsin.pow(2) + eps) + eps)
        conf.mul_(float(group["confidence_decay"])).add_(signed_support.clamp(-1.0, 1.0), alpha=1.0 - float(group["confidence_decay"]))
        conf.clamp_(-4.0, 4.0)

        action: Tensor = state["action"]
        action = self._contact_update_pair(
            qcos, state["p_cos"], pcos_port, omega, state["calcium_cos"], action, group, rank
        )
        state["action"] = self._contact_update_pair(
            qsin, state["p_sin"], psin_port, omega, state["calcium_sin"], action, group, rank
        )
        del self._current_surprise

        mode_gates = torch.sigmoid(float(group["confidence_gain"]) * conf).to(dtype)
        pred = _reconstruct_step(qcos.to(dtype), qsin.to(dtype), freqs, tuple(param.shape), dtype, gates=mode_gates, basis_pairs=basis_pairs)
        error = tangent_force - pred

        force_rms = torch.sqrt(tangent_force.pow(2).mean() + eps)
        pred_rms = torch.sqrt(pred.pow(2).mean() + eps)
        if pred_rms.item() > math.sqrt(eps):
            alignment = ((tangent_force * pred).mean() / (force_rms * pred_rms + eps)).clamp(-1.0, 1.0)
            align_gate = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0)
        else:
            align_gate = torch.zeros((), device=param.device, dtype=dtype)
        warmup = 1.0 if int(group["warmup_steps"]) == 0 else (1.0 - math.exp(-step / max(1, int(group["warmup_steps"]))))
        surprise_penalty = 1.0 / (1.0 + state["surprise"].to(dtype))
        mix = float(group["prediction_mix"]) * warmup * float(align_gate.item()) * float(surprise_penalty.item())

        clip = float(group["drive_clip"])
        bounded_force = torch.tanh(tangent_force / clip) * clip
        rational_force = tangent_force / (1.0 + tangent_force.abs() / clip)
        root_force = tangent_force.sign() * torch.sqrt(tangent_force.abs().clamp(max=clip * clip) + eps)
        root_force = _rms_normalize(root_force, torch.sqrt(rational_force.pow(2).mean() + eps), eps)
        spike_mix = float(group["spike_mix"]) / (1.0 + float(state["surprise"].item()))
        base_force = (1.0 - spike_mix) * rational_force + spike_mix * bounded_force
        root_mix = (float(group["root_channel_mix"]) if hard_enabled else 0.0) / (1.0 + 0.5 * float(state["surprise"].item()))
        shaped_force = (1.0 - root_mix) * base_force + root_mix * root_force
        novelty = torch.tanh(error / (clip * (1.0 + state["surprise"].to(dtype)))) * clip
        if hard_enabled and float(group["echo_mix"]) != 0.0:
            echo_cos, echo_sin = _project_step(tangent_force, freqs, rank, basis_pairs)
            echo = _reconstruct_step(echo_cos, echo_sin, freqs, tuple(param.shape), dtype, basis_pairs=basis_pairs)
            echo = _rms_normalize(echo, torch.sqrt(shaped_force.pow(2).mean() + eps), eps)
        else:
            echo = 0.0
        if torch.is_tensor(hash_pred):
            hash_pred = _rms_normalize(hash_pred, torch.sqrt(shaped_force.pow(2).mean() + eps), eps)
        drive = (
            (1.0 - mix) * shaped_force
            + mix * pred
            + float(group["novelty_mix"]) * novelty
            + (float(group["echo_mix"]) if hard_enabled else 0.0) * echo
            + (float(group["hash_drive_mix"]) if hard_enabled else 0.0) * hash_pred
        )

        drive_energy_now = drive.pow(2).mean().to(state["drive_energy"].dtype)
        state["drive_energy"] = self._update_reservoir(state["drive_energy"], drive_energy_now, group["reservoir_decay"], step)
        gain = float(group["target_update_rms"]) / math.sqrt(float(state["drive_energy"].item()) + eps)
        gain = max(float(group["min_gain"]), min(float(group["max_gain"]), gain))
        param.add_((drive * gain).to(dtype=param.dtype), alpha=-float(group["lr"]))

    def _scalar_step(self, param: Tensor, grad: Tensor, group: dict) -> None:
        dtype = _work_dtype(param.dtype)
        state = self.state[param]
        if len(state) == 0:
            self._init_scalar(param, _state_dtype(dtype))
            state = self.state[param]
        state["step"] += 1
        step = int(state["step"])
        eps = float(group["eps"])

        force = self._base_force(param, grad, group, dtype)
        force_rms_now = torch.sqrt(force.pow(2).mean() + eps)
        state["force_rms"] = self._update_reservoir(state["force_rms"], force_rms_now, group["reservoir_decay"], step)
        scale = state["force_rms"].to(dtype).clamp_min(eps)

        density = torch.log1p((force.abs() / scale + eps).pow(float(group["energy_power"]))).mean()
        metric_port = (density - state["metric_q"].to(dtype).tanh()).to(state["metric_q"].dtype)
        mdt = float(group["metric_dt"])
        mq: Tensor = state["metric_q"]
        mp: Tensor = state["metric_impulse"]
        mp.mul_(max(0.0, 1.0 - mdt * float(group["metric_friction"])))
        mp.add_(metric_port - mq, alpha=mdt)
        mq.add_(mp, alpha=mdt)
        mq.clamp_(-3.0, 3.0)
        metric = torch.exp(float(group["metric_coupling"]) * mq.to(dtype)).clamp(0.06, 16.0)
        rel = (force.abs() / scale + eps).clamp(max=1.0e4)
        local_sharp = float(group["local_sharpness"])
        local_metric = (1.0 + local_sharp * torch.sqrt(rel.clamp(max=100.0))).clamp(1.0, 8.0) if local_sharp != 0.0 else 1.0
        instant_metric = _instant_axis_metric(
            rel, float(group["instant_metric_power"]), float(group["instant_metric_coupling"]), eps
        )
        tangent_force = (force / (scale * metric * local_metric * instant_metric + eps)).clamp(-1.0e4, 1.0e4)

        q: Tensor = state["q"]
        impulse: Tensor = state["impulse"]
        pred = q.to(dtype).expand_as(tangent_force)
        error = tangent_force - pred
        surprise_now = torch.sqrt(error.pow(2).mean() + eps)
        state["surprise"] = self._update_reservoir(state["surprise"], surprise_now, group["surprise_decay"], step)
        port = error.mean().to(q.dtype)
        dt = float(group["dt"])
        surprise = state["surprise"].to(q.dtype)
        plasticity = 1.0 + float(group["surprise_gain"]) * surprise / (1.0 + surprise)
        brake = 1.0 / (1.0 + float(group["surprise_brake"]) * surprise)
        action: Tensor = state["action"]
        energy = 0.5 * (impulse.pow(2) + q.pow(2))
        action.mul_(0.985).add_((port * impulse - energy).to(action.dtype), alpha=dt)
        friction = float(group["impulse_friction"]) + float(group["contact_gain"]) * torch.tanh(action.abs()).item()
        impulse.mul_(max(0.0, 1.0 - dt * friction))
        impulse.add_(plasticity * brake * port - float(group["restoring"]) * q - float(group["cubic"]) * q.pow(3), alpha=dt)
        q.add_(impulse, alpha=dt)
        q.clamp_(-8.0, 8.0)

        pred = q.to(dtype).expand_as(tangent_force)
        error = tangent_force - pred
        force_rms = torch.sqrt(tangent_force.pow(2).mean() + eps)
        pred_rms = torch.sqrt(pred.pow(2).mean() + eps)
        if pred_rms.item() > math.sqrt(eps):
            alignment = ((tangent_force * pred).mean() / (force_rms * pred_rms + eps)).clamp(-1.0, 1.0)
            align_gate = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0)
        else:
            align_gate = torch.zeros((), device=param.device, dtype=dtype)
        warmup = 1.0 if int(group["warmup_steps"]) == 0 else (1.0 - math.exp(-step / max(1, int(group["warmup_steps"]))))
        mix = float(group["prediction_mix"]) * warmup * float(align_gate.item()) / (1.0 + float(state["surprise"].item()))
        clip = float(group["drive_clip"])
        bounded_force = torch.tanh(tangent_force / clip) * clip
        rational_force = tangent_force / (1.0 + tangent_force.abs() / clip)
        root_force = tangent_force.sign() * torch.sqrt(tangent_force.abs().clamp(max=clip * clip) + eps)
        root_force = _rms_normalize(root_force, torch.sqrt(rational_force.pow(2).mean() + eps), eps)
        spike_mix = float(group["spike_mix"]) / (1.0 + float(state["surprise"].item()))
        base_force = (1.0 - spike_mix) * rational_force + spike_mix * bounded_force
        root_mix = float(group["root_channel_mix"]) / (1.0 + 0.5 * float(state["surprise"].item()))
        shaped_force = (1.0 - root_mix) * base_force + root_mix * root_force
        novelty = torch.tanh(error / (clip * (1.0 + state["surprise"].to(dtype)))) * clip
        drive = (1.0 - mix) * shaped_force + mix * pred + float(group["novelty_mix"]) * novelty
        drive_energy_now = drive.pow(2).mean().to(state["drive_energy"].dtype)
        state["drive_energy"] = self._update_reservoir(state["drive_energy"], drive_energy_now, group["reservoir_decay"], step)
        gain = float(group["target_update_rms"]) / math.sqrt(float(state["drive_energy"].item()) + eps)
        gain = max(float(group["min_gain"]), min(float(group["max_gain"]), gain))
        param.add_((drive * gain).to(dtype=param.dtype), alpha=-float(group["lr"]))


CRESSO = CRESSO4
