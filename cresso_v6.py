"""CRESSO6: speed-first contact-refractory optimizer.

CRESSO6 keeps the parts of CRESSO5 that are computationally cheap and removes the
parts that dominate runtime in pure PyTorch: coordinate grids, trig basis construction,
per-step spectral reconstruction, hash modulo fields, and row/column expansion.

Two variants are provided:

* CRESSO6Fast: group-level scalar contact/refractory state with a foreach AXPY update.
  Persistent state is O(number of parameter groups). This is the speed target.

* CRESSO6Hard: block-level contact state. It is more adaptive than CRESSO6Fast while
  keeping persistent state far below AdamW. It is slower than Fast but still avoids
  spectral/hash/grid reconstruction.

Both variants use current gradients directly. There is no dense momentum, dense second
moment, Newton-Schulz, polar decomposition, orthogonalization, or imported optimizer rule.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "CRESSO6Fast",
    "CRESSO6Hard",
    "CRESSO6",
    "count_optimizer_state_elements",
    "theoretical_fast_state_elements",
    "theoretical_hard_state_elements",
]


def _validate_scalar(name: str, value: float, lower: float | None = None, upper: float | None = None) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    if lower is not None and float(value) < lower:
        raise ValueError(f"{name} must be >= {lower}, got {value}")
    if upper is not None and float(value) > upper:
        raise ValueError(f"{name} must be <= {upper}, got {value}")


def count_optimizer_state_elements(optimizer: Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel())
    for group in getattr(optimizer, "param_groups", []):
        gstate = group.get("_cresso6_state") if isinstance(group, dict) else None
        if isinstance(gstate, dict):
            for value in gstate.values():
                if torch.is_tensor(value):
                    total += int(value.numel())
    return int(total)


def theoretical_fast_state_elements(num_groups: int = 1) -> int:
    # step, energy, previous energy, gain, refractory, reward prediction.
    return int(6 * num_groups)


def theoretical_hard_state_elements(shapes: Sequence[Sequence[int]], block_size: int = 4096) -> int:
    total = 6  # group scalar state
    for shape in shapes:
        n = 1
        for d in shape:
            n *= int(d)
        if n > 0:
            total += max(1, math.ceil(n / int(block_size)))
    return int(total)


class CRESSO6Fast(Optimizer):
    """Ultra-fast contact-refractory gradient optimizer.

    Persistent state is six scalars per parameter group. The common path is:

        p <- decay * p - lr * group_gain / sqrt(group_energy + eps) * grad

    ``group_gain`` is updated by a small contact/refractory dynamical system from
    gradient-energy decrease/increase. The refresh can be made sparse with
    ``refresh_interval`` to reduce reductions. ``sign_mix`` adds a no-state direct
    reflex channel; keep it at 0 for maximum speed.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 3e-2,
        weight_decay: float = 0.0,
        refresh_interval: int = 16,
        energy_beta: float = 0.90,
        gain_adapt: float = 0.05,
        refractory_gain: float = 0.03,
        gain_min: float = 0.02,
        gain_max: float = 100.0,
        precision_max: float = 1e5,
        sign_mix: float = 0.0,
        warmup_steps: int = 0,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        _validate_scalar("lr", lr, 0.0)
        _validate_scalar("weight_decay", weight_decay, 0.0)
        if int(refresh_interval) <= 0:
            raise ValueError("refresh_interval must be positive")
        _validate_scalar("energy_beta", energy_beta, 0.0, 0.9999)
        _validate_scalar("gain_adapt", gain_adapt, 0.0, 10.0)
        _validate_scalar("refractory_gain", refractory_gain, 0.0, 10.0)
        _validate_scalar("gain_min", gain_min, 0.0)
        _validate_scalar("gain_max", gain_max, gain_min)
        _validate_scalar("precision_max", precision_max, 1.0)
        _validate_scalar("sign_mix", sign_mix, 0.0, 1.0)
        if int(warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        _validate_scalar("eps", eps, 0.0)
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            refresh_interval=int(refresh_interval),
            energy_beta=float(energy_beta),
            gain_adapt=float(gain_adapt),
            refractory_gain=float(refractory_gain),
            gain_min=float(gain_min),
            gain_max=float(gain_max),
            precision_max=float(precision_max),
            sign_mix=float(sign_mix),
            warmup_steps=int(warmup_steps),
            eps=float(eps),
            maximize=bool(maximize),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            maximize = bool(group["maximize"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("CRESSO6Fast does not support sparse gradients")
                if not torch.is_floating_point(g) or g.numel() == 0:
                    continue
                params.append(p)
                grads.append(-g if maximize else g)
            if not params:
                continue

            state = group.setdefault("_cresso6_state", {})
            dev = params[0].device
            if not state:
                state["step"] = torch.zeros((), dtype=torch.float32, device=dev)
                state["energy"] = torch.zeros((), dtype=torch.float32, device=dev)
                state["prev_energy"] = torch.zeros((), dtype=torch.float32, device=dev)
                state["gain"] = torch.ones((), dtype=torch.float32, device=dev)
                state["refractory"] = torch.zeros((), dtype=torch.float32, device=dev)
                state["reward_pred"] = torch.zeros((), dtype=torch.float32, device=dev)

            state["step"].add_(1.0)
            step_i = int(state["step"].item())
            interval = int(group["refresh_interval"])
            refresh = step_i == 1 or interval == 1 or ((step_i - 1) % interval == 0)

            if refresh:
                total = torch.zeros((), dtype=torch.float32, device=dev)
                count = 0
                for g in grads:
                    g32 = g if g.dtype == torch.float32 else g.float()
                    total.add_(torch.sum(g32 * g32).to(torch.float32))
                    count += int(g.numel())
                e_now = total / max(count, 1)
                beta = float(group["energy_beta"])
                if step_i == 1:
                    state["energy"].copy_(e_now)
                    state["prev_energy"].copy_(e_now)
                else:
                    eps = float(group["eps"])
                    old_energy = state["energy"].clone()
                    state["energy"].mul_(beta).add_(e_now, alpha=1.0 - beta)
                    reward = torch.tanh((state["prev_energy"] - e_now) / (state["prev_energy"].abs() + e_now.abs() + eps))
                    rpe = reward - state["reward_pred"]
                    state["reward_pred"].add_(rpe, alpha=0.05)
                    # Refractory response grows when current gradient energy exceeds the contact reservoir.
                    rebound = ((e_now - old_energy).clamp_min(0.0) / (e_now.abs() + old_energy.abs() + eps)).clamp(0.0, 4.0)
                    state["refractory"].mul_(0.90).add_(rebound, alpha=0.10)
                    log_mult = float(group["gain_adapt"]) * torch.tanh(rpe) - float(group["refractory_gain"]) * state["refractory"]
                    state["gain"].mul_(torch.exp(log_mult).clamp(0.75, 1.25)).clamp_(float(group["gain_min"]), float(group["gain_max"]))
                    state["prev_energy"].copy_(e_now)

            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            if wd != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * wd)

            precision = torch.rsqrt(state["energy"] + float(group["eps"])).clamp(max=float(group["precision_max"]))
            gain = state["gain"] * precision
            warmup = int(group["warmup_steps"])
            if warmup > 0 and step_i <= warmup:
                gain = gain * (step_i / float(warmup))
            sign_mix = float(group["sign_mix"])
            alpha = -lr * float(gain.item()) * (1.0 - sign_mix)
            if alpha != 0.0:
                torch._foreach_add_(params, grads, alpha=alpha)
            if sign_mix != 0.0:
                # No persistent state. This is slower than the pure fast path because sign tensors are materialized.
                sign_grads = [g.sign() for g in grads]
                # Sign channel uses the contact gain but not precision; otherwise it can explode when gradients are tiny.
                sign_alpha = -lr * float(state["gain"].item()) * sign_mix
                torch._foreach_add_(params, sign_grads, alpha=sign_alpha)
        return loss


class CRESSO6Hard(Optimizer):
    """Block-adaptive contact-refractory optimizer.

    This variant replaces expensive CRESSO5 spectral/hash reconstruction with block contact
    reservoirs. It stores one scalar energy per contiguous block and uses it as a cheap local
    precision field. Memory is roughly ``N / block_size`` instead of AdamW's ``2N``.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 3e-2,
        weight_decay: float = 0.0,
        block_size: int = 4096,
        refresh_interval: int = 4,
        energy_beta: float = 0.92,
        gain_adapt: float = 0.04,
        refractory_gain: float = 0.02,
        gain_min: float = 0.02,
        gain_max: float = 100.0,
        precision_max: float = 1e5,
        sign_mix: float = 0.0,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        _validate_scalar("lr", lr, 0.0)
        _validate_scalar("weight_decay", weight_decay, 0.0)
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        if int(refresh_interval) <= 0:
            raise ValueError("refresh_interval must be positive")
        _validate_scalar("energy_beta", energy_beta, 0.0, 0.9999)
        _validate_scalar("gain_adapt", gain_adapt, 0.0, 10.0)
        _validate_scalar("refractory_gain", refractory_gain, 0.0, 10.0)
        _validate_scalar("gain_min", gain_min, 0.0)
        _validate_scalar("gain_max", gain_max, gain_min)
        _validate_scalar("precision_max", precision_max, 1.0)
        _validate_scalar("sign_mix", sign_mix, 0.0, 1.0)
        _validate_scalar("eps", eps, 0.0)
        defaults = dict(
            lr=float(lr), weight_decay=float(weight_decay), block_size=int(block_size),
            refresh_interval=int(refresh_interval), energy_beta=float(energy_beta),
            gain_adapt=float(gain_adapt), refractory_gain=float(refractory_gain),
            gain_min=float(gain_min), gain_max=float(gain_max), precision_max=float(precision_max),
            sign_mix=float(sign_mix), eps=float(eps), maximize=bool(maximize)
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            block = int(group["block_size"])
            interval = int(group["refresh_interval"])
            beta = float(group["energy_beta"])
            eps = float(group["eps"])
            pmax = float(group["precision_max"])
            sign_mix = float(group["sign_mix"])
            maximize = bool(group["maximize"])
            # Group-level contact state.
            gstate = group.setdefault("_cresso6_state", {})
            first_param = next((p for p in group["params"] if p.grad is not None), None)
            if first_param is None:
                continue
            dev = first_param.device
            if not gstate:
                gstate["step"] = torch.zeros((), dtype=torch.float32, device=dev)
                gstate["energy"] = torch.zeros((), dtype=torch.float32, device=dev)
                gstate["prev_energy"] = torch.zeros((), dtype=torch.float32, device=dev)
                gstate["gain"] = torch.ones((), dtype=torch.float32, device=dev)
                gstate["refractory"] = torch.zeros((), dtype=torch.float32, device=dev)
                gstate["reward_pred"] = torch.zeros((), dtype=torch.float32, device=dev)
            gstate["step"].add_(1.0)
            step_i = int(gstate["step"].item())
            refresh = step_i == 1 or interval == 1 or ((step_i - 1) % interval == 0)
            if refresh:
                total = torch.zeros((), dtype=torch.float32, device=dev)
                count = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    gg = -p.grad.detach() if maximize else p.grad.detach()
                    if gg.is_sparse or not torch.is_floating_point(gg) or gg.numel() == 0:
                        continue
                    gg32 = gg if gg.dtype == torch.float32 else gg.float()
                    total.add_(torch.sum(gg32 * gg32).to(torch.float32))
                    count += int(gg.numel())
                e_now = total / max(count, 1)
                if step_i == 1:
                    gstate["energy"].copy_(e_now)
                    gstate["prev_energy"].copy_(e_now)
                else:
                    old = gstate["energy"].clone()
                    gstate["energy"].mul_(beta).add_(e_now, alpha=1.0 - beta)
                    reward = torch.tanh((gstate["prev_energy"] - e_now) / (gstate["prev_energy"].abs() + e_now.abs() + eps))
                    rpe = reward - gstate["reward_pred"]
                    gstate["reward_pred"].add_(rpe, alpha=0.05)
                    rebound = ((e_now - old).clamp_min(0.0) / (e_now.abs() + old.abs() + eps)).clamp(0.0, 4.0)
                    gstate["refractory"].mul_(0.90).add_(rebound, alpha=0.10)
                    log_mult = float(group["gain_adapt"]) * torch.tanh(rpe) - float(group["refractory_gain"]) * gstate["refractory"]
                    gstate["gain"].mul_(torch.exp(log_mult).clamp(0.75, 1.25)).clamp_(float(group["gain_min"]), float(group["gain_max"]))
                    gstate["prev_energy"].copy_(e_now)

            group_gain = float(gstate["gain"].item())
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = -p.grad.detach() if maximize else p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("CRESSO6Hard does not support sparse gradients")
                if not torch.is_floating_point(g) or g.numel() == 0:
                    continue
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                n = int(p.numel())
                nb = max(1, math.ceil(n / block))
                if len(state) == 0:
                    state["block_energy"] = torch.full((nb,), float(g.float().pow(2).mean().item()) + eps, dtype=torch.float32, device=p.device)
                be = state["block_energy"]
                gf = g.reshape(-1)
                pf = p.reshape(-1)
                if refresh:
                    full = n // block
                    tail = n - full * block
                    if full > 0:
                        m = gf[: full * block]
                        m32 = m if m.dtype == torch.float32 else m.float()
                        vals = (m32.view(full, block) * m32.view(full, block)).mean(dim=1)
                        be[:full].mul_(beta).add_(vals, alpha=1.0 - beta)
                    if tail > 0:
                        t = gf[full * block :]
                        t32 = t if t.dtype == torch.float32 else t.float()
                        val = torch.mean(t32 * t32)
                        be[full].mul_(beta).add_(val, alpha=1.0 - beta)
                scale = torch.rsqrt(be + eps).clamp(max=pmax) * group_gain
                full = n // block
                tail = n - full * block
                if full > 0:
                    update = gf[: full * block].view(full, block) * scale[:full].to(dtype=gf.dtype).view(full, 1)
                    pf[: full * block].view(full, block).add_(update, alpha=-lr * (1.0 - sign_mix))
                    if sign_mix != 0.0:
                        pf[: full * block].view(full, block).add_(gf[: full * block].view(full, block).sign(), alpha=-lr * group_gain * sign_mix)
                if tail > 0:
                    pf[full * block :].add_(gf[full * block :] * scale[full].to(dtype=gf.dtype), alpha=-lr * (1.0 - sign_mix))
                    if sign_mix != 0.0:
                        pf[full * block :].add_(gf[full * block :].sign(), alpha=-lr * group_gain * sign_mix)
        return loss


# Default alias: fastest production target.
CRESSO6 = CRESSO6Fast

class CRESSO6Apex(Optimizer):
    """Hybrid loss-first variant: dense micro-plasticity only for small tensors.

    Large tensors use the CRESSO6Fast scalar contact path. Small tensors use a
    dense eligibility/uncertainty micro-field because small heads, biases, embeddings,
    and toy-benchmark layers are exactly where zero-history optimizers lose loss.

    For large model matrices, set ``micro_max_size`` to a small value or 0 to keep
    state far below AdamW. For small research benchmarks, a higher value gives a
    loss-first profile while the expensive CRESSO5 spectral/hash machinery remains absent.
    """
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 3e-2,
        weight_decay: float = 0.0,
        micro_max_size: int = 32768,
        beta_impulse: float = 0.85,
        beta_energy: float = 0.98,
        direct_mix: float = 0.02,
        novelty_mix: float = 0.0,
        error_energy_mix: float = 0.05,
        refresh_interval: int = 16,
        gain_adapt: float = 0.04,
        refractory_gain: float = 0.02,
        gain_min: float = 0.02,
        gain_max: float = 50.0,
        precision_max: float = 1e8,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        _validate_scalar("lr", lr, 0.0); _validate_scalar("weight_decay", weight_decay, 0.0)
        if int(micro_max_size) < 0: raise ValueError("micro_max_size must be non-negative")
        _validate_scalar("beta_impulse", beta_impulse, 0.0, 0.9999)
        _validate_scalar("beta_energy", beta_energy, 0.0, 0.9999)
        _validate_scalar("direct_mix", direct_mix, 0.0, 1.0)
        _validate_scalar("novelty_mix", novelty_mix, 0.0, 1.0)
        _validate_scalar("error_energy_mix", error_energy_mix, 0.0, 1.0)
        if int(refresh_interval) <= 0: raise ValueError("refresh_interval must be positive")
        _validate_scalar("gain_adapt", gain_adapt, 0.0, 10.0); _validate_scalar("refractory_gain", refractory_gain, 0.0, 10.0)
        _validate_scalar("gain_min", gain_min, 0.0); _validate_scalar("gain_max", gain_max, gain_min)
        _validate_scalar("precision_max", precision_max, 1.0); _validate_scalar("eps", eps, 0.0)
        defaults=dict(lr=float(lr), weight_decay=float(weight_decay), micro_max_size=int(micro_max_size),
                      beta_impulse=float(beta_impulse), beta_energy=float(beta_energy), direct_mix=float(direct_mix),
                      novelty_mix=float(novelty_mix), error_energy_mix=float(error_energy_mix), refresh_interval=int(refresh_interval),
                      gain_adapt=float(gain_adapt), refractory_gain=float(refractory_gain), gain_min=float(gain_min),
                      gain_max=float(gain_max), precision_max=float(precision_max), eps=float(eps), maximize=bool(maximize))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None):  # type: ignore[override]
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for group in self.param_groups:
            lr=float(group['lr']); wd=float(group['weight_decay']); micro=int(group['micro_max_size'])
            beta_i=float(group['beta_impulse']); beta_e=float(group['beta_energy'])
            direct=float(group['direct_mix']); novelty=float(group['novelty_mix']); err_mix=float(group['error_energy_mix'])
            eps=float(group['eps']); pmax=float(group['precision_max']); maximize=bool(group['maximize'])
            # Group contact state, reused by both micro and large paths.
            first=next((p for p in group['params'] if p.grad is not None), None)
            if first is None: continue
            dev=first.device
            gstate=group.setdefault('_cresso6_state', {})
            if not gstate:
                gstate['step']=torch.zeros((), dtype=torch.float32, device=dev)
                gstate['energy']=torch.zeros((), dtype=torch.float32, device=dev)
                gstate['prev_energy']=torch.zeros((), dtype=torch.float32, device=dev)
                gstate['gain']=torch.ones((), dtype=torch.float32, device=dev)
                gstate['refractory']=torch.zeros((), dtype=torch.float32, device=dev)
                gstate['reward_pred']=torch.zeros((), dtype=torch.float32, device=dev)
            gstate['step'].add_(1.0); step_i=int(gstate['step'].item())
            interval=int(group['refresh_interval']); refresh=step_i==1 or interval==1 or ((step_i-1)%interval==0)
            large_params=[]; large_grads=[]
            if refresh:
                total=torch.zeros((), dtype=torch.float32, device=dev); count=0
                for p in group['params']:
                    if p.grad is None: continue
                    g=(-p.grad.detach() if maximize else p.grad.detach())
                    if g.is_sparse or not torch.is_floating_point(g) or g.numel()==0: continue
                    g32=g if g.dtype==torch.float32 else g.float()
                    total.add_(torch.sum(g32*g32).to(torch.float32)); count += int(g.numel())
                e_now=total/max(count,1)
                if step_i==1:
                    gstate['energy'].copy_(e_now); gstate['prev_energy'].copy_(e_now)
                else:
                    old=gstate['energy'].clone(); epsv=eps
                    gstate['energy'].mul_(0.90).add_(e_now, alpha=0.10)
                    reward=torch.tanh((gstate['prev_energy']-e_now)/(gstate['prev_energy'].abs()+e_now.abs()+epsv))
                    rpe=reward-gstate['reward_pred']; gstate['reward_pred'].add_(rpe, alpha=0.05)
                    rebound=((e_now-old).clamp_min(0.0)/(e_now.abs()+old.abs()+epsv)).clamp(0.0,4.0)
                    gstate['refractory'].mul_(0.90).add_(rebound, alpha=0.10)
                    log_mult=float(group['gain_adapt'])*torch.tanh(rpe)-float(group['refractory_gain'])*gstate['refractory']
                    gstate['gain'].mul_(torch.exp(log_mult).clamp(0.75,1.25)).clamp_(float(group['gain_min']), float(group['gain_max']))
                    gstate['prev_energy'].copy_(e_now)
            contact_gain=float(gstate['gain'].item())
            large_precision=torch.rsqrt(gstate['energy']+eps).clamp(max=pmax)
            large_alpha=-lr*contact_gain*float(large_precision.item())
            for p in group['params']:
                if p.grad is None: continue
                g=(-p.grad.detach() if maximize else p.grad.detach())
                if g.is_sparse: raise RuntimeError('CRESSO6Apex does not support sparse gradients')
                if not torch.is_floating_point(g) or g.numel()==0: continue
                if wd != 0.0: p.mul_(1.0-lr*wd)
                if int(p.numel()) <= micro and micro > 0:
                    st=self.state[p]
                    if len(st)==0:
                        st['impulse']=torch.zeros_like(g)
                        st['energy']=torch.zeros_like(g)
                    imp=st['impulse']; en=st['energy']
                    err=g-imp
                    imp.mul_(beta_i).add_(g, alpha=1.0-beta_i)
                    en_in=g*g
                    if err_mix != 0.0:
                        en_in = en_in + err_mix*(err*err)
                    en.mul_(beta_e).add_(en_in, alpha=1.0-beta_e)
                    scale=torch.rsqrt(en+eps).clamp(max=pmax)
                    drive=imp.mul(1.0-direct-novelty).add(g, alpha=direct)
                    if novelty != 0.0:
                        drive=drive.add(err, alpha=novelty)
                    p.add_(drive*scale, alpha=-lr*contact_gain)
                else:
                    large_params.append(p); large_grads.append(g)
            if large_params:
                torch._foreach_add_(large_params, large_grads, alpha=large_alpha)
        return loss

try:
    __all__.append('CRESSO6Apex')
except Exception:
    pass
