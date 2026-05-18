from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_cuda_ops(*, verbose: bool | None = None) -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("CRESSO5 CUDA ops require torch.cuda.is_available()")

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "cresso_cuda" / "cresso_cuda.cpp"),
        str(root / "cresso_cuda" / "cresso_cuda_kernel.cu"),
    ]
    if verbose is None:
        verbose = os.environ.get("CRESSO_CUDA_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) > (9, 0):
            # CUDA 12.0 does not know Blackwell sm_120 yet. Building compute_90
            # PTX lets the installed driver JIT for newer GPUs.
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    _MODULE = load(
        name="cresso_v5_cuda",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE
