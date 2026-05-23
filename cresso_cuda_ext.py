from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from types import ModuleType

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None
_MODULE_NAME: str | None = None
_EXTENSION_ABI_TAG = "safe2"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _nvcc_path() -> str | None:
    cudacxx = os.environ.get("CUDACXX")
    if cudacxx:
        return cudacxx

    homes = [
        os.environ.get("CRESSO_CUDA_HOME"),
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda-12.8",
        "/usr/local/cuda",
        str(Path.home() / ".local" / "cuda-12.8"),
    ]
    for home in homes:
        if not home:
            continue
        nvcc = Path(home) / "bin" / "nvcc"
        if nvcc.is_file():
            return str(nvcc)

    return shutil.which("nvcc")


def _cuda_home_from_nvcc(nvcc: str | None) -> str | None:
    if not nvcc:
        return None
    nvcc_path = Path(nvcc).resolve()
    if nvcc_path.name != "nvcc" or nvcc_path.parent.name != "bin":
        return None
    home = nvcc_path.parent.parent
    return str(home) if home.is_dir() else None


def _configure_cuda_toolkit() -> str | None:
    nvcc = _nvcc_path()
    cuda_home = _cuda_home_from_nvcc(nvcc)
    if nvcc:
        os.environ["CUDACXX"] = nvcc
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home
        cpp_extension.CUDA_HOME = cuda_home
    return nvcc


def _run_nvcc(nvcc: str | None, *args: str) -> str:
    if not nvcc:
        return ""
    try:
        result = subprocess.run(
            [nvcc, *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout or ""


def _nvcc_version() -> str:
    return _run_nvcc(_nvcc_path(), "--version").strip()


def _nvcc_supports_device_arch(major: int, minor: int) -> bool:
    nvcc = _nvcc_path()
    if not nvcc:
        return False
    arch = f"sm_{major}{minor}"
    compute = f"compute_{major}{minor}"
    listed = "\n".join(
        (
            _run_nvcc(nvcc, "--list-gpu-code"),
            _run_nvcc(nvcc, "--list-gpu-arch"),
        )
    )
    return arch in listed or compute in listed


def _arch_list_entries(arch_list: str) -> list[str]:
    return [entry for entry in re.split(r"[\s,;]+", arch_list.strip()) if entry]


def _arch_list_has_native_device(arch_list: str, major: int, minor: int) -> bool:
    dotted = f"{major}.{minor}"
    sm = f"sm_{major}{minor}"
    compute = f"compute_{major}{minor}"
    for entry in _arch_list_entries(arch_list):
        base = entry[:-4] if entry.upper().endswith("+PTX") else entry
        base_l = base.lower()
        if base == dotted or base_l == sm or base_l == compute:
            return True
    return False


def _validate_requested_arch_list(name: str, arch_list: str, major: int, minor: int) -> str:
    sm = f"sm_{major}{minor}"
    if not _arch_list_has_native_device(arch_list, major, minor):
        if _env_flag("CRESSO_CUDA_ALLOW_PTX_FALLBACK"):
            warnings.warn(
                f"{name}={arch_list!r} does not include native {sm}; using it only because "
                "CRESSO_CUDA_ALLOW_PTX_FALLBACK=1 was set.",
                RuntimeWarning,
                stacklevel=3,
            )
            return arch_list
        raise RuntimeError(
            f"{name}={arch_list!r} does not include native GPU architecture {sm}. "
            "CRESSO5 CUDA ops refuse PTX-only or wrong-architecture builds by default; "
            "install/select a Linux CUDA toolkit with native GPU support or set "
            "CRESSO_CUDA_ALLOW_PTX_FALLBACK=1 only for diagnostics."
        )
    if _nvcc_supports_device_arch(major, minor):
        return arch_list

    nvcc = _nvcc_path() or "nvcc"
    version = _nvcc_version() or "unknown nvcc version"
    raise RuntimeError(
        f"{name}={arch_list!r} requests native GPU architecture {sm}, but {nvcc} does not support {sm} "
        f"({version}). Install/select a Linux CUDA toolkit whose nvcc supports {sm}."
    )


def _configure_cuda_arch_list() -> str:
    _configure_cuda_toolkit()
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    sm = f"sm_{major}{minor}"

    existing = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if existing:
        return _validate_requested_arch_list("TORCH_CUDA_ARCH_LIST", existing, major, minor)

    override = os.environ.get("CRESSO_CUDA_ARCH_LIST")
    if override:
        arch_list = _validate_requested_arch_list("CRESSO_CUDA_ARCH_LIST", override, major, minor)
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        return arch_list

    if _nvcc_supports_device_arch(major, minor):
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        return arch

    nvcc = _nvcc_path() or "nvcc"
    version = _nvcc_version() or "unknown nvcc version"
    if _env_flag("CRESSO_CUDA_ALLOW_PTX_FALLBACK"):
        warnings.warn(
            "CRESSO5 CUDA ops are using an explicit PTX fallback because "
            f"{nvcc} does not support {sm}. This can exercise driver JIT paths; "
            "install a Linux CUDA toolkit with native GPU support for production training.",
            RuntimeWarning,
            stacklevel=2,
        )
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
        return "9.0+PTX"

    raise RuntimeError(
        "CRESSO5 CUDA ops refuse to build without a native GPU architecture. "
        f"Detected CUDA device capability {arch} ({sm}), but {nvcc} does not support {sm} "
        f"({version}). Install/select a Linux CUDA toolkit whose nvcc supports {sm}, "
        "or set CRESSO_CUDA_ALLOW_PTX_FALLBACK=1 only for diagnostics."
    )


def _extension_module_name(arch_list: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", arch_list).strip("_").lower() or "default"
    return f"cresso_v5_cuda_{suffix}_{_EXTENSION_ABI_TAG}"


def load_cuda_ops(*, verbose: bool | None = None) -> ModuleType:
    global _MODULE, _MODULE_NAME
    if not torch.cuda.is_available():
        raise RuntimeError("CRESSO5 CUDA ops require torch.cuda.is_available()")

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "cresso_cuda" / "cresso_cuda.cpp"),
        str(root / "cresso_cuda" / "cresso_cuda_kernel.cu"),
    ]
    if verbose is None:
        verbose = os.environ.get("CRESSO_CUDA_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
    arch_list = _configure_cuda_arch_list()
    module_name = _extension_module_name(arch_list)
    if _MODULE is not None and _MODULE_NAME == module_name:
        return _MODULE

    _MODULE = load(
        name=module_name,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    _MODULE_NAME = module_name
    return _MODULE
