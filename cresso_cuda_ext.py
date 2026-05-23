from __future__ import annotations

import hashlib
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
_EXTENSION_ABI_TAG = "safe5"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _torch_cuda_version_tuple() -> tuple[int, int] | None:
    version = torch.version.cuda
    if not version:
        return None
    match = re.match(r"^(\d+)(?:\.(\d+))?", str(version))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _torch_bundled_cuda_home() -> str | None:
    torch_cuda = _torch_cuda_version_tuple()
    if torch_cuda is None:
        return None
    major, _minor = torch_cuda
    torch_root = Path(torch.__file__).resolve().parent
    home = torch_root.parent / "nvidia" / f"cu{major}"
    return str(home) if (home / "bin" / "nvcc").is_file() else None


def _nvcc_from_home(home: str | None) -> str | None:
    if not home:
        return None
    nvcc = Path(home).expanduser() / "bin" / "nvcc"
    return str(nvcc) if nvcc.is_file() else None


def _explicit_nvcc_path() -> str | None:
    return (
        os.environ.get("CUDACXX")
        or _nvcc_from_home(os.environ.get("CRESSO_CUDA_HOME"))
        or _nvcc_from_home(os.environ.get("CUDA_HOME"))
    )


def _candidate_nvcc_paths() -> list[str]:
    candidates: list[str | None] = [
        os.environ.get("CUDACXX"),
        _nvcc_from_home(os.environ.get("CRESSO_CUDA_HOME")),
        _nvcc_from_home(os.environ.get("CUDA_HOME")),
        _nvcc_from_home(_torch_bundled_cuda_home()),
    ]
    torch_cuda = _torch_cuda_version_tuple()
    if torch_cuda is not None:
        major, minor = torch_cuda
        candidates.extend(
            [
                _nvcc_from_home(f"/usr/local/cuda-{major}.{minor}"),
                _nvcc_from_home(f"/usr/local/cuda-{major}"),
            ]
        )
    candidates.extend(
        [
            _nvcc_from_home("/usr/local/cuda-12.8"),
            _nvcc_from_home("/usr/local/cuda"),
            _nvcc_from_home(str(Path.home() / ".local" / "cuda-12.8")),
            shutil.which("nvcc"),
        ]
    )
    seen: set[str] = set()
    paths: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = str(Path(candidate).expanduser())
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _nvcc_path() -> str | None:
    explicit = _explicit_nvcc_path()
    if explicit:
        return explicit
    return _runtime_compatible_nvcc_path()


def _runtime_compatible_nvcc_path() -> str | None:
    candidates = _candidate_nvcc_paths()
    if not candidates:
        return None
    if _env_flag("CRESSO_CUDA_ALLOW_RUNTIME_MISMATCH"):
        return candidates[0]

    torch_cuda = _torch_cuda_version_tuple()
    if torch_cuda is None:
        return candidates[0]

    for candidate in candidates:
        nvcc_cuda = _nvcc_version_tuple(candidate)
        if nvcc_cuda is None or nvcc_cuda[0] == torch_cuda[0]:
            return candidate
    return candidates[0]


def _cuda_home_from_nvcc(nvcc: str | None) -> str | None:
    if not nvcc:
        return None
    nvcc_path = Path(nvcc).resolve()
    if nvcc_path.name != "nvcc" or nvcc_path.parent.name != "bin":
        return None
    home = nvcc_path.parent.parent
    return str(home) if home.is_dir() else None


def _cuda_lib_dir(cuda_home: str | None) -> Path | None:
    if not cuda_home:
        return None
    home = Path(cuda_home)
    for name in ("lib64", "lib"):
        lib = home / name
        if lib.is_dir():
            return lib
    return None


def _ensure_cuda_runtime_linker_name(cuda_home: str | None) -> None:
    torch_cuda = _torch_cuda_version_tuple()
    lib_dir = _cuda_lib_dir(cuda_home)
    if torch_cuda is None or lib_dir is None:
        return
    major, _minor = torch_cuda
    unversioned = lib_dir / "libcudart.so"
    versioned = lib_dir / f"libcudart.so.{major}"
    if unversioned.exists():
        resolved = unversioned.resolve()
        if versioned.exists() and resolved != versioned.resolve() and not _env_flag("CRESSO_CUDA_ALLOW_RUNTIME_MISMATCH"):
            raise RuntimeError(
                f"CRESSO5 CUDA ops refuse to link against {unversioned}, which resolves to {resolved}, "
                f"because PyTorch uses CUDA {major}.x. Fix the CUDA toolkit path or set "
                "CRESSO_CUDA_ALLOW_RUNTIME_MISMATCH=1 only for diagnostics."
            )
        return
    if not versioned.exists():
        return
    try:
        unversioned.symlink_to(versioned.name)
    except OSError as exc:
        raise RuntimeError(
            f"CRESSO5 CUDA ops could not create {unversioned} -> {versioned.name}. "
            "The selected CUDA toolkit lacks the unversioned linker name required by PyTorch extensions."
        ) from exc


def _configure_cuda_toolkit() -> str | None:
    nvcc = _nvcc_path()
    _validate_cuda_runtime_match(nvcc)
    cuda_home = _cuda_home_from_nvcc(nvcc)
    _ensure_cuda_runtime_linker_name(cuda_home)
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


def _nvcc_version_tuple(nvcc: str | None) -> tuple[int, int] | None:
    output = _run_nvcc(nvcc, "--version")
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _validate_cuda_runtime_match(nvcc: str | None) -> None:
    torch_cuda = _torch_cuda_version_tuple()
    nvcc_cuda = _nvcc_version_tuple(nvcc)
    if torch_cuda is None or nvcc_cuda is None:
        return
    if torch_cuda[0] == nvcc_cuda[0]:
        return
    if _env_flag("CRESSO_CUDA_ALLOW_RUNTIME_MISMATCH"):
        warnings.warn(
            "CRESSO5 CUDA ops are building with an nvcc runtime major version that does not "
            f"match PyTorch: nvcc CUDA {nvcc_cuda[0]}.{nvcc_cuda[1]} at {nvcc}, "
            f"PyTorch CUDA {torch_cuda[0]}.{torch_cuda[1]}. This is for diagnostics only.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise RuntimeError(
        "CRESSO5 CUDA ops refuse to build with mixed CUDA runtime major versions: "
        f"nvcc CUDA {nvcc_cuda[0]}.{nvcc_cuda[1]} at {nvcc}, "
        f"but PyTorch was built for CUDA {torch_cuda[0]}.{torch_cuda[1]}. "
        "Install/select an nvcc toolkit matching PyTorch CUDA, or set "
        "CRESSO_CUDA_ALLOW_RUNTIME_MISMATCH=1 only for diagnostics."
    )


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


def _source_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        p = Path(path)
        digest.update(str(p.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(p.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _extension_module_name(arch_list: str, source_hash: str | None = None) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", arch_list).strip("_").lower() or "default"
    src = re.sub(r"[^0-9A-Za-z]+", "_", source_hash or "nosrc").strip("_").lower()
    return f"cresso_v5_cuda_{suffix}_{src}_{_EXTENSION_ABI_TAG}"


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
    module_name = _extension_module_name(arch_list, _source_hash(sources))
    if _MODULE is not None and _MODULE_NAME == module_name:
        return _MODULE

    cuda_lib = _cuda_lib_dir(os.environ.get("CUDA_HOME"))
    extra_ldflags = [f"-Wl,-rpath,{cuda_lib}"] if cuda_lib is not None else []
    _MODULE = load(
        name=module_name,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        extra_ldflags=extra_ldflags,
        with_cuda=True,
        verbose=bool(verbose),
    )
    _MODULE_NAME = module_name
    return _MODULE
