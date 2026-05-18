# CRESSO

CRESSO is a family of compact-state PyTorch optimizers for training large neural
networks without dense Adam-style first and second moment buffers.

The current main implementation is **CRESSO5**, a contact-refractory multihash
spectral optimizer with optional fused CUDA C++ operators for the expensive
coordinate-field work in `step()`.

License: **GPL-3.0-only**.

## What CRESSO5 Is

CRESSO5 treats each trainable parameter tensor as a field. Instead of storing
dense momentum and dense variance tensors for every parameter element, it evolves
a compact internal state made from:

- deterministic multiscale spectral carriers,
- transient axial geometry from the current gradient field,
- fixed-capacity multihash refractory reservoirs,
- contact action, surprise, confidence, and drive regulation.

For large tensors, persistent optimizer state is approximately:

```text
O(rank * ndim + rank + hash_tables * hash_bins)
```

This is independent of the number of parameter elements. Dense gradients and
temporary dense update fields are still used during `step()`, because PyTorch
backpropagation and the final parameter update are dense.

## Why CUDA Operators Matter

The pure PyTorch reference path dynamically builds coordinate fields, basis
waves, trigonometric projections, hash indices, scatter reductions, and dense
reconstructions inside every optimizer step. That is correct, but it causes many
small kernel launches and repeated `cos` / `sin` / hash work.

CRESSO5 now includes fused CUDA C++ operators for the heavy pieces:

- compact axis trigonometry cache,
- basis normalization stats,
- basis projection,
- basis reconstruction,
- coordinate hash reduce-mean,
- coordinate hash gather.

The CUDA path is enabled by default with `cuda_ops="auto"`. If CUDA extension
loading fails, CRESSO5 falls back to the PyTorch reference path. Use
`cuda_ops="required"` when you want failures to be explicit.

## Measured Speed

Measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition with PyTorch
`2.11.0+cu130`.

| Case | PyTorch path | CUDA path | Speedup |
| --- | ---: | ---: | ---: |
| Single tensor `(2048, 2048)`, rank=8 | 50.822 ms/step | 11.212 ms/step | 4.53x |
| Single tensor `(4096, 4096)`, rank=8 | 97.635 ms/step | 34.411 ms/step | 2.84x |
| 64 LoRA-like tensors `(4096,16)/(16,4096)`, rank=8 | 2802.094 ms/step | 360.247 ms/step | 7.78x |

Actual speed depends on rank, tensor shape, hash settings, CUDA toolkit, GPU,
and whether a workload is launch-bound or memory-bandwidth-bound.

## Install

Clone the repository:

```bash
git clone https://github.com/win10ogod/cresso.git
cd cresso
```

Install PyTorch with CUDA first. Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install this package in editable mode:

```bash
pip install -e .
```

For CUDA C++ JIT compilation, you also need:

```bash
pip install ninja
```

and a CUDA toolkit with `nvcc` available on `PATH`.

On newer Blackwell GPUs with an older `nvcc`, the loader defaults to
`TORCH_CUDA_ARCH_LIST=9.0+PTX` when no arch list is already set, so the driver can
JIT PTX for the installed GPU.

## Basic Usage

```python
import torch
from cresso_v5 import CRESSO5

model = torch.nn.Linear(4096, 4096, bias=False).cuda()
optimizer = CRESSO5(
    model.parameters(),
    lr=2.0e-3,
    rank=8,
    max_frequency=5,
    hash_bins=64,
    hash_tables=2,
    cuda_ops="auto",      # default: use CUDA fast path when available
)

x = torch.randn(8, 4096, device="cuda")
loss = model(x).pow(2).mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

## CUDA Mode

```python
CRESSO5(params, cuda_ops="auto")
```

Use CUDA fused operators when possible. If the extension cannot be compiled or
loaded, continue with the PyTorch reference path.

```python
CRESSO5(params, cuda_ops="required")
```

Require CUDA fused operators. This is useful for benchmarking and production
runs where silent fallback would hide a performance issue.

```python
CRESSO5(params, cuda_ops="off")
```

Use the PyTorch reference path only. This is useful for parity testing and
debugging.

## LoRA / Fine-Tuning Usage

CRESSO5 can be passed anywhere a normal PyTorch optimizer is accepted:

```python
from cresso_v5 import CRESSO5

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = CRESSO5(
    trainable,
    lr=2.0e-3,
    rank=4,
    max_frequency=3,
    min_spectral_size=4096,
    hash_bins=32,
    hash_tables=1,
    target_update_rms=0.35,
    drive_clip=1.2,
)
```

For Hugging Face Trainer-style flows that accept an optimizer tuple:

```python
optimizers = (optimizer, None)
```

Then pass `optimizers=optimizers` to the trainer and keep the trainer's own
optimizer name set to a harmless built-in value such as `adamw_torch`, because
the actual optimizer object is already supplied.

## Important Parameters

- `rank`: number of spectral carriers. Higher rank increases expressivity and
  compute. Typical values: `4` for LoRA-heavy runs, `8` for stronger spectral
  behavior.
- `max_frequency`: maximum deterministic carrier frequency.
- `min_spectral_size`: tensors below this size use the scalar fallback unless
  `micro_field_max_size` routes them to the micro-field path.
- `hash_bins`: number of cells per hash table.
- `hash_tables`: number of independent salted hash tables.
- `basis_cache_limit_elements`: pure PyTorch dense transient basis cache limit.
  The CUDA path uses compact axis cache instead of dense basis tensors.
- `hard_channel_min_size`: minimum tensor size for hash/refractory hard-channel
  logic.
- `thin_matrix_hard_cutoff`: disables hard-channel logic for very thin matrices
  when the smallest matrix dimension is at or below this cutoff.
- `target_update_rms`, `min_gain`, `max_gain`, `drive_clip`: control final update
  scaling and clipping.

## State Size Utilities

```python
from cresso_v5 import count_optimizer_state_elements, theoretical_state_elements

print(count_optimizer_state_elements(optimizer))
print(theoretical_state_elements((4096, 4096), rank=8, hash_bins=64, hash_tables=2))
```

## Tests

Run all included tests:

```bash
python -m pytest -q
```

Run CUDA parity tests:

```bash
python -m pytest tests/test_cresso_v5_cuda_ops.py -q
```

The CUDA tests compare fused operators and full `optimizer.step()` behavior
against the PyTorch reference path within floating-point tolerances.

## Files

- `cresso_v5.py`: main CRESSO5 optimizer and CUDA fast-path wiring.
- `cresso_cuda_ext.py`: PyTorch JIT extension loader.
- `cresso_cuda/cresso_cuda.cpp`: C++ pybind entry points.
- `cresso_cuda/cresso_cuda_kernel.cu`: CUDA kernels.
- `cresso_v4.py`, `cresso_v3.py`: earlier optimizer prototypes.
- `tests/test_cresso_v5_cuda_ops.py`: CUDA operator and optimizer parity tests.

## Notes

CRESSO is an experimental optimizer. It is intended for users who are actively
testing optimizer behavior, VRAM tradeoffs, and training dynamics. Always compare
against your existing baseline for loss curves, throughput, stability, and final
model quality.
