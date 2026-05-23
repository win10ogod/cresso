from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "cresso_v5.py"
SPEC = importlib.util.spec_from_file_location("cresso_v5", SCRIPT_PATH)
cresso_v5 = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(cresso_v5)


class CressoV5CudaSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        cresso_v5._CRESSO_CUDA_OPS = None
        cresso_v5._CRESSO_CUDA_ERROR = None
        cresso_v5._CRESSO_CUDA_WARNED = False

    def test_unsafe_cuda_extension_build_does_not_silently_fallback_in_auto_mode(self) -> None:
        fake_ext = types.ModuleType("cresso_cuda_ext")

        def load_cuda_ops():
            raise RuntimeError("CRESSO5 CUDA ops refuse to build without a native GPU architecture")

        fake_ext.load_cuda_ops = load_cuda_ops  # type: ignore[attr-defined]

        with mock.patch.object(cresso_v5.torch.cuda, "is_available", return_value=True), \
            mock.patch.dict(sys.modules, {"cresso_cuda_ext": fake_ext}), \
            self.assertRaisesRegex(RuntimeError, "refused unsafe CUDA extension build"):
            cresso_v5._load_cuda_ops(required=False)

    def test_cuda_ops_required_rejects_cpu_params(self) -> None:
        param = torch.nn.Parameter(torch.ones(4, 4))
        param.grad = torch.ones_like(param)
        opt = cresso_v5.CRESSO5([param], cuda_ops="required", min_spectral_size=1)

        with self.assertRaisesRegex(RuntimeError, "cuda_ops='required'.*CUDA parameter"):
            opt.step()

    def test_reservoir_update_keeps_state_tensor_identity(self) -> None:
        param = torch.nn.Parameter(torch.ones(1))
        opt = cresso_v5.CRESSO5([param])
        state = torch.tensor(1.0)

        first = opt._update_reservoir(state, torch.tensor(3.0), decay=0.5, step=1)
        self.assertIs(first, state)
        self.assertEqual(float(state), 3.0)

        second = opt._update_reservoir(state, torch.tensor(5.0), decay=0.5, step=2)
        self.assertIs(second, state)
        self.assertEqual(float(state), 4.0)

    def test_scalar_cuda_update_does_not_allocate_dense_tangent_or_drive_workspace(self) -> None:
        kernel_path = Path(__file__).resolve().parents[1] / "cresso_cuda" / "cresso_cuda_kernel.cu"
        source = kernel_path.read_text(encoding="utf-8")
        start = source.index("void scalar_update_2d_cuda(")
        scalar_update_block = source[start:]

        self.assertNotIn("auto tangent = torch::empty(param.sizes()", scalar_update_block)
        self.assertNotIn("auto drive = torch::empty(param.sizes()", scalar_update_block)

    def test_scalar_cuda_update_does_not_force_dense_grad_work_copy(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        start = source.index("ops.scalar_update_2d(")
        end = source.index("eps,", start)
        scalar_call = source[start:end]

        self.assertNotIn("grad.to(dtype=dtype).contiguous()", scalar_call)


if __name__ == "__main__":
    unittest.main()
