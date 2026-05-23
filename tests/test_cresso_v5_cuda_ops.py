from __future__ import annotations

import importlib.util
from unittest import mock
import unittest
import types
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "cresso_v5.py"
SPEC = importlib.util.spec_from_file_location("cresso_v5", SCRIPT_PATH)
cresso_v5 = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(cresso_v5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CRESSO5 CUDA operator tests")
class CressoV5CudaOpsTests(unittest.TestCase):
    def test_cuda_basis_projection_and_reconstruction_match_pytorch_reference(self) -> None:
        shape = (17, 19)
        rank = 8
        dtype = torch.float32
        device = torch.device("cuda")
        torch.manual_seed(1234)
        x = torch.randn(shape, device=device, dtype=dtype)
        freqs = cresso_v5._make_multiscale_frequencies(rank, len(shape), 5, device)

        ops = cresso_v5._load_cuda_ops(required=True)
        cuda_cos, cuda_sin = ops.basis_project(x, freqs, list(shape))
        ref_cos, ref_sin = cresso_v5._project(x, freqs, rank)

        torch.testing.assert_close(cuda_cos, ref_cos, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(cuda_sin, ref_sin, rtol=2e-4, atol=2e-5)

        gates = torch.sigmoid(torch.linspace(-2.0, 2.0, rank, device=device, dtype=dtype))
        cuda_out = ops.basis_reconstruct(ref_cos, ref_sin, freqs, list(shape), gates)
        ref_out = cresso_v5._reconstruct(ref_cos, ref_sin, freqs, shape, dtype, gates=gates)

        torch.testing.assert_close(cuda_out, ref_out, rtol=4e-4, atol=4e-5)

    def test_cuda_hash_reduce_and_gather_match_pytorch_reference(self) -> None:
        shape = (23, 29)
        bins = 37
        salt = 2
        device = torch.device("cuda")
        dtype = torch.float32
        torch.manual_seed(5678)
        x = torch.randn(shape, device=device, dtype=dtype)
        values = torch.randn(bins, device=device, dtype=dtype)

        ops = cresso_v5._load_cuda_ops(required=True)
        idx = cresso_v5._hash_index_tensor(shape, bins, device, salt=salt)

        cuda_mean = ops.hash_reduce_mean(x, bins, salt)
        ref_mean = cresso_v5._hash_reduce_mean(x, idx, bins)
        torch.testing.assert_close(cuda_mean, ref_mean, rtol=2e-5, atol=2e-6)

        cuda_gather = ops.hash_gather(values, list(shape), salt)
        ref_gather = values[idx]
        torch.testing.assert_close(cuda_gather, ref_gather, rtol=0.0, atol=0.0)

    def test_cuda_optimizer_step_matches_pytorch_reference_path(self) -> None:
        shape = (31, 37)
        device = torch.device("cuda")
        dtype = torch.float32
        torch.manual_seed(9012)
        initial = torch.randn(shape, device=device, dtype=dtype)
        grad = torch.randn(shape, device=device, dtype=dtype)
        cuda_param = torch.nn.Parameter(initial.clone())
        ref_param = torch.nn.Parameter(initial.clone())
        cuda_param.grad = grad.clone()
        ref_param.grad = grad.clone()
        common_kwargs = dict(
            lr=2.0e-3,
            rank=8,
            max_frequency=5,
            min_spectral_size=1,
            hard_channel_min_size=1,
            thin_matrix_hard_cutoff=0,
            hash_bins=37,
            hash_tables=2,
            basis_cache_limit_elements=0,
        )
        cuda_opt = cresso_v5.CRESSO5([cuda_param], cuda_ops="required", **common_kwargs)
        ref_opt = cresso_v5.CRESSO5([ref_param], cuda_ops="off", **common_kwargs)

        cuda_opt.step()
        ref_opt.step()
        torch.cuda.synchronize()

        torch.testing.assert_close(cuda_param, ref_param, rtol=8e-4, atol=8e-5)
        cuda_state = cuda_opt.state[cuda_param]
        ref_state = ref_opt.state[ref_param]
        for name in (
            "q_cos",
            "q_sin",
            "p_cos",
            "p_sin",
            "metric_cos",
            "metric_sin",
            "hash_energy",
            "hash_impulse",
            "hash_refractory",
        ):
            torch.testing.assert_close(cuda_state[name], ref_state[name], rtol=8e-4, atol=8e-5)

    def test_cuda_spectral_update_supports_fp16_and_bf16_params(self) -> None:
        shape = (31, 37)
        device = torch.device("cuda")
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            with self.subTest(dtype=str(dtype)):
                torch.manual_seed(8642)
                initial = (0.05 * torch.randn(shape, device=device, dtype=torch.float32)).to(dtype)
                grads = [(0.05 * torch.randn(shape, device=device, dtype=torch.float32)).to(dtype) for _ in range(2)]
                cuda_param = torch.nn.Parameter(initial.clone())
                ref_param = torch.nn.Parameter(initial.clone())
                common_kwargs = dict(
                    lr=1.0e-3,
                    rank=8,
                    max_frequency=5,
                    min_spectral_size=1,
                    hard_channel_min_size=1,
                    thin_matrix_hard_cutoff=0,
                    hash_bins=37,
                    hash_tables=2,
                    basis_cache_limit_elements=0,
                )
                cuda_opt = cresso_v5.CRESSO5([cuda_param], cuda_ops="required", **common_kwargs)
                ref_opt = cresso_v5.CRESSO5([ref_param], cuda_ops="off", **common_kwargs)

                for grad in grads:
                    cuda_param.grad = grad.clone()
                    ref_param.grad = grad.clone()
                    cuda_opt.step()
                    ref_opt.step()
                torch.cuda.synchronize()

                atol = 3.0e-3 if dtype is torch.float16 else 3.0e-2
                torch.testing.assert_close(cuda_param.float(), ref_param.float(), rtol=0.0, atol=atol)
                self.assertEqual(cuda_opt.state[cuda_param]["q_cos"].dtype, torch.float32)

    def test_cuda_fast_path_does_not_add_persistent_basis_cache_state(self) -> None:
        shape = (128, 128)
        device = torch.device("cuda")
        p = torch.nn.Parameter(torch.randn(shape, device=device, dtype=torch.float32))
        p.grad = torch.randn_like(p)
        opt = cresso_v5.CRESSO5(
            [p],
            cuda_ops="required",
            lr=2.0e-3,
            rank=4,
            max_frequency=3,
            min_spectral_size=1,
            hard_channel_min_size=1,
            thin_matrix_hard_cutoff=0,
            hash_bins=16,
            hash_tables=1,
            basis_cache_limit_elements=0,
        )

        opt.step()
        torch.cuda.synchronize()

        state = opt.state[p]
        self.assertNotIn("basis_stats", state)
        self.assertNotIn("basis_axis_cache", state)
        self.assertEqual(
            cresso_v5.count_optimizer_state_elements(opt),
            cresso_v5.theoretical_state_elements(shape, rank=4, hash_bins=16, hash_tables=1),
        )

    def test_cuda_thin_scalar_update_matches_pytorch_reference_path(self) -> None:
        shape = (128, 16)
        device = torch.device("cuda")
        dtype = torch.float32
        torch.manual_seed(2468)
        initial = torch.randn(shape, device=device, dtype=dtype)
        grads = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
        cuda_param = torch.nn.Parameter(initial.clone())
        ref_param = torch.nn.Parameter(initial.clone())
        common_kwargs = dict(
            lr=2.0e-3,
            rank=4,
            thin_matrix_route="scalar",
            thin_matrix_max_width=64,
            min_spectral_size=1,
        )

        ops = cresso_v5._load_cuda_ops(required=True)
        self.assertTrue(hasattr(ops, "scalar_update_2d"))
        cuda_opt = cresso_v5.CRESSO5([cuda_param], cuda_ops="required", **common_kwargs)
        ref_opt = cresso_v5.CRESSO5([ref_param], cuda_ops="off", **common_kwargs)

        for grad in grads:
            cuda_param.grad = grad.clone()
            ref_param.grad = grad.clone()
            cuda_opt.step()
            ref_opt.step()
        torch.cuda.synchronize()

        torch.testing.assert_close(cuda_param, ref_param, rtol=8e-4, atol=8e-5)
        cuda_state = cuda_opt.state[cuda_param]
        ref_state = ref_opt.state[ref_param]
        for name in ("q", "impulse", "metric_q", "metric_impulse", "force_rms", "drive_energy", "surprise", "action"):
            torch.testing.assert_close(cuda_state[name], ref_state[name], rtol=8e-4, atol=8e-5)

    def test_cuda_thin_scalar_update_supports_fp16_and_bf16_params(self) -> None:
        shape = (64, 16)
        device = torch.device("cuda")
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            with self.subTest(dtype=str(dtype)):
                torch.manual_seed(3579)
                initial = (0.05 * torch.randn(shape, device=device, dtype=torch.float32)).to(dtype)
                grads = [(0.05 * torch.randn(shape, device=device, dtype=torch.float32)).to(dtype) for _ in range(2)]
                cuda_param = torch.nn.Parameter(initial.clone())
                ref_param = torch.nn.Parameter(initial.clone())
                common_kwargs = dict(
                    lr=1.0e-3,
                    rank=4,
                    thin_matrix_route="scalar",
                    thin_matrix_max_width=64,
                    min_spectral_size=1,
                )
                cuda_opt = cresso_v5.CRESSO5([cuda_param], cuda_ops="required", **common_kwargs)
                ref_opt = cresso_v5.CRESSO5([ref_param], cuda_ops="off", **common_kwargs)

                for grad in grads:
                    cuda_param.grad = grad.clone()
                    ref_param.grad = grad.clone()
                    cuda_opt.step()
                    ref_opt.step()
                torch.cuda.synchronize()

                atol = 2.5e-3 if dtype is torch.float16 else 2.5e-2
                torch.testing.assert_close(cuda_param.float(), ref_param.float(), rtol=0.0, atol=atol)
                cuda_state = cuda_opt.state[cuda_param]
                ref_state = ref_opt.state[ref_param]
                self.assertEqual(cuda_state["q"].dtype, torch.float32)
                for name in ("q", "impulse", "metric_q", "metric_impulse", "force_rms", "drive_energy", "surprise", "action"):
                    torch.testing.assert_close(cuda_state[name], ref_state[name], rtol=5e-3, atol=5e-4)

    def test_cuda_ops_required_does_not_silently_skip_mixed_precision_scalar_op(self) -> None:
        device = torch.device("cuda")
        p = torch.nn.Parameter(torch.randn((32, 8), device=device, dtype=torch.float16))
        p.grad = torch.randn_like(p)
        opt = cresso_v5.CRESSO5(
            [p],
            cuda_ops="required",
            rank=4,
            thin_matrix_route="scalar",
            thin_matrix_max_width=64,
            min_spectral_size=1,
        )
        with mock.patch.object(cresso_v5, "_load_cuda_ops", return_value=types.SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "scalar_update_2d"):
                opt.step()


if __name__ == "__main__":
    unittest.main()
