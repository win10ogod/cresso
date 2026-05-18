from __future__ import annotations

import importlib.util
import unittest
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


if __name__ == "__main__":
    unittest.main()
