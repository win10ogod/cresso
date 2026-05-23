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


class CressoV5RoutingTests(unittest.TestCase):
    def test_lora_thin_matrix_uses_scalar_route_by_default(self) -> None:
        p = torch.nn.Parameter(torch.randn(128, 16))
        p.grad = torch.randn_like(p)
        opt = cresso_v5.CRESSO5([p], lr=2.0e-3, rank=4)

        opt.step()

        state = opt.state[p]
        self.assertIn("q", state)
        self.assertNotIn("q_cos", state)
        self.assertNotIn("freqs", state)
        self.assertEqual(cresso_v5.count_optimizer_state_elements(opt), 8)
        self.assertEqual(cresso_v5.theoretical_state_elements(tuple(p.shape), rank=4), 8)

    def test_thin_matrix_spectral_route_can_be_requested_explicitly(self) -> None:
        p = torch.nn.Parameter(torch.randn(128, 16))
        p.grad = torch.randn_like(p)
        opt = cresso_v5.CRESSO5([p], lr=2.0e-3, rank=4, thin_matrix_route="spectral")

        opt.step()

        state = opt.state[p]
        self.assertIn("q_cos", state)
        self.assertIn("freqs", state)
        self.assertEqual(
            cresso_v5.count_optimizer_state_elements(opt),
            cresso_v5.theoretical_state_elements(tuple(p.shape), rank=4, thin_matrix_route="spectral"),
        )


if __name__ == "__main__":
    unittest.main()
