from __future__ import annotations

import importlib.util
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "cresso_cuda_ext.py"
SPEC = importlib.util.spec_from_file_location("cresso_cuda_ext", SCRIPT_PATH)
cresso_cuda_ext = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(cresso_cuda_ext)


class CressoCudaExtArchTests(unittest.TestCase):
    def test_blackwell_requires_native_nvcc_arch_by_default(self) -> None:
        env: dict[str, str] = {}
        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_path", return_value="/usr/bin/nvcc"), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=False), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version", return_value="Cuda compilation tools, release 12.0"), \
            self.assertRaisesRegex(RuntimeError, "sm_120.*does not support"):
            cresso_cuda_ext._configure_cuda_arch_list()

        self.assertNotIn("TORCH_CUDA_ARCH_LIST", os.environ)

    def test_native_blackwell_arch_is_selected_when_nvcc_supports_it(self) -> None:
        env: dict[str, str] = {}
        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_path", return_value="/usr/local/cuda-12.8/bin/nvcc"), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=True):
            arch = cresso_cuda_ext._configure_cuda_arch_list()
            self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "12.0")

        self.assertEqual(arch, "12.0")

    def test_ptx_fallback_is_explicit_opt_in(self) -> None:
        env = {"CRESSO_CUDA_ALLOW_PTX_FALLBACK": "1"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
                mock.patch.object(cresso_cuda_ext, "_nvcc_path", return_value="/usr/bin/nvcc"), \
                mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=False), \
                mock.patch.object(cresso_cuda_ext, "_nvcc_version", return_value="Cuda compilation tools, release 12.0"):
                arch = cresso_cuda_ext._configure_cuda_arch_list()
                self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "9.0+PTX")

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, RuntimeWarning))
        self.assertIn("explicit PTX fallback", str(caught[0].message))

        self.assertEqual(arch, "9.0+PTX")

    def test_existing_torch_arch_list_cannot_bypass_native_arch_guard(self) -> None:
        env = {"TORCH_CUDA_ARCH_LIST": "9.0+PTX"}
        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_path", return_value="/usr/bin/nvcc"), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=False), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version", return_value="Cuda compilation tools, release 12.0"), \
            self.assertRaisesRegex(RuntimeError, "TORCH_CUDA_ARCH_LIST.*sm_120"):
            cresso_cuda_ext._configure_cuda_arch_list()

    def test_override_arch_list_cannot_bypass_native_arch_guard(self) -> None:
        env = {"CRESSO_CUDA_ARCH_LIST": "9.0+PTX"}
        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_path", return_value="/usr/bin/nvcc"), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=False), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version", return_value="Cuda compilation tools, release 12.0"), \
            self.assertRaisesRegex(RuntimeError, "CRESSO_CUDA_ARCH_LIST.*sm_120"):
            cresso_cuda_ext._configure_cuda_arch_list()

    def test_extension_module_name_is_arch_specific(self) -> None:
        self.assertNotEqual(
            cresso_cuda_ext._extension_module_name("12.0"),
            cresso_cuda_ext._extension_module_name("9.0+PTX"),
        )
        self.assertIn("12_0", cresso_cuda_ext._extension_module_name("12.0"))


if __name__ == "__main__":
    unittest.main()
