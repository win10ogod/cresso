from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
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
            mock.patch.object(cresso_cuda_ext, "_validate_cuda_runtime_match", return_value=None), \
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
            mock.patch.object(cresso_cuda_ext, "_validate_cuda_runtime_match", return_value=None), \
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
                mock.patch.object(cresso_cuda_ext, "_validate_cuda_runtime_match", return_value=None), \
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
            mock.patch.object(cresso_cuda_ext, "_validate_cuda_runtime_match", return_value=None), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=False), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version", return_value="Cuda compilation tools, release 12.0"), \
            self.assertRaisesRegex(RuntimeError, "TORCH_CUDA_ARCH_LIST.*sm_120"):
            cresso_cuda_ext._configure_cuda_arch_list()

    def test_override_arch_list_cannot_bypass_native_arch_guard(self) -> None:
        env = {"CRESSO_CUDA_ARCH_LIST": "9.0+PTX"}
        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_path", return_value="/usr/bin/nvcc"), \
            mock.patch.object(cresso_cuda_ext, "_validate_cuda_runtime_match", return_value=None), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch", return_value=False), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version", return_value="Cuda compilation tools, release 12.0"), \
            self.assertRaisesRegex(RuntimeError, "CRESSO_CUDA_ARCH_LIST.*sm_120"):
            cresso_cuda_ext._configure_cuda_arch_list()

    def test_default_nvcc_prefers_pytorch_runtime_major_and_native_arch(self) -> None:
        env: dict[str, str] = {}
        cu128 = "/usr/local/cuda-12.8/bin/nvcc"
        cu130 = "/home/win10/.local/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"

        def version_for(path: str | None):
            return (12, 8) if path == cu128 else (13, 0)

        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext, "_candidate_nvcc_paths", return_value=[cu128, cu130]), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "is_available", return_value=True), \
            mock.patch.object(cresso_cuda_ext.torch.cuda, "get_device_capability", return_value=(12, 0)), \
            mock.patch.object(cresso_cuda_ext, "_torch_cuda_version_tuple", return_value=(13, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version_tuple", side_effect=version_for), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_supports_device_arch_path", side_effect=lambda path, *_: path == cu130):
            self.assertEqual(cresso_cuda_ext._nvcc_path(), cu130)

    def test_windows_nvcc_from_home_accepts_exe_and_bin_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bin_dir = home / "bin"
            bin_dir.mkdir()
            nvcc = bin_dir / "nvcc.exe"
            nvcc.write_text("", encoding="utf-8")

            with mock.patch.object(cresso_cuda_ext, "_is_windows", return_value=True):
                self.assertEqual(cresso_cuda_ext._nvcc_from_home(str(home)), str(nvcc))
                self.assertEqual(cresso_cuda_ext._nvcc_from_home(str(bin_dir)), str(nvcc))

    def test_windows_candidates_prefer_torch_cuda_minor_before_stale_cuda_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cu128 = root / "cuda128"
            stale = root / "cuda121"
            for home in (cu128, stale):
                bin_dir = home / "bin"
                bin_dir.mkdir(parents=True)
                (bin_dir / "nvcc.exe").write_text("", encoding="utf-8")

            env = {
                "CUDA_PATH_V12_8": str(cu128),
                "CUDA_HOME": str(stale),
            }
            with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(cresso_cuda_ext, "_is_windows", return_value=True), \
                mock.patch.object(cresso_cuda_ext, "_torch_cuda_version_tuple", return_value=(12, 8)):
                candidates = cresso_cuda_ext._candidate_nvcc_paths()

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0], str(cu128 / "bin" / "nvcc.exe"))
        self.assertIn(str(stale / "bin" / "nvcc.exe"), candidates)

    def test_windows_build_flags_do_not_pass_unix_rpath_or_gcc_o3_to_msvc(self) -> None:
        with mock.patch.object(cresso_cuda_ext, "_is_windows", return_value=True):
            self.assertEqual(cresso_cuda_ext._extra_cflags(), ["/O2"])
            self.assertIn("--diag_suppress=221", cresso_cuda_ext._extra_cuda_cflags())
            self.assertEqual(cresso_cuda_ext._extra_ldflags(Path("C:/CUDA/lib/x64")), [])

        with mock.patch.object(cresso_cuda_ext, "_is_windows", return_value=False):
            self.assertEqual(cresso_cuda_ext._extra_cflags(), ["-O3"])
            self.assertNotIn("--diag_suppress=221", cresso_cuda_ext._extra_cuda_cflags())
            lib = Path("/usr/local/cuda/lib64")
            self.assertEqual(cresso_cuda_ext._extra_ldflags(lib), [f"-Wl,-rpath,{lib}"])

    def test_configure_cuda_toolkit_rejects_mixed_runtime_major(self) -> None:
        env: dict[str, str] = {"CUDACXX": "/usr/local/cuda-12.8/bin/nvcc"}
        with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(cresso_cuda_ext, "_torch_cuda_version_tuple", return_value=(13, 0)), \
            mock.patch.object(cresso_cuda_ext, "_nvcc_version_tuple", return_value=(12, 8)), \
            self.assertRaisesRegex(RuntimeError, "mixed CUDA runtime"):
            cresso_cuda_ext._configure_cuda_toolkit()

    def test_cuda_runtime_linker_name_is_created_for_pytorch_wheel_toolkits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            lib = home / "lib"
            lib.mkdir()
            (lib / "libcudart.so.13").write_text("", encoding="utf-8")

            with mock.patch.object(cresso_cuda_ext, "_torch_cuda_version_tuple", return_value=(13, 0)), \
                mock.patch.object(cresso_cuda_ext, "_is_windows", return_value=False):
                if os.name == "nt":
                    with mock.patch.object(Path, "symlink_to") as symlink_to:
                        cresso_cuda_ext._ensure_cuda_runtime_linker_name(str(home))
                    symlink_to.assert_called_once_with("libcudart.so.13")
                else:
                    cresso_cuda_ext._ensure_cuda_runtime_linker_name(str(home))
                    self.assertTrue((lib / "libcudart.so").is_symlink())
                    self.assertEqual(os.readlink(lib / "libcudart.so"), "libcudart.so.13")

    def test_extension_module_name_is_arch_specific(self) -> None:
        self.assertNotEqual(
            cresso_cuda_ext._extension_module_name("12.0"),
            cresso_cuda_ext._extension_module_name("9.0+PTX"),
        )
        self.assertIn("12_0", cresso_cuda_ext._extension_module_name("12.0"))

    def test_extension_module_name_includes_source_hash(self) -> None:
        name = cresso_cuda_ext._extension_module_name("12.0", "deadbeef1234")
        self.assertIn("deadbeef1234", name)

    def test_extension_abi_tag_forces_rebuild_after_loader_safety_changes(self) -> None:
        name = cresso_cuda_ext._extension_module_name("12.0", "deadbeef1234")

        self.assertIn("safe8", name)
        self.assertNotIn("safe7", name)

    def test_stable_source_paths_copy_sources_into_linux_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "src"
            src_dir.mkdir()
            cpp = src_dir / "cresso_cuda.cpp"
            cu = src_dir / "cresso_cuda_kernel.cu"
            cpp.write_text("// cpp\n", encoding="utf-8")
            cu.write_text("// cu\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"CRESSO_CUDA_BUILD_ROOT": str(Path(tmp) / "cache")}, clear=True):
                copied = cresso_cuda_ext._stable_source_paths([str(cpp), str(cu)], "abc123")

            copied_paths = [Path(path) for path in copied]
            self.assertEqual([path.name for path in copied_paths], ["cresso_cuda.cpp", "cresso_cuda_kernel.cu"])
            for path in copied_paths:
                self.assertTrue(path.is_file())
                self.assertIn("abc123", path.as_posix())
                self.assertNotEqual(path.parent, src_dir)
            self.assertEqual(copied_paths[0].read_text(encoding="utf-8"), "// cpp\n")
            self.assertEqual(copied_paths[1].read_text(encoding="utf-8"), "// cu\n")

    def test_build_directory_is_under_cresso_cache_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"CRESSO_CUDA_BUILD_ROOT": tmp}, clear=True):
                build_dir = cresso_cuda_ext._extension_build_directory("cresso_v5_cuda_test_safe8")

        self.assertTrue(build_dir.name.endswith("safe8"))
        self.assertIn(f"py{sys.version_info.major}{sys.version_info.minor}", build_dir.as_posix())

    def test_build_parallelism_defaults_to_single_job_but_respects_existing_max_jobs(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cresso_cuda_ext._configure_build_parallelism()
            self.assertEqual(os.environ["MAX_JOBS"], "1")

        with mock.patch.dict(os.environ, {"MAX_JOBS": "3"}, clear=True):
            cresso_cuda_ext._configure_build_parallelism()
            self.assertEqual(os.environ["MAX_JOBS"], "3")

        with mock.patch.dict(os.environ, {"CRESSO_CUDA_MAX_JOBS": "2"}, clear=True):
            cresso_cuda_ext._configure_build_parallelism()
            self.assertEqual(os.environ["MAX_JOBS"], "2")


if __name__ == "__main__":
    unittest.main()
