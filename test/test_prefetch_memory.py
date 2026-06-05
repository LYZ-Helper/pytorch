import os
import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestPrefetchMemory(TestCase):
    """Test case for torch_npu.prefetch_memory_() which invokes
    aclrtMemManagedPrefetchAsync to prefetch tensor data via
    managed memory API (H2D).
    """

    @SupportedDevices(['Ascend910B'])
    def _prefetch_and_verify(self, input_shape, non_blocking=False):
        """Helper: create tensor on NPU, call prefetch_memory_, verify data intact."""
        _, npu_input = create_common_tensor(input_shape, -100, 100)
        original = npu_input.cpu().numpy()
        torch_npu.prefetch_memory_(npu_input, non_blocking=non_blocking)
        result = npu_input.cpu().numpy()
        self.assertRtolEqual(original, result)

    @SupportedDevices(['Ascend910B'])
    def test_prefetch_memory_blocking(self):
        """Test blocking prefetch with various tensor shapes and dtypes."""
        shapes_and_dtypes = [
            [np.float16, -1, (4, 4)],
            [np.float32, -1, (8, 8)],
            [np.float16, -1, (2, 3, 4)],
            [np.int32,   -1, (16,)],
            [np.float32, -1, (1, 1024, 1024)],
        ]
        for shape in shapes_and_dtypes:
            with self.subTest(shape=shape):
                self._prefetch_and_verify(shape, non_blocking=False)

    @SupportedDevices(['Ascend910B'])
    def test_prefetch_memory_non_blocking(self):
        """Test non-blocking prefetch: data intact after synchronize."""
        input_shape = [np.float16, -1, (8, 8)]
        _, npu_input = create_common_tensor(input_shape, -100, 100)
        original = npu_input.cpu().numpy()
        torch_npu.prefetch_memory_(npu_input, non_blocking=True)
        torch.npu.synchronize()
        result = npu_input.cpu().numpy()
        self.assertRtolEqual(original, result)

    @SupportedDevices(['Ascend910B'])
    def test_prefetch_memory_empty_tensor(self):
        """Test prefetch_memory_ on an empty tensor."""
        empty_tensor = torch.empty(0, device='npu', dtype=torch.float16)
        torch_npu.prefetch_memory_(empty_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_prefetch_memory_cpu_tensor_raises(self):
        """Test that prefetch_memory_ on a CPU tensor raises RuntimeError."""
        cpu_tensor = torch.randn(4, 4)
        with self.assertRaises(RuntimeError):
            torch_npu.prefetch_memory_(cpu_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_prefetch_memory_multiple_calls(self):
        """Test prefetch_memory_ called multiple times is safe."""
        input_shape = [np.float16, -1, (16, 16)]
        _, npu_input = create_common_tensor(input_shape, -100, 100)
        original = npu_input.cpu().numpy()
        for _ in range(5):
            torch_npu.prefetch_memory_(npu_input)
        result = npu_input.cpu().numpy()
        self.assertRtolEqual(original, result)

    @SupportedDevices(['Ascend910B'])
    def test_prefetch_memory_large_tensor(self):
        """Test prefetch on a relatively large tensor."""
        input_shape = [np.float32, -1, (512, 512)]
        self._prefetch_and_verify(input_shape, non_blocking=False)


if __name__ == "__main__":
    run_tests()
