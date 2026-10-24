"""Unit and integration tests for Device enum, memory transfers, and GPU parity."""

from __future__ import annotations

import pytest

from GoTorch import Device, Tensor
from GoTorch.backend.native_backend import NDArray


class TestNDArrayDevice:
    """Verifies NDArray device properties, memory safety, and cross-device round-trips."""

    def test_default_device_is_cpu(self) -> None:
        """Verifies default constructed and standard NDArrays reside on CPU."""
        empty_arr = NDArray()
        assert empty_arr.device == Device.CPU

        arr = NDArray([1.0, 2.0, 3.0], shape=(3,))
        assert arr.device == Device.CPU

    def test_to_cpu_identity(self) -> None:
        """Verifies transferring a CPU NDArray to CPU returns the array on CPU."""
        arr = NDArray([1.0, 2.0, 3.0], shape=(3,))
        arr_cpu = arr.to(Device.CPU)
        assert arr_cpu.device == Device.CPU
        assert list(arr_cpu.data) == [1.0, 2.0, 3.0]

    def test_cpu_to_gpu_and_back_preserves_data(self) -> None:
        """Verifies transferring contiguous data from CPU to GPU and back preserves elements."""
        original_data = [1.5, -2.5, 3.25, 4.0, 0.0, -10.125]
        arr_cpu = NDArray(original_data, shape=(2, 3))
        assert arr_cpu.device == Device.CPU

        arr_gpu = arr_cpu.to(Device.GPU)
        assert arr_gpu.device == Device.GPU
        assert arr_gpu.shape == (2, 3)

        arr_roundtrip = arr_gpu.to(Device.CPU)
        assert arr_roundtrip.device == Device.CPU
        assert arr_roundtrip.shape == (2, 3)
        assert list(arr_roundtrip.data) == pytest.approx(original_data)

    def test_gpu_direct_data_access_raises_runtime_error(self) -> None:
        """Verifies accessing .data directly on a GPU NDArray raises RuntimeError."""
        arr_gpu = NDArray([1.0, 2.0], shape=(2,)).to(Device.GPU)
        assert arr_gpu.device == Device.GPU
        with pytest.raises(
            RuntimeError, match="Direct access to GPU memory unavailable"
        ):
            _ = arr_gpu.data

    def test_non_contiguous_roundtrip_preserves_data(self) -> None:
        """Verifies non-contiguous strided views preserve data across CPU-GPU transfers."""
        arr = NDArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
        arr_transposed = arr.transpose(0, 1)
        assert not arr_transposed.is_contiguous()
        expected_data = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        assert list(arr_transposed.data) == expected_data

        arr_gpu = arr_transposed.to(Device.GPU)
        assert arr_gpu.device == Device.GPU
        assert arr_gpu.shape == (3, 2)

        arr_back = arr_gpu.to(Device.CPU)
        assert arr_back.device == Device.CPU
        assert arr_back.shape == (3, 2)
        assert list(arr_back.data) == pytest.approx(expected_data)

    def test_scalar_roundtrip(self) -> None:
        """Verifies 0-D scalar arrays preserve value and shape across device transfers."""
        scalar_cpu = NDArray([42.5], shape=())
        assert scalar_cpu.device == Device.CPU
        assert scalar_cpu.shape == ()

        scalar_gpu = scalar_cpu.to(Device.GPU)
        assert scalar_gpu.device == Device.GPU
        assert scalar_gpu.shape == ()

        scalar_back = scalar_gpu.to(Device.CPU)
        assert scalar_back.device == Device.CPU
        assert scalar_back.shape == ()
        assert list(scalar_back.data) == pytest.approx([42.5])

    def test_empty_array_roundtrip(self) -> None:
        """Verifies empty arrays preserve empty state across device transfers."""
        empty_cpu = NDArray([], shape=(0,))
        assert empty_cpu.isEmpty()
        assert empty_cpu.device == Device.CPU

        empty_gpu = empty_cpu.to(Device.GPU)
        assert empty_gpu.device == Device.GPU
        assert empty_gpu.isEmpty()

        empty_back = empty_gpu.to(Device.CPU)
        assert empty_back.device == Device.CPU
        assert empty_back.isEmpty()
        assert list(empty_back.data) == []


class TestDeviceMismatchExceptions:
    """Verifies that operations across heterogeneous devices raise appropriate exceptions."""

    def test_ndarray_matmul_cpu_gpu_mismatch(self) -> None:
        """Verifies NDArray.matmul raises ValueError when multiplying CPU and GPU arrays."""
        a = NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        b = NDArray([1.0, 0.0, 0.0, 1.0], shape=(2, 2)).to(Device.GPU)

        with pytest.raises(
            ValueError,
            match=r"Device mismatch in matmul.*left operand is on cpu but right operand is on gpu",
        ):
            a.matmul(b)

    def test_ndarray_matmul_gpu_cpu_mismatch(self) -> None:
        """Verifies NDArray.matmul raises ValueError when multiplying GPU and CPU arrays."""
        a = NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)).to(Device.GPU)
        b = NDArray([1.0, 0.0, 0.0, 1.0], shape=(2, 2))

        with pytest.raises(
            ValueError,
            match=r"Device mismatch in matmul.*left operand is on gpu but right operand is on cpu",
        ):
            a.matmul(b)

    def test_tensor_matmul_device_mismatch(self) -> None:
        """Verifies Tensor @ raises ValueError when operands reside on different devices."""
        t_cpu = Tensor(NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        t_gpu = Tensor(NDArray([1.0, 0.0, 0.0, 1.0], shape=(2, 2))).to(Device.GPU)

        with pytest.raises(ValueError, match=r"Device mismatch in matmul"):
            _ = t_cpu @ t_gpu

        with pytest.raises(ValueError, match=r"Device mismatch in matmul"):
            _ = t_gpu @ t_cpu


class TestTensorDeviceFrontend:
    """Verifies Tensor frontend device property and to(Device) method behavior."""

    def test_tensor_default_device(self) -> None:
        """Verifies newly constructed Tensor inherits underlying NDArray CPU device."""
        t = Tensor(NDArray([1.0, 2.0], shape=(2,)))
        assert t.device == Device.CPU

    def test_tensor_to_device(self) -> None:
        """Verifies Tensor.to(Device.GPU) transfers data and updates device property."""
        t = Tensor(NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        t_gpu = t.to(Device.GPU)

        assert t_gpu.device == Device.GPU
        assert t_gpu.data.device == Device.GPU
        assert t_gpu.shape == (2, 2)
        assert t_gpu.op is t.op
        assert t_gpu.inputs == t.inputs

        t_cpu = t_gpu.to(Device.CPU)
        assert t_cpu.device == Device.CPU
        assert t_cpu.data.device == Device.CPU
        assert list(t_cpu.data.data) == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_tensor_to_same_device_noop(self) -> None:
        """Verifies Tensor.to on already resident device preserves identity/device."""
        t = Tensor(NDArray([1.0, 2.0], shape=(2,)))
        t_same = t.to(Device.CPU)
        assert t_same.device == Device.CPU
        assert list(t_same.data.data) == [1.0, 2.0]


class TestMatMulParity:
    """Verifies matrix multiplication numerical parity between CPU and GPU executions."""

    def test_matmul_parity_square_matrices(self) -> None:
        """Verifies CPU and GPU matmul results match exactly for square matrices."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        b_data = [5.0, 6.0, 7.0, 8.0]
        shape = (2, 2)

        a_cpu = NDArray(a_data, shape=shape)
        b_cpu = NDArray(b_data, shape=shape)
        c_cpu = a_cpu.matmul(b_cpu)

        a_gpu = a_cpu.to(Device.GPU)
        b_gpu = b_cpu.to(Device.GPU)
        c_gpu = a_gpu.matmul(b_gpu)

        assert c_gpu.device == Device.GPU
        c_gpu_cpu = c_gpu.to(Device.CPU)
        assert c_gpu_cpu.shape == c_cpu.shape
        assert list(c_gpu_cpu.data) == pytest.approx(
            list(c_cpu.data), rel=1e-5, abs=1e-5
        )

    def test_matmul_parity_rectangular_matrices(self) -> None:
        """Verifies CPU and GPU matmul parity for non-square matrices (M x K @ K x N)."""
        a_data = [
            1.0, -1.0, 2.0, 0.5,
            0.0, 3.0, -2.0, 1.5,
            -0.5, 2.5, 1.0, -3.0,
        ]
        b_data = [
            2.0, 1.0,
            -1.0, 0.5,
            3.0, -2.0,
            0.0, 4.0,
        ]
        a_cpu = NDArray(a_data, shape=(3, 4))
        b_cpu = NDArray(b_data, shape=(4, 2))
        c_cpu = a_cpu.matmul(b_cpu)

        c_gpu = a_cpu.to(Device.GPU).matmul(b_cpu.to(Device.GPU))
        c_gpu_cpu = c_gpu.to(Device.CPU)

        assert c_gpu_cpu.shape == (3, 2)
        assert list(c_gpu_cpu.data) == pytest.approx(
            list(c_cpu.data), rel=1e-5, abs=1e-5
        )

    def test_matmul_parity_non_multiple_of_block_size(self) -> None:
        """Verifies GPU boundary threads execute accurately on non-multiples of 16-block size."""
        m, k, n = 17, 19, 23
        a_data = [float((i * 3 + j * 7) % 13 - 6) for i in range(m) for j in range(k)]
        b_data = [float((i * 5 + j * 11) % 17 - 8) for i in range(k) for j in range(n)]

        a_cpu = NDArray(a_data, shape=(m, k))
        b_cpu = NDArray(b_data, shape=(k, n))
        c_cpu = a_cpu.matmul(b_cpu)

        c_gpu = a_cpu.to(Device.GPU).matmul(b_cpu.to(Device.GPU))
        c_gpu_cpu = c_gpu.to(Device.CPU)

        assert c_gpu_cpu.shape == (m, n)
        assert list(c_gpu_cpu.data) == pytest.approx(
            list(c_cpu.data), rel=1e-5, abs=1e-5
        )

    def test_matmul_parity_transposed_non_contiguous(self) -> None:
        """Verifies GPU matmul handles non-contiguous transposed input tensors with correct parity."""
        a_cpu = NDArray(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3)
        ).transpose(0, 1)
        b_cpu = NDArray([2.0, -1.0, 0.5, 3.0], shape=(2, 2))
        c_cpu = a_cpu.matmul(b_cpu)

        a_gpu = a_cpu.to(Device.GPU)
        b_gpu = b_cpu.to(Device.GPU)
        c_gpu = a_gpu.matmul(b_gpu)
        c_gpu_cpu = c_gpu.to(Device.CPU)

        assert c_gpu_cpu.shape == (3, 2)
        assert list(c_gpu_cpu.data) == pytest.approx(
            list(c_cpu.data), rel=1e-5, abs=1e-5
        )

    def test_tensor_matmul_gpu_parity(self) -> None:
        """Verifies Tensor frontend @ operation on GPU produces results matching CPU evaluation."""
        a = Tensor(NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        b = Tensor(NDArray([2.0, 0.0, 1.0, 2.0], shape=(2, 2)))
        c_cpu = a @ b

        a_gpu = a.to(Device.GPU)
        b_gpu = b.to(Device.GPU)
        c_gpu = a_gpu @ b_gpu

        assert c_gpu.device == Device.GPU
        c_back = c_gpu.to(Device.CPU)
        assert list(c_back.data.data) == pytest.approx(
            list(c_cpu.data.data), rel=1e-5, abs=1e-5
        )
