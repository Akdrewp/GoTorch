import collections.abc
import typing
from typing import ClassVar, overload

CPU: Device
GPU: Device
__backend_type__: str

class Device:
    """Members:

      CPU

      GPU"""
    __members__: ClassVar[dict] = ...  # read-only
    CPU: ClassVar[Device] = ...
    GPU: ClassVar[Device] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: GoTorch.backend.native_backend.Device, value: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(*args, **kwargs)
        Overloaded function.

        1. __eq__(self: GoTorch.backend.native_backend.Device, other: GoTorch.backend.native_backend.Device, /) -> bool

        2. __eq__(self: GoTorch.backend.native_backend.Device, other: object, /) -> bool
        """
    def __hash__(self) -> int:
        """__hash__(self: object, /) -> int"""
    def __index__(self) -> int:
        """__index__(self: GoTorch.backend.native_backend.Device, /) -> int"""
    def __int__(self) -> int:
        """__int__(self: GoTorch.backend.native_backend.Device, /) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(*args, **kwargs)
        Overloaded function.

        1. __ne__(self: GoTorch.backend.native_backend.Device, other: GoTorch.backend.native_backend.Device, /) -> bool

        2. __ne__(self: GoTorch.backend.native_backend.Device, other: object, /) -> bool
        """
    @property
    def name(self) -> str:
        """name(self: object, /) -> str

        name(self: object, /) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: GoTorch.backend.native_backend.Device) -> int"""

class NDArray:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: GoTorch.backend.native_backend.NDArray) -> None

        2. __init__(self: GoTorch.backend.native_backend.NDArray, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], strides: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] = [], offset: typing.SupportsInt | typing.SupportsIndex = 0) -> None
        """
    @overload
    def __init__(self, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], strides: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] = ..., offset: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: GoTorch.backend.native_backend.NDArray) -> None

        2. __init__(self: GoTorch.backend.native_backend.NDArray, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], strides: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] = [], offset: typing.SupportsInt | typing.SupportsIndex = 0) -> None
        """
    def BCELoss(self, target: NDArray, reduction: str = ...) -> NDArray:
        """BCELoss(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def BCEWithLogitsLoss(self, target: NDArray, reduction: str = ...) -> NDArray:
        """BCEWithLogitsLoss(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def BCEwithLogitsLoss(self, target: NDArray, reduction: str = ...) -> NDArray:
        """BCEwithLogitsLoss(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def add(self, addend: NDArray) -> NDArray:
        """add(self: GoTorch.backend.native_backend.NDArray, addend: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def bce_loss_backward(self, target: NDArray, grad_output: NDArray = ..., reduction: str = ...) -> NDArray:
        """bce_loss_backward(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray = <GoTorch.backend.native_backend.NDArray object at 0x792e933da270>, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def bce_with_logits_loss_backward(self, target: NDArray, grad_output: NDArray = ..., reduction: str = ...) -> NDArray:
        """bce_with_logits_loss_backward(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray = <GoTorch.backend.native_backend.NDArray object at 0x792e92a24230>, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def isEmpty(self) -> bool:
        """isEmpty(self: GoTorch.backend.native_backend.NDArray) -> bool"""
    def is_contiguous(self) -> bool:
        """is_contiguous(self: GoTorch.backend.native_backend.NDArray) -> bool"""
    def leaky_relu(self, alpha: typing.SupportsFloat | typing.SupportsIndex = ...) -> NDArray:
        """leaky_relu(self: GoTorch.backend.native_backend.NDArray, alpha: typing.SupportsFloat | typing.SupportsIndex = 0.009999999776482582) -> GoTorch.backend.native_backend.NDArray"""
    def leaky_relu_backward(self, grad_output: NDArray, alpha: typing.SupportsFloat | typing.SupportsIndex = ...) -> NDArray:
        """leaky_relu_backward(self: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray, alpha: typing.SupportsFloat | typing.SupportsIndex = 0.009999999776482582) -> GoTorch.backend.native_backend.NDArray"""
    def matmul(self, other: NDArray) -> NDArray:
        """matmul(self: GoTorch.backend.native_backend.NDArray, other: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def matmul_coalesced(self, other: NDArray) -> NDArray:
        """matmul_coalesced(self: GoTorch.backend.native_backend.NDArray, other: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def matmul_coalesced_coarsened(self, other: NDArray) -> NDArray:
        """matmul_coalesced_coarsened(self: GoTorch.backend.native_backend.NDArray, other: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def matmul_naive(self, other: NDArray) -> NDArray:
        """matmul_naive(self: GoTorch.backend.native_backend.NDArray, other: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def matmul_tiled(self, other: NDArray) -> NDArray:
        """matmul_tiled(self: GoTorch.backend.native_backend.NDArray, other: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def mse_loss(self, target: NDArray, reduction: str = ...) -> NDArray:
        """mse_loss(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def mse_loss_backward(self, target: NDArray, grad_output: NDArray = ..., reduction: str = ...) -> NDArray:
        """mse_loss_backward(self: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray = <GoTorch.backend.native_backend.NDArray object at 0x792e92bb7d70>, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
    def ones_like(self) -> NDArray:
        """ones_like(self: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def relu(self) -> NDArray:
        """relu(self: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def relu_backward(self, grad_output: NDArray) -> NDArray:
        """relu_backward(self: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def sigmoid(self) -> NDArray:
        """sigmoid(self: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def sigmoid_backward(self, grad_output: NDArray) -> NDArray:
        """sigmoid_backward(self: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def size(self) -> int:
        """size(self: GoTorch.backend.native_backend.NDArray) -> int"""
    def sub(self, subtrahend: NDArray) -> NDArray:
        """sub(self: GoTorch.backend.native_backend.NDArray, subtrahend: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def sum(self, dim: object = ..., keepdim: bool = ...) -> NDArray:
        """sum(self: GoTorch.backend.native_backend.NDArray, dim: object = None, keepdim: bool = False) -> GoTorch.backend.native_backend.NDArray"""
    def tanh(self) -> NDArray:
        """tanh(self: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def tanh_backward(self, grad_output: NDArray) -> NDArray:
        """tanh_backward(self: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def to(self, device: Device) -> NDArray:
        """to(self: GoTorch.backend.native_backend.NDArray, device: GoTorch.backend.native_backend.Device) -> GoTorch.backend.native_backend.NDArray"""
    def transpose(self, dim0: typing.SupportsInt | typing.SupportsIndex = ..., dim1: typing.SupportsInt | typing.SupportsIndex = ...) -> NDArray:
        """transpose(self: GoTorch.backend.native_backend.NDArray, dim0: typing.SupportsInt | typing.SupportsIndex = 0, dim1: typing.SupportsInt | typing.SupportsIndex = 1) -> GoTorch.backend.native_backend.NDArray"""
    def zeros_like(self) -> NDArray:
        """zeros_like(self: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.NDArray"""
    def __add__(self, arg0: object) -> object:
        """__add__(self: GoTorch.backend.native_backend.NDArray, arg0: object) -> object"""
    def __getitem__(self, arg0: object) -> float:
        """__getitem__(self: GoTorch.backend.native_backend.NDArray, arg0: object) -> float"""
    def __radd__(self, arg0: object) -> object:
        """__radd__(self: GoTorch.backend.native_backend.NDArray, arg0: object) -> object"""
    def __rsub__(self, arg0: object) -> object:
        """__rsub__(self: GoTorch.backend.native_backend.NDArray, arg0: object) -> object"""
    def __setitem__(self, arg0: object, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """__setitem__(self: GoTorch.backend.native_backend.NDArray, arg0: object, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    def __sub__(self, arg0: object) -> object:
        """__sub__(self: GoTorch.backend.native_backend.NDArray, arg0: object) -> object"""
    @property
    def data(self) -> list[float]:
        """(arg0: GoTorch.backend.native_backend.NDArray) -> list[float]"""
    @property
    def device(self) -> Device:
        """(arg0: GoTorch.backend.native_backend.NDArray) -> GoTorch.backend.native_backend.Device"""
    @property
    def offset(self) -> int:
        """(arg0: GoTorch.backend.native_backend.NDArray) -> int"""
    @property
    def shape(self) -> tuple:
        """(arg0: GoTorch.backend.native_backend.NDArray) -> tuple"""
    @property
    def strides(self) -> tuple:
        """(arg0: GoTorch.backend.native_backend.NDArray) -> tuple"""

def BCELoss(in_features: NDArray, true_features: NDArray, reduction: str = ...) -> NDArray:
    """BCELoss(in_features: GoTorch.backend.native_backend.NDArray, true_features: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
def BCEWithLogitsLoss(in_features: NDArray, true_features: NDArray, reduction: str = ...) -> NDArray:
    """BCEWithLogitsLoss(in_features: GoTorch.backend.native_backend.NDArray, true_features: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
def BCEwithLogitsLoss(in_features: NDArray, true_features: NDArray, reduction: str = ...) -> NDArray:
    """BCEwithLogitsLoss(in_features: GoTorch.backend.native_backend.NDArray, true_features: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
def bce_loss_backward(in_features: NDArray, true_features: NDArray, grad_output: NDArray = ..., reduction: str = ...) -> NDArray:
    """bce_loss_backward(in_features: GoTorch.backend.native_backend.NDArray, true_features: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray = <NDArray shape=(0,)>, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
def bce_with_logits_loss_backward(in_features: NDArray, true_features: NDArray, grad_output: NDArray = ..., reduction: str = ...) -> NDArray:
    """bce_with_logits_loss_backward(in_features: GoTorch.backend.native_backend.NDArray, true_features: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray = <NDArray shape=(0,)>, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
def mse_loss(prediction: NDArray, target: NDArray, reduction: str = ...) -> NDArray:
    """mse_loss(prediction: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
def mse_loss_backward(prediction: NDArray, target: NDArray, grad_output: NDArray = ..., reduction: str = ...) -> NDArray:
    """mse_loss_backward(prediction: GoTorch.backend.native_backend.NDArray, target: GoTorch.backend.native_backend.NDArray, grad_output: GoTorch.backend.native_backend.NDArray = <NDArray shape=(0,)>, reduction: str = 'mean') -> GoTorch.backend.native_backend.NDArray"""
