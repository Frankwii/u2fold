from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from torch import nn


class BasePoolingMethod(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    model_config = ConfigDict(frozen=True)

    kernel_size: int = Field(
        default=2,
        description=(
            "Size of the kernel to use for pooling. Refers to the side of the"
            " (square) kernel; not its radius."
        ),
    )
    stride: int = Field(
        default=2,
        description="Stride to use for pooling. The same for all directions.",
    )

    @abstractmethod
    def instantiate(self) -> nn.Module: ...

    def format_value(self) -> str:
        return f"{getattr(self, 'method')}-{self.kernel_size}-{self.stride}"
