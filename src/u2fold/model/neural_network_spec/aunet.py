from typing import Literal, final

from u2fold.model.neural_network_spec.generic import UNetLikeSpec

@final
class AUNetSpec(UNetLikeSpec):  # pyright: ignore[reportUninitializedInstanceVariable]
    """Config for an AUNet architecture"""
    name: Literal["aunet"]
