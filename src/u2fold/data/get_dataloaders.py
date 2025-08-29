from pathlib import Path
from typing import cast
from u2fold.data.dataloader_generics.base import U2FoldDataLoader
from u2fold.data.dataset_splits import SplitData
from u2fold.utils.track import get_from_tag


def get_dataloaders(
    dataset: str, dataset_path: Path, batch_size: int, device: str
) -> SplitData[U2FoldDataLoader]:  # pyright: ignore[reportMissingTypeArgument]
    dataloader_class = cast(
        U2FoldDataLoader, get_from_tag(f"data/dataloader/{dataset}")  # pyright: ignore[reportMissingTypeArgument]
    )

    return dataloader_class.get_dataloaders(dataset_path, batch_size, device)
