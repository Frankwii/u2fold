import os
from pathlib import Path
from typing import Iterator


def compute_output_paths(output_dir: Path, *inputs: Path) -> Iterator[Path]:
    common_path = Path(os.path.commonpath([*inputs, output_dir]))
    for img_path in inputs:
        yield output_dir / img_path.relative_to(common_path)
