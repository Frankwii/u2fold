from pathlib import Path
from typing import Literal, Self

import PIL.Image
from pydantic import BaseModel, Field, field_validator, model_validator

from u2fold.utils.path import compute_output_paths


class ExecSpec(BaseModel):
    mode: Literal["exec"]
    input: list[Path] = Field(
        title="Input images",
        description="A list with the paths to the images to be processed."
    )
    output_dir: Path = Field(
        title="Directory for output images",
        description="Where to store the processed images. The output images "
        "will have the same names as the inputs, but paths relative to this "
        "directory."
    )
    override_dir_contents: bool = Field(
        default=False,
        description="Whether to override existing files in `output_dir` with the "
        "same names as the inputs.",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        return Path(value).resolve()

    @field_validator("input", mode="after")
    @classmethod
    def validate_inputs(cls, value: list[Path]):
        for img_path in value:
            if not img_path.exists():
                raise FileNotFoundError(f"Input image file not found: {img_path}")
            # Check whether the image is valid without fully loading it
            # (open is lazy)
            with PIL.Image.open(img_path):
                pass
        return value

    @model_validator(mode="after")
    def validate_output_dir(self) -> Self:
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if not self.override_dir_contents:
            for output_path in compute_output_paths(self.output_dir, *self.input):
                if output_path.exists():
                    raise FileExistsError(
                        f"Input image `{output_path.name}` already exists: {output_path}."
                    )

        return self
