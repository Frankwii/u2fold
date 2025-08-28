from pydantic import BaseModel, Field


class AlgorithmicSpec(BaseModel):
    """Specification of the "algorithmic" parts of the program.

    This includes the values used for the parameters of the "deterministic"
    part of the algorithm.
    """

    guided_filter_patch_radius: int = Field(
        ge=0,
        title="Patch radius for the guided filter",
        description="Patch radius used in the guided filter when refining the "
        "coarse transmission map. Defined as the number of pixels from the "
        "center to any side of the square, without counting the center itself.",
        examples=[12, 16, 20],
    )

    guided_filter_regularization_coefficient: float = Field(
        gt=0,
        title="Regularization coefficient for the guided filter",
        description="Coefficient used for weighing the square L^2 norm of "
        "patch means in the guided filter refinement of the coarse "
        "transmission map.",
        examples=[0.01, 0.0001],
    )

    step_size: float = Field(
        gt=0,
        title="Step size",
        description="Step size of the analytical proximity operator. That is, "
        "the step size used for the proximity operator associated to the "
        "dual variable.",
        examples=[0.01, 0.005],
    )

    transmission_map_patch_radius: int = Field(
        ge=0,
        title="Patch radius for the transmission map estimation",
        description="Patch radius used for estimating the coarse transmission "
        "map. Defined as the number of pixels from the center to any side of "
        "the square, without counting the center itself.",
        examples=[6, 8, 10],
    )

    transmission_map_saturation_coefficient: float = Field(
        ge=0,
        title="Saturation coefficient for the transmission map estimation",
        description="Coefficient used for weighing saturation maps when "
        "estimating transmission maps. This corresponds to \\lambda in "
        "https://doi.org/10.1109/TCSVT.2021.3115791.\n"
        "This should be a positive real number, ideally between 0 and 1 "
        "and close to 1.",
        examples=[0.5, 0.7],
    )

    greedy_iterations: int = Field(
        ge=1,
        title="Number of greedy iterations",
        description="Number of greedy iterations to perform in the top-level "
        "minimization scheme. That is, how many times each of the two "
        "variables is minimized.",
        examples=[3]
    )

    stages: int = Field(
        ge=1,
        title="Number of stages",
        description="Number of stages (iterations) to perform inside each "
        "greedy iteration. This conceptually corresponds to the number of "
        "iterations the (non-unfolded) primal-dual algorithm would have. "
        "The total number of neural networks in the overall algorithm is "
        "the number of greedy iterations times the number of stages.",
        examples=[3]
    )
