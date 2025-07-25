from typing import Callable, Self, cast

import torch
from torch import Tensor

type StandardProximityOperator[A] = Callable[[Tensor, float, A], Tensor]
type UnfoldedProximityOperator[A] = Callable[[Tensor, A], Tensor]

type ProximityOperator[A] = (
    StandardProximityOperator[A] | UnfoldedProximityOperator[A]
)

type StepSize = float

type ProximityBundle[ProximityArg] = tuple[
    ProximityOperator[ProximityArg], bool, float, ProximityArg
]

type LinearOperator[B] = Callable[[Tensor, B], Tensor]
type ConjugateLinearOperator[B] = LinearOperator[B]
type LinearBundle[LinearArg] = tuple[
    LinearOperator[LinearArg], ConjugateLinearOperator[LinearArg], LinearArg
]


def _primal_dual_iteration[X, Y, Z](
    primal_variable: Tensor,
    dual_variable: Tensor,
    primal_proximity_bundle: ProximityBundle[X],
    dual_proximity_bundle: ProximityBundle[Y],
    linear_bundle: LinearBundle[Z],
) -> tuple[Tensor, Tensor]:
    linear_operator, conjugate_linear_operator, arg = linear_bundle
    _, _, primal_step_size, _ = primal_proximity_bundle
    _, _, dual_step_size, _ = dual_proximity_bundle

    primal = _apply_proximity(
        primal_variable
        - primal_step_size * conjugate_linear_operator(dual_variable, arg),
        primal_proximity_bundle,
    )

    tmp = 2 * primal - primal_variable

    dual = _apply_proximity(
        dual_variable + dual_step_size * linear_operator(tmp, arg),
        dual_proximity_bundle,
    )

    return primal, dual


def _apply_proximity[X](
    input: Tensor,
    proximity_bundle: ProximityBundle[X],
) -> Tensor:
    operator, requires_step_size, step_size, arg = proximity_bundle

    if requires_step_size:
        return cast(StandardProximityOperator[X], operator)(
            input, step_size, arg
        )
    else:
        return cast(UnfoldedProximityOperator[X], operator)(input, arg)


class PrimalDualSchema[X, Y, Z]:
    def __init__(self) -> None:
        self.__fixed_components: dict[str, bool] = {
            "primal proximity": False,
            "primal argument": False,
            "dual proximity": False,
            "dual argument": False,
            "step sizes": False,
            "linear operator": False,
            "linear argument": False,
        }

        self.__is_fixed = False

        self.__primal_proximity: ProximityOperator[X]
        self.__primal_requires_step_size: bool
        self.__dual_proximity: ProximityOperator[Y]
        self.__dual_requires_step_size: bool
        self.__primal_step_size: float
        self.__dual_step_size: float
        self.__linear_operator: LinearOperator[Z]
        self.__conjugate_linear_operator: LinearOperator[Z]

    def not_fixed(self) -> list[str]:
        return [k for k, v in self.__fixed_components.items() if not v]

    def is_fixed(self) -> bool:
        if self.__is_fixed:
            return True

        self.__is_fixed = len(self.not_fixed()) == 0

        return self.__is_fixed

    def with_primal_proximity(
        self, proximity_operator: ProximityOperator[X], requires_step_size: bool
    ) -> Self:
        self.__primal_proximity = proximity_operator
        self.__primal_requires_step_size = requires_step_size
        self.__fixed_components["primal proximity"] = True

        return self

    def with_dual_proximity(
        self, proximity_operator: ProximityOperator[Y], requires_step_size: bool
    ) -> Self:
        self.__dual_proximity = proximity_operator
        self.__dual_requires_step_size = requires_step_size
        self.__fixed_components["dual proximity"] = True

        return self

    def with_linear_operator(
        self,
        linear_operator: LinearOperator[Z],
        conjugate_linear_operator: LinearOperator[Z],
    ) -> Self:
        self.__linear_operator = linear_operator
        self.__conjugate_linear_operator = conjugate_linear_operator

        self.__fixed_components["linear operator"] = True

        return self

    def with_step_sizes(
        self, primal_step_size: float, dual_step_size: float
    ) -> Self:
        self.__primal_step_size = primal_step_size
        self.__dual_step_size = dual_step_size

        self.__fixed_components["step sizes"] = True

        return self

    def with_primal_argument(self, primal_argument: X) -> Self:
        self.__primal_argument = primal_argument

        self.__fixed_components["primal argument"] = True

        return self

    def with_dual_argument(self, dual_argument: Y) -> Self:
        self.__dual_argument = dual_argument

        self.__fixed_components["dual argument"] = True

        return self

    def with_linear_argument(self, linear_argument: Z) -> Self:
        self.__linear_argument = linear_argument

        self.__fixed_components["linear argument"] = True

        return self

    def run(
        self,
        initial_primal_variable: Tensor,
        initial_dual_variable: Tensor,
        n_steps: int,
    ) -> tuple[Tensor, Tensor]:
        if not self.is_fixed():
            formatted_missing_arguments = {
                ", ".join(s.capitalize() for s in self.not_fixed())
            }
            raise ValueError(
                f"Trying to build primal-dual schema without fixing all"
                f" necessary arguments! Missing: {formatted_missing_arguments}."
            )

        primal_bundle = (
            self.__primal_proximity,
            self.__primal_requires_step_size,
            self.__primal_step_size,
            self.__primal_argument,
        )

        dual_bundle = (
            self.__dual_proximity,
            self.__dual_requires_step_size,
            self.__dual_step_size,
            self.__dual_argument,
        )

        linear_bundle = (
            self.__linear_operator,
            self.__conjugate_linear_operator,
            self.__linear_argument,
        )

        return self._run(
            initial_primal_variable,
            initial_dual_variable,
            primal_bundle,
            dual_bundle,
            linear_bundle,
            n_steps,
        )

    def _run(
        self,
        initial_primal_variable: Tensor,
        initial_dual_variable: Tensor,
        primal_bundle: ProximityBundle[X],
        dual_bundle: ProximityBundle[Y],
        linear_bundle: LinearBundle[Z],
        n_steps: int,
    ) -> tuple[Tensor, Tensor]:
        primal_variable = initial_primal_variable
        dual_variable = initial_dual_variable

        for _ in range(n_steps):
            primal_variable, dual_variable = _primal_dual_iteration(
                primal_variable=primal_variable,
                dual_variable=dual_variable,
                primal_proximity_bundle=primal_bundle,
                dual_proximity_bundle=dual_bundle,
                linear_bundle=linear_bundle,
            )

        return primal_variable, dual_variable
