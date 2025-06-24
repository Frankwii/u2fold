from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils.track import tag


@tag("cli_argument/common/patch_radius", "affect_weights")
class PatchRadius(CLIArgument[int]):
    def short_name(self) -> str:
        return "-p"

    def long_name(self) -> str:
        return "--patch-radius"

    def help(self) -> str:
        return (
            "Radius of the square used for estimating the transmission maps."
            " That is, the number of pixels from the center to the side of the"
            " square, in a line parallel to one of the side of the square, "
            " without counting the center itself."
            "\n\n"
            " This is used both for the coarse transmission map and for the"
            " guided filter, as defined in"
            " https://doi.org/10.1109/TCSVT.2021.3115791."
        )

    def default(self) -> int:
        return 5

    def _validate_value(self, value: int) -> None:
        if not 0 < value <= 1000:
            raise ValueError("Invalid patch radius. It should be a positive"
                             " integer much smaller than image sizes.")


@tag("cli_argument/common/saturation_coefficient", "affect_weights")
class SaturationCoefficient(CLIArgument[float]):
    def short_name(self) -> str:
        return "-S"

    def long_name(self) -> str:
        return "--saturation-coefficient"

    def help(self) -> str:
        return (
            "Coefficient used for weighing saturation maps when estimating"
            " transmission maps. This corresponds to \\lambda in"
            " https://doi.org/10.1109/TCSVT.2021.3115791."
            "\n\n"
            " This should be a positive real number, ideally between 0 and 1"
            " and close to 1."
        )

    def default(self) -> float:
        return 0.9

    def _validate_value(self, value: float) -> None:
        if not 0 < value <= 10:
            raise ValueError(
                "Invalid saturation coefficient. It should be a positive"
                " number, ideally between 0 and 1 and close to 1."
            )

@tag("cli_argument/common/regularization_coefficient", "affect_weights")
class RegularizationCoefficient(CLIArgument[float]):
    def short_name(self) -> str:
        return "-R"

    def long_name(self) -> str:
        return "--regularization-coefficient"

    def help(self) -> str:
        return (
            "Coefficient used for weighing the square L^2 norm of patch means"
            " in the guided filter refinement of the coarse transmission map."
        )

    def default(self) -> float:
        return 0.01

    def _validate_value(self, value: float) -> None: pass
