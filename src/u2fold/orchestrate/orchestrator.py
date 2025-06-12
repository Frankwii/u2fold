from abc import ABC, abstractmethod

from u2fold.config_parsing.config_dataclasses import (
    ExecConfig,
    TrainConfig,
    U2FoldConfig,
)


class Orchestrator(ABC):
    def __init__(self, config: U2FoldConfig) -> None: ...

    @abstractmethod
    def run(self) -> None: ...

class TrainOrchestrator(Orchestrator):
    def __init__(self, config: TrainConfig) -> None: ...

    def run(self) -> None: ...

class ExecOrchestrator(Orchestrator):
    def __init__(self, config: ExecConfig) -> None: ...

    def run(self) -> None: ...
