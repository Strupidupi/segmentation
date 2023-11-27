from main import main

from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging


class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        main()

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass