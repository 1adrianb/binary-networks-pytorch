from omegaconf import OmegaConf
import torch
from tensorboard_utils import get_tensorboard_logger

from common import LoggerUnited


def test_logger():
    # make some config
    cfg = OmegaConf.create(
        {
            "PROJECT": {
                "ROOT": "/home/dev/work_main/2022/NASH/NASLib/",
                # this parameter should be in __BASE__ config
                "DEFAULT_DIRS": [
                    "nash_logging",
                    "search_optimizers",
                    "search_spaces",
                    "trainers",
                ],
            },
            "EXPERIMENT": {"DIR": "../experiments/exp1"},
            "TENSORBOARD_SETUP": {
                "FLUSH_EVERY_N_MIN": 1,
                "LOG_PARAMS": True,
                "LOG_PARAMS_EVERY_N_ITERS": 1,
            },
        }
    )

    # setup logging
    logger = LoggerUnited(cfg, online_logger="tensorboard")
    # work with TensorFlow

    log_frequency = 5
    max_iteration = 500
    num_epochs = 100
    num_batches = 15
    it = 0
    for epoch in range(num_epochs):
        for batch_id in range(num_batches):
            batch_time = [
                1,
                3,
                2,
            ]  # seconds model processed the batch
            loss = torch.tensor(100 / (it + 1))

            # optimizer.step() ...

            logger.on_update(
                iteration=it,
                loss=loss,
                log_frequency=log_frequency,
                batch_time=batch_time,
                max_iteration=max_iteration,
            )

            it += 1
            if it == max_iteration:
                break
        # compute metrics here ...
        logger.log_metrics(
            tab="Train",
            metrics={
                "accuracy": torch.tensor(0.99).item(),
                "f1": torch.tensor(0.8).item(),
            },
            phase_idx=it,
        )

        if it == max_iteration:
            break

    logger.log("All Done!", logger.INFO_MSG)

    # close the logging streams including the filehandlers
    logger.shutdown_logging()


if __name__ == "__main__":
    test_logger()
