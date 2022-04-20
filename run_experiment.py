import os
import copy
from omegaconf import OmegaConf as omg

from trainers.trainer import Trainer
from nash_logging.common import LoggerUnited
from trainers.metrics import get_metrics_and_loss


def run_experiment(model, dataloaders, experiment_name, target):

    env_config = omg.load(
        "/home/dev/work_main/2022/NASH/NASLib/configs/env.yaml"
    )
    new_conf = copy.deepcopy(env_config)
    new_conf.EXPERIMENT.DIR = os.path.join(
        env_config.EXPERIMENT.DIR, experiment_name
    )
    logger = LoggerUnited(env_config, online_logger="tensorboard")

    optimizer = {
        "main": (
            "ADAM",
            {
                "lr": 0.001,
                "weight_decay": 0,
            },
        )
    }
    TARGET = target
    criterion, metrics = get_metrics_and_loss(
        "CrossEntropyLoss", ["accuracy", "f1_macro"], TARGET
    )

    trainer = Trainer(
        criterion,
        metrics,
        optimizers=optimizer,
        phases=["train", "validation"],
        num_epochs=10,
        device=1,
        logger=logger,
    )

    trainer.set_model(model, {"main": model.parameters()})
    trainer.set_dataloaders(dataloaders)
    trainer.train()
    history = trainer.get_history()

    for k in history:
        for i, value in enumerate(history[k]):
            logger.log_metrics("Best val scores", {k: value}, i)
        print(k, max(history[k]))
