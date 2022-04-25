import os
import copy
import argparse
from omegaconf import OmegaConf as omg

from trainers.trainer import Trainer
from nash_logging.common import LoggerUnited
from trainers.metrics import get_metrics_and_loss



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--batch_size', metavar='batch_size', default=32,
                    help='batch size')

parser.add_argument('--exp_name', metavar='exp_name', default='test',
                    help='experiment name')


parser.add_argument('--gpu', metavar='gpu', default=None,
                    help='gpu')


parser.add_argument('--epochs', metavar='epochs', default=100,
                    help='gpu')

args = vars(parser.parse_args())


def run_experiment(model,get_loaders, target, ):

    env_config = omg.load('./configs/env.yaml')
    
    if args['gpu'] is not None:
        env_config.HARDWARE.GPU = int(args['gpu'])

    new_conf = copy.deepcopy(env_config)
    new_conf.EXPERIMENT.DIR = os.path.join(
        env_config.EXPERIMENT.DIR, args['exp_name']
    )

    logger = LoggerUnited(new_conf, online_logger="tensorboard")
    
    train_loader, test_loader = get_loaders(workers=new_conf.HARDWARE.WORKERS,
                                            batch_size=int(args['batch_size']))

    dataloaders = {"train": train_loader, "validation": test_loader}

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
        num_epochs=int(args['epochs']),
        device=env_config.HARDWARE.GPU,
        logger=logger,
        log_training=True,
    )

    trainer.set_model(model, {"main": model.parameters()})
    trainer.set_dataloaders(dataloaders)
    trainer.train()
    history = trainer.get_history()

    for k in history:
        for i, value in enumerate(history[k]):
            logger.log_metrics("Best val scores", {k: value}, i)
        print(k, max(history[k]))
