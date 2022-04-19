import pprint

import logging
from omegaconf import DictConfig, OmegaConf

from nash_logging.checkpoint import get_checkpoint_folder


def print_cfg(cfg):
    """
    Supports printing both DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    if isinstance(cfg, DictConfig):
        if hasattr(cfg, "pretty"):
            # Backward compatibility
            logging.info(cfg.pretty())
        else:
            # Newest version of OmegaConf
            logging.info(OmegaConf.to_yaml(cfg))
    else:
        logging.info(pprint.pformat(cfg))


def save_cfg(cfg):
    checkpoint_folder = get_checkpoint_folder(cfg)
    logging.info(f"Saving config to {checkpoint_folder}/config.yaml:")
    if isinstance(cfg, DictConfig):
        OmegaConf.save(config=cfg, f=f"{checkpoint_folder}/config.yaml")
    else:
        logging.error(f"Config is not of DictConfig type")