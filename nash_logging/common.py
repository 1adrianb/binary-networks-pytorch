from nash_logging.logger import Logger
from nash_logging.config import print_cfg, save_cfg
from nash_logging.checkpoint import save_source_files
from nash_logging.checkpoint import get_checkpoint_folder
from nash_logging.tensorboard_utils import get_tensorboard_logger

online_loggers = {"tensorboard": get_tensorboard_logger}


class LoggerUnited:
    INFO_MSG = 1
    DEBUG_MSG = 2

    def __init__(self, config, online_logger=None):
        if online_logger is not None:
            self.online_logger = online_loggers[online_logger](config)
        self.use_online = False if online_logger is None else True
        self.system_logger = self.setup_loggers(config)

    def setup_loggers(self, cfg):
        # setup logging
        logger = Logger(__name__, output_dir=get_checkpoint_folder(cfg))
        logger.log_gpu_stats()
        logger.print_gpu_memory_usage()

        print_cfg(cfg)
        save_cfg(cfg)
        save_source_files(cfg)

        return logger

    def log(self, message, type=None):
        self.system_logger.log(message, type=None)

    def shutdown_logging(self):
        self.system_logger.shutdown_logging()

    def log_gpu_stats(self):
        self.system_logger.log_gpu_stats()

    def print_gpu_memory_usage(self):
        self.system_logger.print_gpu_memory_usage()

    def save_custom_txt(self, content="", name="name.txt", subdir=None):
        self.system_logger.save_custom_txt_output(content, name, subdir)

    def on_update(
        self,
        iteration,
        loss,
        log_frequency,
        batch_time,
        max_iteration,
    ):

        if self.use_online:
            self.online_logger.on_update(
                iteration=iteration,
                loss=loss,
                log_frequency=log_frequency,
                batch_time=batch_time,
                max_iteration=max_iteration,
            )

    def log_metrics(self, tab="Train", metrics={}, phase_idx=0):

        if self.use_online:
            self.online_logger.log_metrics(tab, metrics, phase_idx)

    def add_histogramm(self, values=None, phase_idx=0, name="histogram"):
        if self.use_online:
            self.online_logger.add_histogramm(
                values=values, phase_idx=phase_idx, name=name
            )

    def add_embedding(self, emb, tag, phase_idx):
        if self.use_online:
            self.online_logger.add_embedding(emb, tag, phase_idx)

    def add_graph(self, model, input):
        if self.use_online:
            self.online_logger.add_graph(model, input)

    def add_images(self, tag, image):
        if self.use_online:
            self.online_logger.add_images(tag, image)