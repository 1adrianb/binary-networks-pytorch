import logging
import torch
from nash_logging.checkpoint import get_checkpoint_folder
from nash_logging.io_utils import makedir
from nash_logging.io_utils import is_file_empty


def is_tensorboard_available():
    """
    Check whether tensorboard is available or not.
    Returns:
        tb_available (bool): based on tensorboard imports, returns whether tensboarboard
                             is available or not.
    """
    try:
        import tensorboard  # noqa F401
        from torch.utils.tensorboard import SummaryWriter  # noqa F401

        tb_available = True
    except ImportError:
        logging.info("Tensorboard is not available")
        tb_available = False
    return tb_available


def get_tensorboard_dir(cfg):
    """
    Get the output directory where the tensorboard events will be written.
    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        tensorboard as well like log directory, logging frequency etc
    Returns:
        tensorboard_dir (str): output directory path
    """
    checkpoint_folder = get_checkpoint_folder(cfg)
    tensorboard_dir = f"{checkpoint_folder}/tb_logs"
    logging.info(f"Tensorboard dir: {tensorboard_dir}")
    makedir(tensorboard_dir)

    tensorboard_csv_dir = f"{checkpoint_folder}/tb_csv"
    logging.info(f"Tensorboard csv dir: {tensorboard_csv_dir}")
    makedir(tensorboard_csv_dir)
    return tensorboard_dir, tensorboard_csv_dir


def get_tensorboard_logger(cfg):
    """
    Construct the Tensorboard logger for visualization from the specified config
    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        tensorboard as well like log directory, logging frequency etc
    Returns:
        TensorboardHook (function): the tensorboard logger constructed
    """
    from torch.utils.tensorboard import SummaryWriter

    # get the tensorboard directory and check tensorboard is installed
    tensorboard_dir, tensorboard_csv_dir = get_tensorboard_dir(cfg)
    flush_secs = cfg.TENSORBOARD_SETUP.FLUSH_EVERY_N_MIN * 60
    return TensorboardLogger(
        tb_writer=SummaryWriter(log_dir=tensorboard_dir, flush_secs=flush_secs),
        tb_csv_dir=tensorboard_csv_dir,
        log_params=cfg.TENSORBOARD_SETUP.LOG_PARAMS,
        log_params_every_n_iterations=cfg.TENSORBOARD_SETUP.LOG_PARAMS_EVERY_N_ITERS,
    )


BYTE_TO_MiB = 2**20


class TensorboardLogger:
    """
    Tensorboard logger
    """

    def __init__(
        self,
        tb_writer,
        tb_csv_dir,
        log_params=False,
        log_params_every_n_iterations=-1,
    ):
        """The constructor method of TensorboardLogger.
        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
                        readthedocs.io/en/latest/tensorboard.html#tensorboardX.
                        SummaryWriter>`_ instance
            tb_csv_dir: directory path to store metrics as csv files
            log_params (bool): whether to log model params to tensorboard
            log_params_every_n_iterations (int): frequency at which parameters
                        should be logged to tensorboard
        """
        super().__init__()
        if not is_tensorboard_available():
            raise RuntimeError(
                "tensorboard not installed, cannot use TensorboardLogger"
            )
        logging.info("Setting up Tensorboard Logger...")
        self.tb_csv_dir = tb_csv_dir
        self.tb_writer = tb_writer
        self.log_params = log_params
        self.log_params_every_n_iterations = log_params_every_n_iterations

        logging.info(
            f"Tensorboard config: log_params: {self.log_params}, "
            f"log_params_freq: {self.log_params_every_n_iterations}, "
        )

    def log_metrics(self, tab="Train", metrics={}, phase_idx=0):
        """
        Log some arbotrary metrics.
        Also resents the CUDA memory counter.
        """
        csv_name = f"{self.tb_csv_dir}/{tab}_metrics.csv"
        csv = open(csv_name, "a")
        if is_file_empty(csv_name):
            csv.write(f"phase_idx,{','.join(metrics.keys())}\n")
        # Log train/test accuracy
        metrics_string = f"{phase_idx}"
        for metric_name, score in metrics.items():
            t = f"{tab}/{metric_name}"
            self.tb_writer.add_scalar(
                tag=t,
                scalar_value=score,
                global_step=phase_idx,
            )
            metrics_string += f",{score}"
            # Reset the GPU Memory counter
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                # torch.cuda.reset_max_memory_cached()
        metrics_string += "\n"
        csv.write(metrics_string)
        csv.close()

    def add_histogramm(self, values=None, phase_idx=0, name="histogram"):

        self.tb_writer.add_histogram(
            tag=f"Hist/{name}", values=values, global_step=phase_idx
        )

    def add_embedding(self, emb, tag, phase_idx):
        self.tb_writer.add_embedding(
            emb,
            metadata=None,
            label_img=None,
            global_step=phase_idx,
            tag=f"Emb/{tag}",
            metadata_header=None,
        )

    def add_graph(self, model, input):
        self.tb_writer.add_graph(
            model, input_to_model=input, verbose=False, use_strict_trace=True
        )

    def add_images(self, tag, images):
        self.tb_writer.add_images(tag, images)

    def on_update(
        self,
        iteration=0,
        loss=None,
        log_frequency=None,
        batch_time=[],
        max_iteration=None,
    ):
        """
        Call after every parameters update if tensorboard logger is enabled.
        Logs the scalars like training loss, learning rate, average training
        iteration time, ETA, gpu memory used, peak gpu memory used.
        """

        if (
            log_frequency is not None
            and iteration % log_frequency == 0
            or (iteration <= 100 and iteration % 5 == 0)
        ):
            # logging.info(f"Logging metrics. Iteration {iteration}")
            self.tb_writer.add_scalar(
                tag="Training/Loss",
                scalar_value=round(loss.data.cpu().item(), 5),
                global_step=iteration,
            )

            # TODO: lr saving
            # self.tb_writer.add_scalar(
            #     tag="Training/Learning_rate",
            #     scalar_value= ...,
            #     global_step=iteration,
            # )

            # Batch processing time
            if len(batch_time) > 0:
                batch_times = batch_time
            else:
                batch_times = [0]

            batch_time_avg_s = sum(batch_times) / max(len(batch_times), 1)
            self.tb_writer.add_scalar(
                tag="Speed/Batch_processing_time_ms",
                scalar_value=int(1000.0 * batch_time_avg_s),
                global_step=iteration,
            )

            # ETA
            if max_iteration is not None:
                avg_time = sum(batch_times) / len(batch_times)
                eta_secs = avg_time * (max_iteration - iteration)
                self.tb_writer.add_scalar(
                    tag="Speed/ETA_hours",
                    scalar_value=eta_secs / 3600.0,
                    global_step=iteration,
                )

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                self.tb_writer.add_scalar(
                    tag="Memory/Peak_GPU_Memory_allocated_MiB",
                    scalar_value=torch.cuda.max_memory_allocated()
                    / BYTE_TO_MiB,
                    global_step=iteration,
                )

                # Memory reserved by PyTorch's memory allocator
                self.tb_writer.add_scalar(
                    tag="Memory/Peak_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.max_memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )

                self.tb_writer.add_scalar(
                    tag="Memory/Current_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )
