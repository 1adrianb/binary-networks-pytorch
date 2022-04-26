import os
import atexit
import functools
import logging
import subprocess
import sys

from iopath.common.file_io import g_pathmgr
from nash_logging.io_utils import makedir


class Logger:
    INFO_MSG = 1
    DEBUG_MSG = 2

    def __init__(self, name, output_dir=None):
        self.setup_logging(name, output_dir)

    def setup_logging(self, name, output_dir=None):
        """
        Setup various logging streams: stdout and file handlers.
        For file handlers, we only setup for the master gpu.
        """
        # get the filename if we want to log to the file as well
        log_filename = None
        if output_dir:
            makedir(output_dir)
            log_filename = f"{output_dir}/log.txt"

        self.output_dir = output_dir
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # create formatter
        FORMAT = (
            "%(asctime)s | %(message)s"
        )
        formatter = logging.Formatter(FORMAT, datefmt="%m/%d %I:%M:%S %p")

        # clean up any pre-existing handlers
        for h in logger.handlers:
            logger.removeHandler(h)
        logger.root.handlers = []

        # setup the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # we log to file as well if user wants
        if log_filename:
            file_handler = logging.StreamHandler(
                self._cached_log_stream(log_filename)
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logging.root = logger

    def log(self, message, type=None):
        """
        Log a `message` using `type` which might be info or debug
        """
        if type == Logger.INFO_MSG or type is None:
            logging.info(message)
        elif type == Logger.DEBUG_MSG:
            logging.debug(message)

    # cache the opened file object, so that different calls to `setup_logger`
    # with the same file name can safely write to the same file.
    @functools.lru_cache(maxsize=None)
    def _cached_log_stream(self, filename):
        # we tune the buffering value so that the logs are updated
        # frequently.
        log_buffer_kb = 10 * 1024  # 10KB
        io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
        atexit.register(io.close)
        return io

    def shutdown_logging(self):
        """
        After training is done, we ensure to shut down all the logger streams.
        """
        logging.info("Shutting down loggers...")
        handlers = logging.root.handlers
        for handler in handlers:
            handler.close()

    def log_gpu_stats(self):
        """
        Log nvidia-smi snapshot. Useful to capture the configuration of gpus.
        """
        try:
            logging.info(
                subprocess.check_output(["nvidia-smi"]).decode("utf-8")
            )
        except FileNotFoundError:
            logging.error(
                "Failed to find the 'nvidia-smi' executable for printing GPU stats"
            )
        except subprocess.CalledProcessError as e:
            logging.error(
                f"nvidia-smi returned non zero error code: {e.returncode}"
            )

    def print_gpu_memory_usage(self):
        """
        Parse the nvidia-smi output and extract the memory used stats.
        Not recommended to use.
        """
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        all_values, count, out_dict = [], 0, {}
        for item in out_list:
            if " MiB" in item:
                out_dict[f"GPU {count}"] = item.strip()
                all_values.append(int(item.split(" ")[0]))
                count += 1
        logging.info(
            f"Memory usage stats:\n"
            f"Per GPU mem used: {out_dict}\n"
            f"nMax memory used: {max(all_values)}"
        )

    def save_custom_txt_output(self, content="", name="name.txt", subdir=None):
        print(content, name, subdir)
        if subdir is not None:
            out_folder = os.path.join(self.output_dir, subdir)
            out_file = os.path.join(out_folder, name)
            os.makedirs(out_folder, exist_ok=True)
        else:
            out_file = os.path.join(self.output_dir, name)

        with open(out_file, "w") as f:
            f.write(content)
