import os
from glob import glob
import logging
import shutil
from pathlib import Path

from iopath.common.file_io import g_pathmgr
from nash_logging.io_utils import makedir


def get_checkpoint_folder(config):
    """
    Check, create and return the checkpoint folder. User can specify their own
    checkpoint directory otherwise the default "." is used.
    """
    odir = config.EXPERIMENT.DIR

    makedir(odir)
    assert g_pathmgr.exists(
        config.EXPERIMENT.DIR
    ), f"Please specify config.CHECKPOINT.DIR parameter. Invalid: {config.CHECKPOINT.DIR}"
    return odir


def save_source_files(config):
    checkpoint_folder = get_checkpoint_folder(config)
    logging.info(f"Saving source files to {checkpoint_folder}/project_source/")
    assert g_pathmgr.exists(
        config.PROJECT.ROOT
    ), f"Please specify config.PROJECT.ROOT parameter. Invalid: {config.PROJECT.ROOT}"
    dirs_to_walk = [
        os.path.join(config.PROJECT.ROOT, x)
        for x in config.PROJECT.DEFAULT_DIRS
    ]
    py_files = glob(os.path.join(config.PROJECT.ROOT, "*.py"))
    for dw in dirs_to_walk:
        py_files.extend(
            [y for x in os.walk(dw) for y in glob(os.path.join(x[0], "*.py"))]
        )
    for pyf in py_files:
        src = pyf
        file_name = f'{pyf.replace(config.PROJECT.ROOT, "")}'
        if file_name[0] == "/":
            file_name = file_name[1:]
        dst = os.path.join(
            f"{checkpoint_folder}/project_source/",
            file_name
        )
        dst_dir = Path(dst).parents[0]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)