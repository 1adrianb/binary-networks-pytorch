# TensorboardLogger:

This class lives in `tensorboard_utils.py`, might be instantiated with `get_tensorboard_logger(cfg)`.

It provides following:
* `::method::on_phase_end` call it at the end of every epoch
* `::method::on_update` call it after every parameters update (i.e. after optimization step)