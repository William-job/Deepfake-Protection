import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter


def setup_logger(log_dir, log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("CAR")
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "training.log"), encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    logger.info(f"Logger initialized. Log directory: {log_dir}")
    return logger, writer


def log_metrics(writer, metrics, step, prefix="train"):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def close_logger(writer):
    writer.close()