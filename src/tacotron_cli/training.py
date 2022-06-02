import logging
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, Optional

from speech_dataset_parser_api import parse_directory

from tacotron.checkpoint_handling import CheckpointDict, get_iteration
from tacotron.logger import Tacotron2Logger
from tacotron.parser import get_entries_from_sdp_entries
from tacotron.training import start_training
from tacotron.utils import (get_last_checkpoint, get_pytorch_filename, parse_json, prepare_logger,
                            split_hparams_string)
from tacotron_cli.io import load_checkpoint, save_checkpoint

# def try_load_checkpoint(train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointDict]:
#     result = None
#     if train_name:
#         train_dir = get_train_dir(base_dir, train_name)
#         checkpoint_path, _ = get_custom_or_last_checkpoint(
#             get_checkpoints_dir(train_dir), checkpoint)
#         result = load_checkpoint(checkpoint_path)
#         logger.info(f"Using warm start model: {checkpoint_path}")
#     return result


def save_checkpoint_iteration(checkpoint: CheckpointDict, save_checkpoint_dir: Path) -> None:
  iteration = get_iteration(checkpoint)
  # TODO
  #checkpoint_path = save_checkpoint_dir / f"{iteration}.pkl"
  checkpoint_path = save_checkpoint_dir / get_pytorch_filename(iteration)
  save_checkpoint(checkpoint, checkpoint_path)


# def restore_model(base_dir: Path, train_name: str, checkpoint_dir: Path) -> None:
#   train_dir = get_train_dir(base_dir, train_name, create=True)
#   logs_dir = get_train_logs_dir(train_dir)
#   logger = prepare_logger(get_train_log_file(logs_dir), reset=True)
#   save_checkpoint_dir = get_checkpoints_dir(train_dir)
#   last_checkpoint, iteration = get_last_checkpoint(checkpoint_dir)
#   logger.info(f"Restoring checkpoint {iteration} from {checkpoint_dir}...")
#   shutil.copy2(last_checkpoint, save_checkpoint_dir)
#   logger.info("Restoring done.")


def init_train_parser(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "tacotron_logs"
  parser.add_argument('train_folder', metavar="TRAIN-FOLDER-PATH", type=Path)
  parser.add_argument('val_folder', metavar="VAL-FOLDER-PATH", type=Path)
  parser.add_argument("tier", metavar="TIER", type=str)
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER-PATH", type=Path)
  parser.add_argument('--custom-hparams', type=str, default=None)
  # Pretrained model
  parser.add_argument('--pretrained-model', type=Path, default=None)
  # Warm start
  parser.add_argument('--warm-start', action='store_true')
  # Symbol weights
  parser.add_argument('--map-symbol_weights', action='store_true')
  parser.add_argument('--custom-symbol-weights-map', type=Path, default=None)
  # Speaker weights
  parser.add_argument('--map-speaker-weights', action='store_true')
  parser.add_argument('--map-from-speaker', type=str, default=None)
  parser.add_argument('--tl-dir', type=Path, default=default_log_path)
  parser.add_argument('--log-path', type=Path,
                      default=default_log_path / "log.txt")
  parser.add_argument('--ckp-log-path', type=Path,
                      default=default_log_path / "log-checkpoints.txt")
  return train_new


def train_new(ns: Namespace) -> None:
  taco_logger = Tacotron2Logger(ns.tl_dir)
  logger = prepare_logger(ns.log_path, reset=True)
  checkpoint_logger = prepare_logger(
    log_file_path=ns.ckp_log_path,
    logger=logging.getLogger("checkpoint-logger"),
    reset=True
  )

  pretrained_model_checkpoint = None
  weights_map = None
  if ns.pretrained_model is not None:
    pretrained_model_checkpoint = load_checkpoint(ns.pretrained_model)

    if ns.custom_symbol_weights_map is not None:
      if not ns.custom_symbol_weights_map.is_file():
        logger.error("Weights map does not exist!")
        return
      weights_map = parse_json(ns.custom_symbol_weights_map)

  save_callback = partial(
    save_checkpoint_iteration,
    save_checkpoint_dir=ns.checkpoints_dir,
  )

  trainset = list(get_entries_from_sdp_entries(
      parse_directory(ns.train_folder, ns.tier, 16)))
  valset = list(get_entries_from_sdp_entries(
      parse_directory(ns.val_folder, ns.tier, 16)))

  custom_hparams = split_hparams_string(ns.custom_hparams)

  logger.info("Starting new model...")
  start_training(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    custom_symbol_weights_map=weights_map,
    pretrained_model=pretrained_model_checkpoint,
    warm_start=ns.warm_start,
    map_symbol_weights=ns.map_symbol_weights,
    map_speaker_weights=ns.map_speaker_weights,
    map_from_speaker_name=ns.map_from_speaker,
    logger=logger,
    checkpoint_logger=checkpoint_logger,
    checkpoint=None,
  )


def init_continue_train_parser(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "tacotron_logs"
  parser.add_argument('train_folder', metavar="TRAIN-FOLDER-PATH", type=Path)
  parser.add_argument('val_folder', metavar="VAL-FOLDER-PATH", type=Path)
  parser.add_argument("tier", metavar="TIER", type=str)
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER-PATH", type=Path)
  parser.add_argument('--custom-hparams', type=str, default=None)
  parser.add_argument('--tl-dir', type=Path, default=default_log_path)
  parser.add_argument('--log-path', type=Path,
                      default=default_log_path / "log.txt")
  parser.add_argument('--ckp-log-path', type=Path,
                      default=default_log_path / "log-checkpoints.txt")
  return continue_train_v2


def continue_train_v2(ns: Namespace) -> None:
  taco_logger = Tacotron2Logger(ns.tl_dir)
  logger = prepare_logger(ns.log_path, reset=False)
  checkpoint_logger = prepare_logger(
    log_file_path=ns.ckp_log_path,
    logger=logging.getLogger("checkpoint-logger"),
    reset=False
  )

  save_callback = partial(
      save_checkpoint_iteration,
      save_checkpoint_dir=ns.checkpoints_dir,
  )

  trainset = list(get_entries_from_sdp_entries(
    parse_directory(ns.train_folder, ns.tier, 16)))
  valset = list(get_entries_from_sdp_entries(
    parse_directory(ns.val_folder, ns.tier, 16)))

  last_checkpoint_path, _ = get_last_checkpoint(ns.checkpoints_dir)
  last_checkpoint = load_checkpoint(last_checkpoint_path)
  custom_hparams = split_hparams_string(ns.custom_hparams)

  logger.info("Continuing training from checkpoint...")
  start_training(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    custom_symbol_weights_map=None,
    pretrained_model=None,
    map_from_speaker_name=None,
    map_symbol_weights=False,
    checkpoint=last_checkpoint,
    logger=logger,
    checkpoint_logger=checkpoint_logger,
    warm_start=False,
    map_speaker_weights=False,
  )
