from general_utils import parse_json
import logging
import os
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Dict, Optional

from tacotron.app.io import (get_checkpoints_dir,
                             get_train_checkpoints_log_file, get_train_dir,
                             get_train_log_file, get_train_logs_dir, load_checkpoint,
                             load_prep_settings, save_checkpoint, save_prep_settings)
from tacotron.core import Tacotron2Logger
from tacotron.core import continue_train as continue_train_core
from tacotron.core import train as train_core
from tacotron.core.checkpoint_handling import CheckpointDict, get_iteration
from tacotron.core.training import start_training
from tacotron.utils import (get_custom_or_last_checkpoint, get_last_checkpoint,
                            get_pytorch_filename, prepare_logger)
from tts_preparation import (get_merged_dir, get_prep_dir,
                             load_merged_speakers_json, load_trainset,
                             load_valset, load_weights_map)
from tts_preparation.app.io import load_merged_symbol_converter


def try_load_checkpoint(base_dir: Path, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointDict]:
  result = None
  if train_name:
    train_dir = get_train_dir(base_dir, train_name)
    checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(train_dir), checkpoint)
    result = load_checkpoint(checkpoint_path)
    logger.info(f"Using warm start model: {checkpoint_path}")
  return result


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


def train(base_dir: Path, ttsp_dir: Path, train_name: str, merge_name: str, prep_name: str, custom_hparams: Optional[Dict[str, str]], pretrained_model: Path, warm_start: bool, map_symbol_weights: bool, custom_symbol_weights_map: Optional[Path], map_speaker_weights: bool, map_from_speaker: Optional[str]) -> None:
  # Parameter: custom_symbol_weights_map -> a JSON file that contains keys equals to symbols from the weights model checkpoint and values equals to the symbols which are in the model to be trained.

  merge_dir = get_merged_dir(ttsp_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  train_dir = get_train_dir(base_dir, train_name)
  logs_dir = get_train_logs_dir(train_dir)
  logs_dir.mkdir(parents=True, exist_ok=True)

  taco_logger = Tacotron2Logger(logs_dir)
  logger = prepare_logger(get_train_log_file(logs_dir), reset=True)
  checkpoint_logger = prepare_logger(
    log_file_path=get_train_checkpoints_log_file(logs_dir),
    logger=logging.getLogger("checkpoint-logger"),
    reset=True
  )

  save_prep_settings(train_dir, ttsp_dir, merge_name, prep_name)

  trainset = load_trainset(prep_dir)
  valset = load_valset(prep_dir)

  pretrained_model_checkpoint = None
  if pretrained_model is not None:
    pretrained_model_checkpoint = load_checkpoint(pretrained_model)

    if custom_symbol_weights_map is not None:
      if not custom_symbol_weights_map.is_file():
        logger.error("Weights map does not exist!")
        return
      weights_map = parse_json(custom_symbol_weights_map)
      weights_map_contains_duplicate_values = len(
        weights_map.values()) > len(set(weights_map.values()))
      if weights_map_contains_duplicate_values:
        logger.error("Invalid weights map: Mapped to the same symbol multiple times!")
        return

  save_callback = partial(
    save_checkpoint_iteration,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
  )

  logger.info("Starting new model...")
  start_training(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    custom_symbol_weights_map=weights_map,
    pretrained_model=pretrained_model_checkpoint,
    warm_start=warm_start,
    map_symbol_weights=map_symbol_weights,
    map_speaker_weights=map_speaker_weights,
    map_from_speaker_name=map_from_speaker,
    logger=logger,
    checkpoint_logger=checkpoint_logger,
    checkpoint=None,
  )


def continue_train(base_dir: Path, train_name: str, custom_hparams: Optional[Dict[str, str]] = None) -> None:
  train_dir = get_train_dir(base_dir, train_name)
  assert train_dir.is_dir()

  logs_dir = get_train_logs_dir(train_dir)
  taco_logger = Tacotron2Logger(logs_dir)
  logger = prepare_logger(get_train_log_file(logs_dir))
  checkpoint_logger = prepare_logger(
    log_file_path=get_train_checkpoints_log_file(logs_dir),
    logger=logging.getLogger("checkpoint-logger")
  )

  checkpoints_dir = get_checkpoints_dir(train_dir)
  last_checkpoint_path, _ = get_last_checkpoint(checkpoints_dir)
  last_checkpoint = load_checkpoint(last_checkpoint_path)

  save_callback = partial(
    save_checkpoint_iteration,
    save_checkpoint_dir=checkpoints_dir,
  )

  ttsp_dir, merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(ttsp_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)
  trainset = load_trainset(prep_dir)
  valset = load_valset(prep_dir)

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
