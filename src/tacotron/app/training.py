import logging
import os
import shutil
from functools import partial
from logging import Logger
from typing import Dict, Optional

from tacotron.app.io import (get_checkpoints_dir,
                             get_train_checkpoints_log_file, get_train_dir,
                             get_train_log_file, get_train_logs_dir,
                             load_prep_settings, save_prep_settings)
from tacotron.core import CheckpointTacotron, Tacotron2Logger
from tacotron.core import continue_train as continue_train_core
from tacotron.core import train as train_core
from tacotron.utils import (get_custom_or_last_checkpoint, get_last_checkpoint,
                            get_pytorch_filename, prepare_logger)
from tts_preparation import (get_merged_dir, get_prep_dir,
                             load_merged_accents_ids,
                             load_merged_speakers_json,
                             load_merged_symbol_converter, load_trainset,
                             load_valset, load_weights_map)


def try_load_checkpoint(base_dir: str, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointTacotron]:
  result = None
  if train_name:
    train_dir = get_train_dir(base_dir, train_name, False)
    checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(train_dir), checkpoint)
    result = CheckpointTacotron.load(checkpoint_path, logger)
    logger.info(f"Using warm start model: {checkpoint_path}")
  return result


def save_checkpoint(checkpoint: CheckpointTacotron, save_checkpoint_dir: str, logger: Logger) -> None:
  checkpoint_path = os.path.join(
    save_checkpoint_dir, get_pytorch_filename(checkpoint.iteration))
  checkpoint.save(checkpoint_path, logger)


def restore_model(base_dir: str, train_name: str, checkpoint_dir: str) -> None:
  train_dir = get_train_dir(base_dir, train_name, create=True)
  logs_dir = get_train_logs_dir(train_dir)
  logger = prepare_logger(get_train_log_file(logs_dir), reset=True)
  save_checkpoint_dir = get_checkpoints_dir(train_dir)
  last_checkpoint, iteration = get_last_checkpoint(checkpoint_dir)
  logger.info(f"Restoring checkpoint {iteration} from {checkpoint_dir}...")
  shutil.copy2(last_checkpoint, save_checkpoint_dir)
  logger.info("Restoring done.")


def train(base_dir: str, ttsp_dir: str, train_name: str, merge_name: str, prep_name: str, warm_start_train_name: Optional[str] = None, warm_start_checkpoint: Optional[int] = None, custom_hparams: Optional[Dict[str, str]] = None, weights_train_name: Optional[str] = None, weights_checkpoint: Optional[int] = None, use_weights_map: Optional[bool] = None, map_from_speaker: Optional[str] = None) -> None:
  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)

  train_dir = get_train_dir(base_dir, train_name, create=True)
  logs_dir = get_train_logs_dir(train_dir)

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

  weights_model = try_load_checkpoint(
    base_dir=base_dir,
    train_name=weights_train_name,
    checkpoint=weights_checkpoint,
    logger=logger
  )

  weights_map = None
  if use_weights_map is not None and use_weights_map:
    weights_train_dir = get_train_dir(base_dir, weights_train_name, False)
    _, weights_merge_name, _ = load_prep_settings(weights_train_dir)
    weights_map = load_weights_map(merge_dir, weights_merge_name)

  warm_model = try_load_checkpoint(
    base_dir=base_dir,
    train_name=warm_start_train_name,
    checkpoint=warm_start_checkpoint,
    logger=logger
  )

  save_callback = partial(
    save_checkpoint,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    logger=logger,
  )

  train_core(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    symbols=load_merged_symbol_converter(merge_dir),
    speakers=load_merged_speakers_json(merge_dir),
    accents=load_merged_accents_ids(merge_dir),
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    weights_map=weights_map,
    weights_checkpoint=weights_model,
    warm_model=warm_model,
    map_from_speaker_name=map_from_speaker,
    logger=logger,
    checkpoint_logger=checkpoint_logger,
  )


def continue_train(base_dir: str, train_name: str, custom_hparams: Optional[Dict[str, str]] = None) -> None:
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logs_dir = get_train_logs_dir(train_dir)
  taco_logger = Tacotron2Logger(logs_dir)
  logger = prepare_logger(get_train_log_file(logs_dir))
  checkpoint_logger = prepare_logger(
    log_file_path=get_train_checkpoints_log_file(logs_dir),
    logger=logging.getLogger("checkpoint-logger")
  )

  checkpoints_dir = get_checkpoints_dir(train_dir)
  last_checkpoint_path, _ = get_last_checkpoint(checkpoints_dir)
  last_checkpoint = CheckpointTacotron.load(last_checkpoint_path, logger)

  save_callback = partial(
    save_checkpoint,
    save_checkpoint_dir=checkpoints_dir,
    logger=logger,
  )

  ttsp_dir, merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  trainset = load_trainset(prep_dir)
  valset = load_valset(prep_dir)

  continue_train_core(
    checkpoint=last_checkpoint,
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    logger=logger,
    checkpoint_logger=checkpoint_logger,
    save_callback=save_callback
  )
