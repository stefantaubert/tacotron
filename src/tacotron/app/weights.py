from logging import getLogger
from pathlib import Path
from typing import cast

from ordered_set import OrderedSet
from tacotron.app.io import (get_checkpoints_dir, get_train_dir,
                             load_prep_settings)
from tacotron.core import CheckpointTacotron
from tacotron.core.model_weights import map_symbols
from tacotron.utils import get_checkpoint, get_last_checkpoint
from tts_preparation import get_merged_dir
from tts_preparation.app.io import save_merged_symbol_converter


def map_missing_symbols(base_dir: Path, from_train_name: str, to_train_name: str) -> None:
  logger = getLogger(__name__)

  from_train_dir = get_train_dir(base_dir, from_train_name)
  assert from_train_dir.is_dir()

  to_train_dir = get_train_dir(base_dir, to_train_name)
  assert to_train_dir.is_dir()

  from_checkpoint_dir = get_checkpoints_dir(from_train_dir)
  _, from_last_it = get_last_checkpoint(from_checkpoint_dir)
  from_checkpoint_path = get_checkpoint(from_checkpoint_dir, from_last_it)
  from_taco_checkpoint = cast(
    CheckpointTacotron, CheckpointTacotron.load(from_checkpoint_path, logger))

  to_checkpoint_dir = get_checkpoints_dir(to_train_dir)
  _, to_last_it = get_last_checkpoint(to_checkpoint_dir)
  to_checkpoint_path = get_checkpoint(to_checkpoint_dir, to_last_it)
  to_taco_checkpoint = cast(CheckpointTacotron, CheckpointTacotron.load(to_checkpoint_path, logger))

  missing_symbols = from_taco_checkpoint.get_symbols().get_all_symbols() - \
      to_taco_checkpoint.get_symbols().get_all_symbols()

  logger.info("Missing: " + " ".join(missing_symbols))

  symbols = OrderedSet(sorted(missing_symbols))
  map_symbols(from_taco_checkpoint, to_taco_checkpoint, symbols=symbols)

  ttsp_dir, merge_name, _ = load_prep_settings(to_train_dir)
  to_merge_dir = get_merged_dir(ttsp_dir, merge_name)
  save_merged_symbol_converter(to_merge_dir, to_taco_checkpoint.get_symbols())

  to_taco_checkpoint.save(to_checkpoint_path, logger)
  logger.info("Done.")
