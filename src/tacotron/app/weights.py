from logging import getLogger
from pathlib import Path
from typing import Literal, Optional, cast
import numpy as np

from ordered_set import OrderedSet
import torch
from tacotron.app.io import (get_checkpoints_dir, get_train_dir, load_checkpoint,
                             load_prep_settings, save_checkpoint)
from tacotron.core.checkpoint_handling import get_symbol_embedding_weights, get_symbol_mapping, update_symbol_embedding_weights, update_symbol_mapping
from tacotron.core.model_weights import AddSymbolEmbeddings, map_symbols
from tacotron.utils import get_checkpoint, get_last_checkpoint, try_copy_to_gpu
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


def map_missing_symbols_v2(base_dir: Path, checkpoint1: Path, checkpoint2: Path, mode: Literal["copy", "predict"], custom_output: Optional[Path]) -> bool:
  assert mode in ["copy", "predict"]

  logger = getLogger(__name__)

  if not checkpoint1.is_file():
    logger.error("Checkpoint 1 was not found!")
    return False

  if not checkpoint2.is_file():
    logger.error("Checkpoint 2 was not found!")
    return False

  try:
    logger.debug(f"Loading checkpoint 1...")
    checkpoint1_dict = load_checkpoint(checkpoint1)
  except Exception as ex:
    logger.error("Checkpoint 1 couldn't be loaded!")
    return False

  try:
    logger.debug(f"Loading checkpoint 2...")
    checkpoint2_dict = load_checkpoint(checkpoint2)
  except Exception as ex:
    logger.error("Checkpoint 2 couldn't be loaded!")
    return False

  symbol_mapping1 = get_symbol_mapping(checkpoint1_dict)
  symbol_mapping2 = get_symbol_mapping(checkpoint2_dict)
  symbol_mapping1["PADDING"] = 0
  symbol_mapping2["PADDING"] = 0
  symbol_emb1 = get_symbol_embedding_weights(checkpoint1_dict)
  symbol_emb2 = get_symbol_embedding_weights(checkpoint2_dict)
  if symbol_emb1.shape[1] != symbol_emb2.shape[1]:
    logger.error("Both models need to have the same symbol embedding dimensions!")
    return False

  if mode == "predict":
    difference_vectors = []

    for symbol1, index1 in symbol_mapping1.items():
      if symbol1 in symbol_mapping2:
        index2 = symbol_mapping2[symbol1]
        vec1 = symbol_emb1[index1]
        vec2 = symbol_emb2[index2]
        difference_vector = vec2 - vec1
        difference_vectors.append(difference_vector)
    difference_vectors_torch = torch.stack(difference_vectors)
    average_difference_vector = torch.mean(difference_vectors_torch, 0)

  target_embedding = symbol_emb2
  target_symbol_mapping = symbol_mapping2
  mapped_symbols = []
  for symbol1, index1 in symbol_mapping1.items():
    if symbol1 not in symbol_mapping2:
      vec1 = symbol_emb1[index1]
      if mode == "predict":
        vec2 = vec1 + average_difference_vector
      else:
        assert mode == "copy"
        vec2 = vec1
      vec2 = torch.reshape(vec2, (1, symbol_emb2.shape[1]))
      target_embedding = torch.cat((target_embedding, vec2))
      target_symbol_mapping[symbol1] = target_embedding.shape[0] - 1
      mapped_symbols.append(symbol1)

  if len(mapped_symbols) == 0:
    logger.info("No symbols are missing. Didn't changed anything.")
    return True

  logger.info(f"Added symbols: {' '.join(mapped_symbols)} (#{len(mapped_symbols)})")

  target_symbol_mapping.pop("PADDING")
  update_symbol_mapping(checkpoint2_dict, target_symbol_mapping)
  update_symbol_embedding_weights(checkpoint2_dict, target_embedding)

  target_path = checkpoint2
  if custom_output is not None:
    target_path = custom_output
  save_checkpoint(checkpoint2_dict, target_path)
  logger.info("Success!")
