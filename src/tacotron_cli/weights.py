from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import Literal, Optional

import torch

from tacotron.checkpoint_handling import (get_symbol_embedding_weights, get_symbol_mapping,
                                          update_symbol_embedding_weights, update_symbol_mapping)
from tacotron_cli.io import load_checkpoint, save_checkpoint


def init_add_missing_weights_parser(parser: ArgumentParser) -> None:
  parser.add_argument('checkpoint1', metavar="CHECKPOINT1-PATH", type=Path)
  parser.add_argument('checkpoint2', metavar="CHECKPOINT2-PATH", type=Path)
  parser.add_argument('--mode', type=str,
                      choices=["copy", "predict"], default="copy")
  parser.add_argument('-out', '--custom-output', type=Path, default=None)
  return map_missing_symbols_v2


def map_missing_symbols_v2(ns: Namespace) -> bool:
  assert ns.mode in ["copy", "predict"]

  logger = getLogger(__name__)

  if not ns.checkpoint1.is_file():
    logger.error("Checkpoint 1 was not found!")
    return False

  if not ns.checkpoint2.is_file():
    logger.error("Checkpoint 2 was not found!")
    return False

  try:
    logger.debug("Loading checkpoint 1...")
    checkpoint1_dict = load_checkpoint(ns.checkpoint1)
  except Exception as ex:
    logger.error("Checkpoint 1 couldn't be loaded!")
    return False

  try:
    logger.debug("Loading checkpoint 2...")
    checkpoint2_dict = load_checkpoint(ns.checkpoint2)
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
    logger.error(
        "Both models need to have the same symbol embedding dimensions!")
    return False

  if ns.mode == "predict":
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
      if ns.mode == "predict":
        vec2 = vec1 + average_difference_vector
      else:
        assert ns.mode == "copy"
        vec2 = vec1
      vec2 = torch.reshape(vec2, (1, symbol_emb2.shape[1]))
      target_embedding = torch.cat((target_embedding, vec2))
      target_symbol_mapping[symbol1] = target_embedding.shape[0] - 1
      mapped_symbols.append(symbol1)

  if len(mapped_symbols) == 0:
    logger.info("No symbols are missing. Didn't changed anything.")
    return True

  logger.info(
      f"Added symbols: {' '.join(mapped_symbols)} (#{len(mapped_symbols)})")

  target_symbol_mapping.pop("PADDING")
  update_symbol_mapping(checkpoint2_dict, target_symbol_mapping)
  update_symbol_embedding_weights(checkpoint2_dict, target_embedding)

  target_path = ns.checkpoint2
  if ns.custom_output is not None:
    target_path = ns.custom_output
  save_checkpoint(checkpoint2_dict, target_path)
  return True
