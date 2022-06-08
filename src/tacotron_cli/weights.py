from argparse import ArgumentParser, Namespace
from logging import getLogger

import torch

from tacotron.checkpoint_handling import (get_symbol_embedding_weights, get_symbol_mapping,
                                          update_symbol_embedding_weights, update_symbol_mapping)
from tacotron.utils import set_torch_thread_to_max
from tacotron_cli.argparse_helper import parse_existing_file
from tacotron_cli.helper import add_device_argument
from tacotron_cli.io import save_checkpoint, try_load_checkpoint


def init_add_missing_weights_parser(parser: ArgumentParser) -> None:
  parser.description = "Copy missing symbols from one checkpoint to another."
  parser.add_argument('checkpoint1', metavar="CHECKPOINT1", type=parse_existing_file,
                      help="path to checkpoint from which the symbols should be copied")
  parser.add_argument('checkpoint2', metavar="CHECKPOINT2", type=parse_existing_file,
                      help="path to checkpoint to which the symbols should be copied")
  parser.add_argument('--mode', type=str,
                      choices=["copy", "predict"], default="copy", help="mode how the weights of the symbols should be transferred: copy => 1:1 copy of weights; predict => predict weights mathematically using a difference vector")
  add_device_argument(parser)
  return map_missing_symbols_ns


def map_missing_symbols_ns(ns: Namespace) -> bool:
  logger = getLogger(__name__)
  set_torch_thread_to_max()

  checkpoint1_dict = try_load_checkpoint(ns.checkpoint1, ns.device, logger)
  if checkpoint1_dict is None:
    return False

  checkpoint2_dict = try_load_checkpoint(ns.checkpoint2, ns.device, logger)
  if checkpoint2_dict is None:
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

  save_checkpoint(checkpoint2_dict, ns.checkpoint2)
  return True
