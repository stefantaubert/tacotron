from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from statistics import mean, median

import pandas as pd
import plotly.offline as plt
import torch
from scipy.spatial.distance import cosine

from tacotron.analysis import (emb_plot_2d, emb_plot_3d, embeddings_to_csv, get_similarities,
                               norm2emb, sims_to_csv_v2)
from tacotron.checkpoint_handling import (get_hparams, get_iteration, get_learning_rate,
                                          get_speaker_embedding_weights, get_speaker_mapping,
                                          get_stress_mapping, get_symbol_embedding_weights,
                                          get_symbol_mapping)
from tacotron.utils import get_symbol_printable, set_torch_thread_to_max
from tacotron_cli.argparse_helper import parse_existing_file, parse_path
from tacotron_cli.helper import add_device_argument
from tacotron_cli.io import try_load_checkpoint


def init_analysis_parser(parser: ArgumentParser) -> None:
  parser.description = "Plot embedding weights in 2D/3D, calculate similarities between symbol weights and export weights as CSV."
  parser.add_argument('checkpoint', metavar="CHECKPOINT", type=parse_existing_file,
                      help="path to the checkpoint from which the weights should be analyzed")
  parser.add_argument('output_directory',
                      metavar="OUTPUT-FOLDER", type=parse_path, help="path to the folder in which the outputs should be saved")
  add_device_argument(parser)
  return analyze_ns


def analyze_ns(ns: Namespace) -> bool:
  logger = getLogger(__name__)
  set_torch_thread_to_max()

  checkpoint = try_load_checkpoint(ns.checkpoint, ns.device, logger)
  if checkpoint is None:
    return False

  ns.output_directory.mkdir(parents=True, exist_ok=True)

  hparams = get_hparams(checkpoint)

  symbol_mapping = get_symbol_mapping(checkpoint)

  logger.info(f"Iteration: {get_iteration(checkpoint)}")
  logger.info(f"Learning rate: {get_learning_rate(checkpoint)}")

  logger.info(
      f"Symbols: {' '.join(get_symbol_printable(symbol) for symbol in symbol_mapping.keys())} (#{len(symbol_mapping)}, dim: {hparams.symbols_embedding_dim})")

  if hparams.use_stress_embedding:
    stress_mapping = get_stress_mapping(checkpoint)
    logger.info(
        f"Stresses: {' '.join(stress_mapping.keys())} (#{len(stress_mapping)})")
  else:
    logger.info("Stresses: No stress embedding is contained.")

  if hparams.use_speaker_embedding:
    speaker_mapping = get_speaker_mapping(checkpoint)
    logger.info(
        f"Speakers: {', '.join(sorted(speaker_mapping.keys()))} (#{len(speaker_mapping)}, dim: {hparams.speakers_embedding_dim})")
  else:
    logger.info("Speakers: No speaker embedding is contained.")

  symbols = ["PADDING"] + list(symbol_mapping.keys())
  symbol_emb = get_symbol_embedding_weights(checkpoint)
  symbol_emb = symbol_emb.cpu().numpy()
  symbols_csv = embeddings_to_csv(symbol_emb, symbols)
  symbols_csv.to_csv(ns.output_directory / "symbol-embeddings.csv",
                     header=None, index=True, sep="\t")

  if hparams.use_speaker_embedding:
    speaker_emb = get_speaker_embedding_weights(checkpoint)
    speaker_emb = speaker_emb.cpu().numpy()
    speakers_csv = embeddings_to_csv(
        speaker_emb, ["PADDING"] + list(speaker_mapping.keys()))
    speakers_csv.to_csv(ns.output_directory / "speaker-embeddings.csv",
                        header=None, index=True, sep="\t")

  sims = get_similarities(symbol_emb)
  df = sims_to_csv_v2(sims, symbols)
  df.to_csv(ns.output_directory / "similarities.csv",
            header=None, index=True, sep="\t")
  emb_normed = norm2emb(symbol_emb)

  fig_2d = emb_plot_2d(emb_normed, symbols)
  plt.plot(fig_2d, filename=str(
      ns.output_directory / "2d.html"), auto_open=False)

  fig_3d = emb_plot_3d(emb_normed, symbols)
  plt.plot(fig_3d, filename=str(
      ns.output_directory / "3d.html"), auto_open=False)

  logger.info(f"Saved analysis to: {ns.output_directory.absolute()}")
  return True


def compare_embeddings(checkpoint1: Path, checkpoint2: Path, device: torch.device, output_directory: Path) -> bool:
  logger = getLogger(__name__)
  set_torch_thread_to_max()

  if not checkpoint1.is_file():
    logger.error("Checkpoint 1 was not found!")
    return False

  if not checkpoint2.is_file():
    logger.error("Checkpoint 2 was not found!")
    return False

  checkpoint1_dict = try_load_checkpoint(checkpoint1, device, logger)
  if checkpoint1_dict is None:
    return False

  checkpoint2_dict = try_load_checkpoint(checkpoint2, device, logger)
  if checkpoint2_dict is None:
    return False

  output_directory.mkdir(parents=True, exist_ok=True)

  symbol_mapping1 = get_symbol_mapping(checkpoint1_dict)
  symbol_mapping2 = get_symbol_mapping(checkpoint2_dict)
  symbol_mapping1["PADDING"] = 0
  symbol_mapping2["PADDING"] = 0
  symbol_emb1 = get_symbol_embedding_weights(checkpoint1_dict).cpu().numpy()
  symbol_emb2 = get_symbol_embedding_weights(checkpoint2_dict).cpu().numpy()

  sims = OrderedDict()
  for symbol1, index1 in symbol_mapping1.items():
    if symbol1 in symbol_mapping2:
      index2 = symbol_mapping2[symbol1]
      vec1 = symbol_emb1[index1]
      vec2 = symbol_emb2[index2]
      dist = 1 - cosine(vec1, vec2)
      sims[symbol1] = dist

  sims_avg = mean(sims.values())
  sims_max = max(sims.values())
  sims_min = min(sims.values())
  sims_med = median(sims.values())

  sims["MIN"] = sims_min
  sims["MAX"] = sims_max
  sims["AVG"] = sims_avg
  sims["MED"] = sims_med

  df = pd.DataFrame(sims.items(), columns=["Symbol", "Cosine similarity"])
  df.to_csv(output_directory / "similarities.csv",
            header=True, index=False, sep="\t")

  logger.info(f"Saved analysis to: {output_directory.absolute()}")
