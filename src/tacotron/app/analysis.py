from collections import OrderedDict
import math
from statistics import mean, median
import numpy as np
from scipy.spatial.distance import cosine
from logging import getLogger
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import plotly.offline as plt
from tacotron.analysis import emb_plot_2d, emb_plot_3d, embeddings_to_csv, get_similarities, norm2emb, sims_to_csv, sims_to_csv_v2
from tacotron.analysis import plot_embeddings as plot_embeddings_core
from tacotron.app.io import get_checkpoints_dir, get_train_dir, load_checkpoint
from tacotron.core.checkpoint_handling import get_hparams, get_speaker_embedding_weights, get_speaker_mapping, get_symbol_embedding_weights, get_symbol_mapping
from tacotron.utils import get_custom_or_last_checkpoint, prepare_logger


def get_analysis_root_dir(train_dir: Path) -> Path:
  return train_dir / "analysis"


def _save_similarities_csv(analysis_dir: Path, checkpoint_it: int, df: pd.DataFrame) -> None:
  path = analysis_dir / f"{checkpoint_it}.csv"
  df.to_csv(path, header=None, index=False, sep="\t")


def _save_symbol_weights_csv(output_dir: Path, checkpoint_it: int, df: pd.DataFrame) -> None:
  path = output_dir / f"{checkpoint_it}_symbol_weights.csv"
  df.to_csv(path, header=None, index=True, sep="\t")


def _save_speaker_weights_csv(output_dir: Path, checkpoint_it: int, df: pd.DataFrame) -> None:
  path = output_dir / f"{checkpoint_it}_speaker_weights.csv"
  df.to_csv(path, header=None, index=True, sep="\t")


def _save_2d_plot(analysis_dir: Path, checkpoint_it: int, fig) -> None:
  path = analysis_dir / f"{checkpoint_it}_2d.html"
  plt.plot(fig, filename=str(path), auto_open=False)


def _save_3d_plot(analysis_dir: Path, checkpoint_it: int, fig) -> None:
  path = analysis_dir / f"{checkpoint_it}_3d.html"
  plt.plot(fig, filename=str(path), auto_open=False)


def plot_embeddings(base_dir: Path, train_name: str, custom_checkpoint: Optional[int] = None) -> None:
  train_dir = get_train_dir(base_dir, train_name)
  assert train_dir.is_dir()
  analysis_dir = get_analysis_root_dir(train_dir)

  logger = prepare_logger()

  checkpoint_path, checkpoint_it = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  checkpoint = cast(CheckpointTacotron, CheckpointTacotron.load(checkpoint_path, logger))
  analysis_dir.mkdir(parents=True, exist_ok=True)

  symbols_csv = embeddings_to_csv(
    keys=list(checkpoint.get_symbols()._ids_to_symbols.keys()),
    embeddings=checkpoint.get_symbol_embedding_weights(),
  )

  if checkpoint.get_hparams(logger).use_speaker_embedding:
    speakers_csv = embeddings_to_csv(
      keys=list(checkpoint.get_speakers().get_all_speakers()),
      embeddings=checkpoint.get_speaker_embedding_weights(),
    )
    _save_speaker_weights_csv(analysis_dir, checkpoint_it, speakers_csv)

  # pylint: disable=no-member
  text, fig_2d, fig_3d = plot_embeddings_core(
    symbols=checkpoint.get_symbols(),
    emb=checkpoint.get_symbol_embedding_weights(),
    logger=logger
  )

  _save_symbol_weights_csv(analysis_dir, checkpoint_it, symbols_csv)
  _save_similarities_csv(analysis_dir, checkpoint_it, text)
  _save_2d_plot(analysis_dir, checkpoint_it, fig_2d)
  _save_3d_plot(analysis_dir, checkpoint_it, fig_3d)
  logger.info(f"Saved analysis to: {analysis_dir}")


def plot_embeddings_v2(checkpoint: Path, output_directory: Path) -> bool:
  logger = getLogger(__name__)

  if not checkpoint.is_file():
    logger.error("Checkpoint was not found!")
    return False

  try:
    logger.debug(f"Loading checkpoint...")
    checkpoint_dict = load_checkpoint(checkpoint)
  except Exception as ex:
    logger.error("Checkpoint couldn't be loaded!")
    return False

  output_directory.mkdir(parents=True, exist_ok=True)

  symbol_mapping = get_symbol_mapping(checkpoint_dict)
  symbols = ["PADDING"] + list(symbol_mapping.keys())
  symbol_emb = get_symbol_embedding_weights(checkpoint_dict)
  symbol_emb = symbol_emb.cpu().numpy()
  symbols_csv = embeddings_to_csv(symbol_emb, symbols)
  symbols_csv.to_csv(output_directory / "symbol-embeddings.csv", header=None, index=True, sep="\t")

  hparams = get_hparams(checkpoint_dict)
  if hparams.use_speaker_embedding:

    speaker_mapping = get_speaker_mapping(checkpoint_dict)
    speaker_emb = get_speaker_embedding_weights(checkpoint_dict)
    speaker_emb = speaker_emb.cpu().numpy()
    speakers_csv = embeddings_to_csv(speaker_emb, ["PADDING"] + list(speaker_mapping.keys()))
    speakers_csv.to_csv(output_directory / "speaker-embeddings.csv",
                        header=None, index=True, sep="\t")

  sims = get_similarities(symbol_emb)
  df = sims_to_csv_v2(sims, symbols)
  df.to_csv(output_directory / "similarities.csv", header=None, index=True, sep="\t")
  emb_normed = norm2emb(symbol_emb)

  fig_2d = emb_plot_2d(emb_normed, symbols)
  plt.plot(fig_2d, filename=str(output_directory / "2d.html"), auto_open=False)

  fig_3d = emb_plot_3d(emb_normed, symbols)
  plt.plot(fig_3d, filename=str(output_directory / "3d.html"), auto_open=False)

  logger.info(f"Saved analysis to: {output_directory.absolute()}")


def compare_embeddings(checkpoint1: Path, checkpoint2: Path, output_directory: Path) -> bool:
  logger = getLogger(__name__)

  if not checkpoint1.is_file():
    logger.error("Checkpoint 1 was not found!")
    return False

  if not checkpoint2.is_file():
    logger.error("Checkpoint 2 was not found!")
    return False

  try:
    logger.debug(f"Loading checkpoint...")
    checkpoint1_dict = load_checkpoint(checkpoint1)
  except Exception as ex:
    logger.error("Checkpoint 1 couldn't be loaded!")
    return False

  try:
    logger.debug(f"Loading checkpoint...")
    checkpoint2_dict = load_checkpoint(checkpoint2)
  except Exception as ex:
    logger.error("Checkpoint 2 couldn't be loaded!")
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
  df.to_csv(output_directory / "similarities.csv", header=True, index=False, sep="\t")

  logger.info(f"Saved analysis to: {output_directory.absolute()}")
