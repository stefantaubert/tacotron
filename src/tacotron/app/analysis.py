import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.offline as plt
from tacotron.analysis import plot_embeddings as plot_embeddings_core
from tacotron.app.io import get_checkpoints_dir, get_train_dir
from tacotron.core import CheckpointTacotron
from tacotron.utils import (get_custom_or_last_checkpoint, get_subdir,
                            prepare_logger, save_df)


def get_analysis_root_dir(train_dir: Path) -> Path:
  return get_subdir(train_dir, "analysis", create=True)


def _save_similarities_csv(analysis_dir: Path, checkpoint_it: int, df: pd.DataFrame):
  path = os.path.join(analysis_dir, f"{checkpoint_it}.csv")
  save_df(df, path, header_columns=None)


def _save_2d_plot(analysis_dir: Path, checkpoint_it: int, fig):
  path = os.path.join(analysis_dir, f"{checkpoint_it}_2d.html")
  plt.plot(fig, filename=path, auto_open=False)


def _save_3d_plot(analysis_dir: Path, checkpoint_it: int, fig):
  path = os.path.join(analysis_dir, f"{checkpoint_it}_3d.html")
  plt.plot(fig, filename=path, auto_open=False)


def plot_embeddings(base_dir: Path, train_name: str, custom_checkpoint: Optional[int] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)
  analysis_dir = get_analysis_root_dir(train_dir)

  logger = prepare_logger()

  checkpoint_path, checkpoint_it = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  checkpoint = CheckpointTacotron.load(checkpoint_path, logger)

  # pylint: disable=no-member
  text, fig_2d, fig_3d = plot_embeddings_core(
    symbols=checkpoint.get_symbols(),
    emb=checkpoint.get_symbol_embedding_weights(),
    logger=logger
  )

  _save_similarities_csv(analysis_dir, checkpoint_it, text)
  _save_2d_plot(analysis_dir, checkpoint_it, fig_2d)
  _save_3d_plot(analysis_dir, checkpoint_it, fig_3d)
  logger.info(f"Saved analysis to: {analysis_dir}")
