from pathlib import Path
from typing import Optional, cast

import pandas as pd
import plotly.offline as plt
from tacotron.analysis import embeddings_to_csv
from tacotron.analysis import plot_embeddings as plot_embeddings_core
from tacotron.app.io import get_checkpoints_dir, get_train_dir
from tacotron.utils import get_custom_or_last_checkpoint, prepare_logger


def get_analysis_root_dir(train_dir: Path) -> Path:
  return train_dir / "analysis"


def _save_similarities_csv(analysis_dir: Path, checkpoint_it: int, df: pd.DataFrame) -> None:
  path = analysis_dir / f"{checkpoint_it}.csv"
  df.to_csv(path, header=None, index=False, sep="\t")


def _save_symbol_weights_csv(analysis_dir: Path, checkpoint_it: int, df: pd.DataFrame) -> None:
  path = analysis_dir / f"{checkpoint_it}_symbol_weights.csv"
  df.to_csv(path, header=None, index=True, sep="\t")


def _save_speaker_weights_csv(analysis_dir: Path, checkpoint_it: int, df: pd.DataFrame) -> None:
  path = analysis_dir / f"{checkpoint_it}_speaker_weights.csv"
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
