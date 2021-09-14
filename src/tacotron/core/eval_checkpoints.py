from logging import Logger
from pathlib import Path
from typing import Dict, Optional

import torch
from tacotron.core.dataloader import (SymbolsMelCollate, parse_batch,
                                      prepare_valloader)
from tacotron.core.hparams import HParams
from tacotron.core.model_checkpoint import CheckpointTacotron
from tacotron.core.training import Tacotron2Loss, load_model, validate_model
from tacotron.utils import (filter_checkpoints, get_all_checkpoint_iterations,
                            get_checkpoint, overwrite_custom_hparams)
from tqdm import tqdm
from tts_preparation import PreparedDataList


def eval_checkpoints(custom_hparams: Optional[Dict[str, str]], checkpoint_dir: Path, select: int, min_it: int, max_it: int, n_symbols: int, n_accents: int, n_speakers: int, valset: PreparedDataList, logger: Logger) -> None:
  its = get_all_checkpoint_iterations(checkpoint_dir)
  logger.info(f"Available iterations {its}")
  filtered_its = filter_checkpoints(its, select, min_it, max_it)
  if len(filtered_its) > 0:
    logger.info(f"Selected iterations: {filtered_its}")
  else:
    logger.info("None selected. Exiting.")
    return

  hparams = HParams(
    n_speakers=n_speakers,
    n_symbols=n_symbols,
    n_accents=n_accents
  )

  hparams = overwrite_custom_hparams(hparams, custom_hparams)

  collate_fn = SymbolsMelCollate(
    hparams.n_frames_per_step,
    padding_symbol_id=0,  # TODO: refactor
    padding_accent_id=0  # TODO: refactor
    # padding_symbol_id=symbols.get_id(PADDING_SYMBOL),
    # padding_accent_id=accents.get_id(PADDING_ACCENT)
  )
  val_loader = prepare_valloader(hparams, collate_fn, valset, logger)

  result = []
  for checkpoint_iteration in tqdm(filtered_its):
    criterion = Tacotron2Loss()
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    full_checkpoint_path = get_checkpoint(checkpoint_dir, checkpoint_iteration)
    state_dict = CheckpointTacotron.load(full_checkpoint_path, logger).state_dict
    model = load_model(hparams, state_dict, logger)
    val_loss, _ = validate_model(model, criterion, val_loader, parse_batch)
    result.append((checkpoint_iteration, val_loss))
    logger.info(f"Validation loss {checkpoint_iteration}: {val_loss:9f}")

  logger.info("Result...")
  logger.info("Sorted after checkpoints:")

  result.sort()
  for cp, loss in result:
    logger.info(f"Validation loss {cp}: {loss:9f}")

  result = [(b, a) for a, b in result]
  result.sort()

  logger.info("Sorted after scores:")
  for loss, cp in result:
    logger.info(f"Validation loss {cp}: {loss:9f}")
