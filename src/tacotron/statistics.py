
from logging import getLogger

from tacotron.checkpoint_handling import *
from tacotron.utils import get_symbol_printable, log_hparams


def get_checkpoint_statistics(checkpoint: CheckpointDict):
  logger = getLogger(__name__)
  hparams = get_hparams(checkpoint)
  log_hparams(hparams, logger)

  iteration = get_iteration(checkpoint)
  logger.info(f"Current iteration: {iteration}")

  learning_rate = get_learning_rate(checkpoint)
  logger.info(f"Current learning rate: {learning_rate}")

  symbol_mapping = get_symbol_mapping(checkpoint)
  logger.info(
      f"Symbols: {' '.join(get_symbol_printable(symbol) for symbol in symbol_mapping.keys())} (#{len(symbol_mapping)}, dim: {hparams.symbols_embedding_dim})")

  if hparams.use_stress_embedding:
    assert has_stress_mapping(checkpoint)
    stress_mapping = get_stress_mapping(checkpoint)
    logger.info(
        f"Stresses: {' '.join(stress_mapping.keys())} (#{len(stress_mapping)})")
  else:
    logger.info("Use no stress embedding.")

  if hparams.use_speaker_embedding:
    assert has_speaker_mapping(checkpoint)
    speaker_mapping = get_speaker_mapping(checkpoint)
    logger.info(
        f"Speakers: {', '.join(sorted(speaker_mapping.keys()))} (#{len(speaker_mapping)}, dim: {hparams.speakers_embedding_dim})")
  else:
    logger.info("Use no speaker embedding.")

  optimizer_state = get_optimizer_state(checkpoint)
  model_state = get_model_state(checkpoint)
  if hparams.use_exponential_lr_decay:
    assert has_scheduler_state(checkpoint)
    scheduler_state = get_scheduler_state(checkpoint)
  else:
    logger.info("Used no exponential learning rate decay.")
