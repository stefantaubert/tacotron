import logging
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Dict, Optional
from typing import OrderedDict as OrderedDictType

import torch
from tacotron.checkpoint import Checkpoint
from tacotron.core.hparams import HParams
from tacotron.core.model import (SPEAKER_EMBEDDING_LAYER_NAME,
                                 SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2)
from tacotron.utils import get_pytorch_filename, overwrite_custom_hparams
from text_utils import AccentsDict, SpeakersDict, SymbolIdDict
from torch import Tensor
from torch.optim.adam import Adam  # pylint: disable=no-name-in-module
from torch.optim.lr_scheduler import ExponentialLR


@dataclass
class CheckpointTacotron(Checkpoint):
  # Renaming of any of these fields will destroy previous models!
  speakers: OrderedDictType[str, int]
  symbols: OrderedDictType[str, int]

  @classmethod
  def from_instances(cls, model: Tacotron2, optimizer: Adam, hparams: HParams, iteration: int, symbols: SymbolIdDict, speakers: SpeakersDict, scheduler: Optional[ExponentialLR]):
    result = cls(
      state_dict=model.state_dict(),
      optimizer=optimizer.state_dict(),
      learning_rate=hparams.learning_rate,
      iteration=iteration,
      hparams=asdict(hparams),
      symbols=symbols.raw(),
      speakers=speakers.raw(),
      scheduler_state_dict=scheduler.state_dict() if scheduler else None,
    )
    return result

  # pylint: disable=arguments-differ
  def get_hparams(self, logger: Logger) -> HParams:
    return super().get_hparams(logger, HParams)

  def get_symbols(self) -> SymbolIdDict:
    return SymbolIdDict.from_raw(self.symbols)

  def get_speakers(self) -> SpeakersDict:
    return SpeakersDict.from_raw(self.speakers)

  def get_symbol_embedding_weights(self) -> Tensor:
    assert SYMBOL_EMBEDDING_LAYER_NAME in self.state_dict
    pretrained_weights = self.state_dict[SYMBOL_EMBEDDING_LAYER_NAME]
    return pretrained_weights

  def get_speaker_embedding_weights(self) -> Tensor:
    assert SPEAKER_EMBEDDING_LAYER_NAME in self.state_dict
    pretrained_weights = self.state_dict[SPEAKER_EMBEDDING_LAYER_NAME]
    return pretrained_weights

  @classmethod
  def load(cls, checkpoint_path: str, logger: Logger):
    result = super().load(checkpoint_path, logger)
    # pylint: disable=no-member
    logger.info(f'Including {len(result.symbols)} symbols.')
    logger.info(f'Including {len(result.speakers)} speaker(s).')
    return result
