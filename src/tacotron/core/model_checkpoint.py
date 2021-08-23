import logging
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Dict, Optional
from typing import OrderedDict as OrderedDictType

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tacotron.checkpoint import Checkpoint
from tacotron.utils import (get_pytorch_filename,
                            overwrite_custom_hparams)
from tacotron.core.hparams import HParams
from tacotron.core.model import (SPEAKER_EMBEDDING_LAYER_NAME,
                                 SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2)
from text_utils import AccentsDict, SpeakersDict, SymbolIdDict
from torch import Tensor
from torch.optim.adam import Adam  # pylint: disable=no-name-in-module


@dataclass
class CheckpointTacotron(Checkpoint):
  # Renaming of any of these fields will destroy previous models!
  speakers: OrderedDictType[str, int]
  symbols: OrderedDictType[str, int]
  accents: OrderedDictType[str, int]

  @classmethod
  def from_instances(cls, model: Tacotron2, optimizer: Adam, hparams: HParams, iteration: int, symbols: SymbolIdDict, accents: AccentsDict, speakers: SpeakersDict, scheduler: ExponentialLR):
    result = cls(
      state_dict=model.state_dict(),
      optimizer=optimizer.state_dict(),
      learning_rate=hparams.learning_rate,
      iteration=iteration,
      hparams=asdict(hparams),
      symbols=symbols.raw(),
      accents=accents.raw(),
      speakers=speakers.raw(),
      scheduler_state_dict=scheduler.state_dict(),
    )
    return result

  # pylint: disable=arguments-differ
  def get_hparams(self, logger: Logger) -> HParams:
    return super().get_hparams(logger, HParams)

  def get_symbols(self) -> SymbolIdDict:
    return SymbolIdDict.from_raw(self.symbols)

  def get_accents(self) -> AccentsDict:
    return AccentsDict.from_raw(self.accents)

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
    logger.info(f'Including {len(result.accents)} accents.')
    logger.info(f'Including {len(result.speakers)} speaker(s).')
    return result


def convert_v1_to_v2_model(old_model_path: str, custom_hparams: Optional[Dict[str, str]], speakers: SpeakersDict, accents: AccentsDict, symbols: SymbolIdDict):
  checkpoint_dict = torch.load(old_model_path, map_location='cpu')
  hparams = HParams(
    n_speakers=len(speakers),
    n_accents=len(accents),
    n_symbols=len(symbols)
  )

  hparams = overwrite_custom_hparams(hparams, custom_hparams)

  chp = CheckpointTacotron(
    state_dict=checkpoint_dict["state_dict"],
    optimizer=checkpoint_dict["optimizer"],
    learning_rate=checkpoint_dict["learning_rate"],
    iteration=checkpoint_dict["iteration"] + 1,
    hparams=asdict(hparams),
    speakers=speakers.raw(),
    symbols=symbols.raw(),
    accents=accents.raw()
  )

  new_model_path = f"{old_model_path}_{get_pytorch_filename(chp.iteration)}"

  chp.save(new_model_path, logging.getLogger())
