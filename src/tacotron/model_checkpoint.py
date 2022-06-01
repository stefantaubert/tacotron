# from dataclasses import asdict, dataclass
# from logging import Logger
# from pathlib import Path
# from typing import Optional
# from typing import OrderedDict as OrderedDictType

# from tacotron.checkpoint import Checkpoint
# from tacotron.hparams import HParams
# from tacotron.model import (SPEAKER_EMBEDDING_LAYER_NAME,
#                             SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2)
# from text_utils import Speaker, SpeakersDict, Symbol, SymbolIdDict
# from torch import Tensor
# from torch.optim.adam import Adam  # pylint: disable=no-name-in-module
# from torch.optim.lr_scheduler import ExponentialLR


# @dataclass
# class CheckpointTacotron(Checkpoint):
#     # Renaming of any of these fields will destroy previous models!
#     speaker_id_dict: OrderedDictType[Speaker, int]
#     symbol_id_dict: OrderedDictType[Symbol, int]

#     @classmethod
#     def from_instances(cls, model: Tacotron2, optimizer: Adam, hparams: HParams, iteration: int, symbols: SymbolIdDict, speakers: SpeakersDict, scheduler: Optional[ExponentialLR]):
#         result = cls(
#             model_state_dict=model.state_dict(),
#             optimizer_state_dict=optimizer.state_dict(),
#             iteration=iteration,
#             hparams=asdict(hparams),
#             symbol_id_dict=symbols.raw(),
#             speaker_id_dict=speakers.raw(),
#             scheduler_state_dict=scheduler.state_dict() if scheduler else None,
#         )
#         return result

#     # pylint: disable=arguments-differ
#     def get_hparams(self, logger: Logger) -> HParams:
#         return super().get_hparams(logger, HParams)

#     def get_symbols(self) -> SymbolIdDict:
#         return SymbolIdDict.from_raw(self.symbol_id_dict)

#     def get_speakers(self) -> SpeakersDict:
#         return SpeakersDict.from_raw(self.speaker_id_dict)

#     def get_symbol_embedding_weights(self) -> Tensor:
#         assert SYMBOL_EMBEDDING_LAYER_NAME in self.model_state_dict
#         pretrained_weights = self.model_state_dict[SYMBOL_EMBEDDING_LAYER_NAME]
#         return pretrained_weights

#     def get_speaker_embedding_weights(self) -> Tensor:
#         assert SPEAKER_EMBEDDING_LAYER_NAME in self.model_state_dict
#         pretrained_weights = self.model_state_dict[SPEAKER_EMBEDDING_LAYER_NAME]
#         return pretrained_weights

#     @classmethod
#     def load(cls, checkpoint_path: Path, logger: Logger):
#         result = super().load(checkpoint_path, logger)
#         # pylint: disable=no-member
#         logger.info(f'Including {len(result.symbol_id_dict)} symbol(s).')
#         logger.info(f'Including {len(result.speaker_id_dict)} speaker(s).')
#         return result
