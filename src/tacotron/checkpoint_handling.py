from collections import OrderedDict
from dataclasses import asdict
from logging import getLogger
from typing import Any, Dict, Optional
from typing import OrderedDict as OrderedDictType

from torch import Tensor
from torch.optim.adam import Adam  # pylint: disable=no-name-in-module
from torch.optim.lr_scheduler import ExponentialLR

from tacotron.hparams import HParams
from tacotron.model import SPEAKER_EMBEDDING_LAYER_NAME, SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2
from tacotron.typing import SpeakerMapping, StressMapping, SymbolMapping
from tacotron.utils import get_dataclass_from_dict

CheckpointDict = OrderedDictType[str, Any]

# Renaming of any of these fields will destroy previous models!
KEY_HPARAMS = "hparams"
KEY_OPTIMIZER_STATE = "optimizer_state"
# optional
KEY_SCHEDULER_STATE = "scheduler_state"
KEY_MODEL_STATE = "model_state"
KEY_ITERATION = "iteration"
KEY_LEARNING_RATE = "learning_rate"
KEY_SYMBOL_MAPPING = "symbol_mapping"
# optional
KEY_SPEAKER_MAPPING = "speaker_mapping"
# optional
KEY_STRESS_MAPPING = "stress_mapping"


def create(model: Tacotron2, optimizer: Adam, hparams: HParams, iteration: int, learning_rate: float, scheduler: Optional[ExponentialLR], symbol_mapping: SymbolMapping, stress_mapping: Optional[StressMapping], speaker_mapping: Optional[SpeakerMapping]) -> CheckpointDict:
  result = OrderedDict()
  result[KEY_HPARAMS] = asdict(hparams)
  result[KEY_MODEL_STATE] = model.state_dict()
  result[KEY_OPTIMIZER_STATE] = optimizer.state_dict()
  if scheduler is not None:
    result[KEY_SCHEDULER_STATE] = scheduler.state_dict()
  result[KEY_ITERATION] = iteration
  result[KEY_LEARNING_RATE] = learning_rate
  result[KEY_SYMBOL_MAPPING] = symbol_mapping
  if speaker_mapping is not None:
    result[KEY_SPEAKER_MAPPING] = speaker_mapping
  if stress_mapping is not None:
    result[KEY_STRESS_MAPPING] = stress_mapping
  return result


def convert_to_inference_only(checkpoint: CheckpointDict) -> None:
  checkpoint.pop(KEY_OPTIMIZER_STATE)
  checkpoint.pop(KEY_LEARNING_RATE)
  if has_scheduler_state(checkpoint):
    checkpoint.pop(KEY_SCHEDULER_STATE)


def has_speaker_mapping(checkpoint: CheckpointDict) -> bool:
  return KEY_SPEAKER_MAPPING in checkpoint


def get_speaker_mapping(checkpoint: CheckpointDict) -> SpeakerMapping:
  assert has_speaker_mapping(checkpoint)
  result = checkpoint[KEY_SPEAKER_MAPPING]
  return result


def get_symbol_mapping(checkpoint: CheckpointDict) -> SymbolMapping:
  assert KEY_SYMBOL_MAPPING in checkpoint
  result = checkpoint[KEY_SYMBOL_MAPPING]
  return result


def update_symbol_mapping(checkpoint: CheckpointDict, mapping: SymbolMapping) -> None:
  assert KEY_SYMBOL_MAPPING in checkpoint
  checkpoint[KEY_SYMBOL_MAPPING] = mapping


def get_iteration(checkpoint: CheckpointDict) -> int:
  assert KEY_ITERATION in checkpoint
  result = checkpoint[KEY_ITERATION]
  return result


def get_learning_rate(checkpoint: CheckpointDict) -> float:
  assert KEY_LEARNING_RATE in checkpoint
  result = checkpoint[KEY_LEARNING_RATE]
  return result


def has_stress_mapping(checkpoint: CheckpointDict) -> bool:
  return KEY_STRESS_MAPPING in checkpoint


def get_stress_mapping(checkpoint: CheckpointDict) -> StressMapping:
  assert has_stress_mapping(checkpoint)
  result = checkpoint[KEY_STRESS_MAPPING]
  return result


def get_model_state(checkpoint: CheckpointDict) -> Dict:
  assert KEY_MODEL_STATE in checkpoint
  result = checkpoint[KEY_MODEL_STATE]
  return result


def get_optimizer_state(checkpoint: CheckpointDict) -> Dict:
  assert KEY_OPTIMIZER_STATE in checkpoint
  result = checkpoint[KEY_OPTIMIZER_STATE]
  return result


def has_scheduler_state(checkpoint: CheckpointDict) -> bool:
  return KEY_SCHEDULER_STATE in checkpoint


def get_scheduler_state(checkpoint: CheckpointDict) -> Dict:
  assert has_scheduler_state(checkpoint)
  result = checkpoint[KEY_SCHEDULER_STATE]
  return result


def get_hparams(checkpoint: CheckpointDict) -> HParams:
  assert KEY_HPARAMS in checkpoint
  hparams = checkpoint[KEY_HPARAMS]
  result, ignored = get_dataclass_from_dict(hparams, HParams)
  if len(ignored) > 0:
    logger = getLogger(__name__)
    logger.warning(
        f"Ignored these hparams from checkpoint because they did not exist in the current HParams: {', '.join(sorted(ignored))} (#{len(ignored)})")
  return result


def get_symbol_embedding_weights(checkpoint: CheckpointDict) -> Tensor:
  model_state = get_model_state(checkpoint)
  assert SYMBOL_EMBEDDING_LAYER_NAME in model_state
  pretrained_weights = model_state[SYMBOL_EMBEDDING_LAYER_NAME]
  return pretrained_weights


def update_symbol_embedding_weights(checkpoint: CheckpointDict, weights: Tensor) -> None:
  model_state = get_model_state(checkpoint)
  assert SYMBOL_EMBEDDING_LAYER_NAME in model_state
  model_state[SYMBOL_EMBEDDING_LAYER_NAME] = weights


def get_speaker_embedding_weights(checkpoint: CheckpointDict) -> Tensor:
  model_state = get_model_state(checkpoint)
  assert SPEAKER_EMBEDDING_LAYER_NAME in model_state
  pretrained_weights = model_state[SPEAKER_EMBEDDING_LAYER_NAME]
  return pretrained_weights
