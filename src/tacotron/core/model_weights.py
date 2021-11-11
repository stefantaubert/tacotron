from collections import OrderedDict
from logging import Logger
from typing import Dict, Optional
from typing import OrderedDict as OrderedDictType

from sklearn.base import BaseEstimator
from tacotron.core.hparams import HParams
from tacotron.core.model import get_speaker_weights, get_symbol_weights
from text_utils import SpeakersDict, SymbolIdDict, SymbolsMap
from text_utils.types import Speaker, Symbol, SymbolId
from torch import Tensor
import numpy as np


def map_weights(model_symbols_id_map: OrderedDictType[SymbolId, SymbolId], model_weights: Tensor, trained_weights: Tensor, logger: Logger) -> None:
  for map_to_id, map_from_id in model_symbols_id_map.items():
    assert 0 <= map_to_id < model_weights.shape[0]
    assert 0 <= map_from_id < trained_weights.shape[0]
    old_weights = model_weights[map_to_id].cpu().numpy()[:5]
    model_weights[map_to_id] = trained_weights[map_from_id]
    logger.debug(f"Mapped {map_from_id} to {map_to_id}.")
    logger.debug(f"Old {old_weights}")
    logger.debug(f"New {model_weights[map_to_id].cpu().numpy()[:5]}")


def get_mapped_symbol_weights(model_symbols: SymbolIdDict, trained_weights: Tensor, trained_symbols: SymbolIdDict, custom_mapping: Optional[SymbolsMap], hparams: HParams, logger: Logger) -> Tensor:
  symbols_match_not_model = trained_weights.shape[0] != len(trained_symbols)
  if symbols_match_not_model:
    logger.exception(
      f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({len(trained_symbols)}).")
    raise Exception()

  if custom_mapping is None:
    symbols_map = SymbolsMap.from_intersection(
      map_to=model_symbols.get_all_symbols(),
      map_from=trained_symbols.get_all_symbols(),
    )
  else:
    symbols_map = custom_mapping
    symbols_map.remove_unknown_symbols(
      known_to_symbol=model_symbols.get_all_symbols(),
      known_from_symbols=trained_symbols.get_all_symbols()
    )

  # Remove all empty mappings
  symbols_wo_mapping = symbols_map.get_symbols_with_empty_mapping()
  symbols_map.pop_batch(symbols_wo_mapping)

  symbols_id_map = symbols_map.convert_to_symbols_ids_map(
    to_symbols=model_symbols,
    from_symbols=trained_symbols,
  )

  model_weights = get_symbol_weights(hparams)

  map_weights(
    model_symbols_id_map=symbols_id_map,
    model_weights=model_weights,
    trained_weights=trained_weights,
    logger=logger
  )

  not_existing_symbols = model_symbols.get_all_symbols() - symbols_map.keys()
  no_mapping = symbols_wo_mapping | not_existing_symbols
  if len(no_mapping) > 0:
    logger.warning(f"Following symbols were not mapped: {no_mapping}")
  else:
    logger.info("All symbols were mapped.")

  return model_weights


def get_mapped_speaker_weights(model_speaker_id_dict: SpeakersDict, trained_weights: Tensor, trained_speaker: SpeakersDict, map_from_speaker_name: Speaker, hparams: HParams, logger: Logger) -> Tensor:
  map_from_id = trained_speaker.get_id(map_from_speaker_name)
  speakers_map: OrderedDictType[int, int] = OrderedDict(
    {new_speaker_id: map_from_id for new_speaker_id in model_speaker_id_dict.values()})

  weights = get_speaker_weights(hparams)

  map_weights(
    model_symbols_id_map=speakers_map,
    model_weights=weights,
    trained_weights=trained_weights,
    logger=logger
  )

  return weights


class Add_Symbol_Embeddings(BaseEstimator):
  def __init__(self):
    pass

  def get_symbols_mapping(self, input_symbols: SymbolIdDict, target_symbols: SymbolIdDict):
    symbols_mapping = SymbolsMap.from_intersection(
      map_from=input_symbols.get_all_symbols(),
      map_to=target_symbols.get_all_symbols(),
    )

    self.symbols_id_mapping = symbols_mapping.convert_to_symbols_ids_map(
      from_symbols=input_symbols,
      to_symbols=target_symbols,
    )

    return self.symbols_id_mapping

  def get_difference_vector_to_add(input_weights: Tensor, target_weights: Tensor, index_mapping: Dict[int, int]):
    assert input_weights.shape[1] == target_weights.shape[1]
    difference_vectors = Tensor([target_weights[index_speaker_2] - input_weights[index_speaker_1]
                                 for index_speaker_1, index_speaker_2 in index_mapping.items()])
    number_of_symbols = difference_vectors.shape[0]
    average_difference_vector = 1 / number_of_symbols * np.sum(difference_vectors, axis=0)

    return average_difference_vector

  def fit(self, input_weights: Tensor, input_symbols: SymbolIdDict, target_weights: Tensor, target_symbols: SymbolIdDict) -> Tensor:
    symbols_mapping = self.get_symbols_mapping(input_symbols, target_symbols)
    self.average_shift = get_difference_vector_to_add(
      input_weights, target_weights, symbols_mapping)
    return self

  def predict(input_symbol: Symbol):
    pass


def add_symbol_embedding(input_weights: Tensor, input_symbols: SymbolIdDict, input_symbol: Symbol, target_weights: Tensor, target_symbols: SymbolIdDict) -> Tensor:
  symbols_mapping = SymbolsMap.from_intersection(
    map_from=input_symbols.get_all_symbols(),
    map_to=target_symbols.get_all_symbols(),
  )

  symbols_id_mapping = symbols_mapping.convert_to_symbols_ids_map(
    from_symbols=input_symbols,
    to_symbols=target_symbols,
  )

  target_weights_updated: Tensor

  # TODO jasmin create embedding for input_symbol

  return target_weights_updated


def get_difference_vector_to_add(speaker_1: np.array, speaker_2: np.array, index_mapping: Dict[int, int]):
  assert speaker_1.shape[1] == speaker_2.shape[1]
  difference_vectors = np.array([speaker_2[index_speaker_2] - speaker_1[index_speaker_1]
                                 for index_speaker_1, index_speaker_2 in index_mapping.items()])
  number_of_symbols = difference_vectors.shape[0]
  average_difference_vector = 1 / number_of_symbols * np.sum(difference_vectors, axis=0)

  return average_difference_vector


def predict_with_adding(speaker_array, diff_vector):
  return speaker_array + diff_vector
