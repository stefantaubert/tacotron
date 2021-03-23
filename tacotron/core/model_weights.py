from collections import OrderedDict
from logging import Logger
from typing import Optional
from typing import OrderedDict as OrderedDictType

from torch import Tensor

from text_utils import SpeakersDict
from text_utils import SymbolIdDict
from text_utils import SymbolsMap
from tacotron.core.hparams import HParams
from tacotron.core.model import get_speaker_weights, get_symbol_weights
from tacotron.core.model_symbols import get_model_symbol_id


def symbols_ids_map_to_model_symbols_ids_map(symbols_id_map: OrderedDictType[int, int], n_accents: int, n_symbols: int, accents_use_own_symbols: bool) -> OrderedDictType[int, int]:
  res: OrderedDictType[int, int] = OrderedDict()

  for accent_id in range(n_accents):
    for map_to_symbol_id, map_from_symbol_id in symbols_id_map.items():

      map_to_model_id = get_model_symbol_id(
        map_to_symbol_id,
        accent_id,
        n_symbols,
        accents_use_own_symbols
      )

      res[map_to_model_id] = map_from_symbol_id

    if not accents_use_own_symbols:
      break

  return res


def map_weights(model_symbols_id_map: OrderedDictType[int, int], model_weights: Tensor, trained_weights: Tensor, logger: Logger) -> None:
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

  model_symbols_id_map = symbols_ids_map_to_model_symbols_ids_map(
    symbols_id_map,
    hparams.n_accents,
    n_symbols=hparams.n_symbols,
    accents_use_own_symbols=hparams.accents_use_own_symbols
  )

  model_weights = get_symbol_weights(hparams)

  map_weights(
    model_symbols_id_map=model_symbols_id_map,
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


def get_mapped_speaker_weights(model_speakers: SpeakersDict, trained_weights: Tensor, trained_speaker: SpeakersDict, map_from_speaker_name: str, hparams: HParams, logger: Logger) -> Tensor:
  map_from_id = trained_speaker.get_id(map_from_speaker_name)
  speakers_map: OrderedDictType[int, int] = OrderedDict(
    {new_speaker_id: map_from_id for new_speaker_id in model_speakers.values()})

  weights = get_speaker_weights(hparams)

  map_weights(
    model_symbols_id_map=speakers_map,
    model_weights=weights,
    trained_weights=trained_weights,
    logger=logger
  )

  return weights
