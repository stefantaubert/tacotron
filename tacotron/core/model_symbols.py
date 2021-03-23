from typing import List
from src.core.common.globals import PADDING_SYMBOL


SHARED_SYMBOLS = [PADDING_SYMBOL]
SHARED_SYMBOLS_COUNT = len(SHARED_SYMBOLS)

def get_model_symbol_ids(symbol_ids: List[int], accent_ids: List[int], n_symbols: int,
                         accents_use_own_symbols: bool,
                         shared_symbol_count: int = SHARED_SYMBOLS_COUNT) -> List[int]:
  assert len(symbol_ids) == len(accent_ids)
  model_symbol_ids = [
    get_model_symbol_id(
      symbol_id,
      accent_id,
      n_symbols,
      accents_use_own_symbols,
      shared_symbol_count
    ) for symbol_id, accent_id in zip(symbol_ids, accent_ids)
  ]

  return model_symbol_ids


def get_model_symbols_count(n_symbols: int, n_accents: int, accents_use_own_symbols: bool,
                            shared_symbol_count: int = SHARED_SYMBOLS_COUNT) -> int:
  assert n_symbols >= shared_symbol_count

  if accents_use_own_symbols:
    return shared_symbol_count + (n_symbols - shared_symbol_count) * n_accents

  return n_symbols


def get_model_symbol_id(symbol_id: int, accent_id: int, n_symbols: int,
                        accents_use_own_symbols: bool,
                        shared_symbol_count: int = SHARED_SYMBOLS_COUNT) -> int:
  assert n_symbols >= shared_symbol_count
  assert symbol_id < n_symbols
  assert accent_id >= 0
  assert symbol_id >= 0

  if accents_use_own_symbols:
    is_shared_symbol = symbol_id < shared_symbol_count
    if is_shared_symbol:
      return symbol_id

    return symbol_id + (n_symbols - shared_symbol_count) * accent_id

  return symbol_id


def get_symbol_id(model_symbol_id: int, n_symbols: int, accents_use_own_symbols: bool,
                  shared_symbol_count: int = SHARED_SYMBOLS_COUNT) -> int:
  assert n_symbols >= shared_symbol_count
  assert model_symbol_id >= 0

  if accents_use_own_symbols:
    is_shared_symbol = model_symbol_id < shared_symbol_count
    if is_shared_symbol:
      return model_symbol_id

    n_symbols_wo_pad = n_symbols - shared_symbol_count
    if n_symbols_wo_pad == 1:
      # if it is not shared it must be the symbol left
      return n_symbols - 1

    return model_symbol_id % n_symbols_wo_pad

  return model_symbol_id
