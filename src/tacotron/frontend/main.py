from collections import OrderedDict
from itertools import chain
from logging import getLogger
from typing import Generator, Iterable, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from tacotron.frontend.ipa_symbols import DURATION_MARKERS, TONE_MARKERS
from tacotron.frontend.stress_detection import StressType, split_stress_ipa_arpa
from tacotron.hparams import HParams
from tacotron.logging import LOGGER_NAME
from tacotron.typing import (Duration, DurationMapping, Durations, Entries, Mapping, MappingId,
                             Speaker, SpeakerMapping, Stress, Stresses, StressMapping, Symbol,
                             SymbolMapping, Symbols, Tone, ToneMapping, Tones)
from tacotron.utils import cut_string

PADDING_SHIFT = 1

NA_LABEL = "-"  # "N/A"

STRESS_LABELS: OrderedDictType[StressType, Stress] = OrderedDict((
    (StressType.UNSTRESSED, "0"),
    (StressType.PRIMARY, "1"),
    (StressType.SECONDARY, "2"),
    # (StressType.NOT_APPLICABLE, "N/A"),
    (StressType.NOT_APPLICABLE, NA_LABEL),
))


def get_mappings_count(mapping: Mapping) -> int:
  return len(mapping) + PADDING_SHIFT


def get_mapped_indices(it: Iterable[str], mapping: Mapping) -> Generator[Optional[MappingId], None, None]:
  result = (mapping.get(entry, None) for entry in it)
  return result


def get_map_keys(symbols: Symbols, hparams: HParams) -> Tuple[Symbols, Optional[Stresses], Optional[Tones], Optional[Durations]]:
  # Order: stresses -> tones -> durations
  stresses = None
  if hparams.use_stress_embedding:
    symbols, stresses = split_stresses(
        symbols, hparams.symbols_are_ipa)

  tones = None
  if hparams.use_tone_embedding:
    symbols, tones = split_tones(symbols)

  durations = None
  if hparams.use_duration_embedding:
    symbols, durations = split_durations(symbols)

  return symbols, stresses, tones, durations


def create_mappings(valset: Entries, trainset: Entries, hparams: HParams) -> Tuple[SymbolMapping, Optional[StressMapping], Optional[ToneMapping], Optional[DurationMapping], Optional[SpeakerMapping]]:
  logger = getLogger(LOGGER_NAME)
  # Order: stresses -> tones -> durations
  unique_symbols = set(get_symbols_iterator(valset, trainset))

  stress_mapping = None
  if hparams.use_stress_embedding:
    unique_symbols, stress_mapping = create_stress_mapping(unique_symbols, hparams.symbols_are_ipa)

  tone_mapping = None
  if hparams.use_tone_embedding:
    if not hparams.symbols_are_ipa:
      logger.warning("If use_tone_embedding is True the symbols need to be in IPA!")
    unique_symbols, tone_mapping = create_tone_mapping(unique_symbols)

  duration_mapping = None
  if hparams.use_duration_embedding:
    unique_symbols, duration_mapping = create_duration_mapping(unique_symbols)

  symbol_mapping = build_mapping(unique_symbols)

  speaker_mapping = None
  if hparams.use_speaker_embedding:
    unique_speakers = set(get_speakers_iterator(valset, trainset))
    speaker_mapping = build_mapping(unique_speakers)

  return symbol_mapping, stress_mapping, tone_mapping, duration_mapping, speaker_mapping


def split_stress(symbol: Symbol, is_ipa: bool) -> Tuple[Symbol, Stress]:
  raw_symbol, stress_type = split_stress_ipa_arpa(symbol, is_ipa)
  stress_symbol = STRESS_LABELS[stress_type]
  return raw_symbol, stress_symbol


def split_tone(symbol: Symbol) -> Tuple[Symbol, Tone]:
  core_symb, tone = cut_string(symbol, TONE_MARKERS)
  if tone == "":
    tone = NA_LABEL
  return core_symb, tone


def split_duration(symbol: Symbol) -> Tuple[Symbol, Duration]:
  core_symb, duration = cut_string(symbol, DURATION_MARKERS)
  if duration == "":
    duration = NA_LABEL
  return core_symb, duration


def split_stresses(symbols: Iterable[Symbol], is_ipa: bool) -> Tuple[Symbols, Stresses]:
  res_symbols = []
  stresses = []
  for symbol in symbols:
    symbol_core, stress = split_stress(symbol, is_ipa)
    res_symbols.append(symbol_core)
    stresses.append(stress)
  return tuple(res_symbols), tuple(stresses)


def split_tones(symbols: Iterable[Symbol]) -> Tuple[Symbols, Tones]:
  res_symbols = []
  tones = []
  for symbol in symbols:
    symbol_core, tone = split_tone(symbol)
    res_symbols.append(symbol_core)
    tones.append(tone)
  return tuple(res_symbols), tuple(tones)


def split_durations(symbols: Iterable[Symbol]) -> Tuple[Symbols, Stresses]:
  res_symbols = []
  durations = []
  for symbol in symbols:
    symbol_core, duration = split_duration(symbol)
    res_symbols.append(symbol_core)
    durations.append(duration)
  return tuple(res_symbols), tuple(durations)


def get_speakers_iterator(valset: Entries, trainset: Entries) -> Generator[Speaker, None, None]:
  all_valspeakers = (entry.speaker_name for entry in valset)
  all_trainspeakers = (entry.speaker_name for entry in trainset)
  all_speakers = chain(all_valspeakers, all_trainspeakers)
  yield from all_speakers


def create_speaker_mapping(valset: Entries, trainset: Entries) -> SpeakerMapping:
  all_valspeakers = (entry.speaker_name for entry in valset)
  all_trainspeakers = (entry.speaker_name for entry in trainset)
  all_speakers = chain(all_valspeakers, all_trainspeakers)
  unique_speakers = set(all_speakers)

  speaker_ids = OrderedDict((
      (speaker, speaker_nr)
      for speaker_nr, speaker in enumerate(sorted(unique_speakers), start=PADDING_SHIFT)
  ))

  return speaker_ids


def get_symbols_iterator(valset: Entries, trainset: Entries) -> Generator[Symbol, None, None]:
  all_valsymbols = (entry.symbols for entry in valset)
  all_trainsymbols = (entry.symbols for entry in trainset)
  all_symbols = chain(all_valsymbols, all_trainsymbols)
  symbols = (symbol for entry_symbols in all_symbols for symbol in entry_symbols)
  yield from symbols


def create_stress_mapping(symbols_it: Set[Symbol], symbols_are_ipa: bool) -> Tuple[Set[Symbol], StressMapping]:
  all_symbols, all_stresses = split_stresses(symbols_it, symbols_are_ipa)

  unique_symbols = set(all_symbols)
  unique_stresses = set(all_stresses)
  stress_mapping = build_mapping(unique_stresses)

  return unique_symbols, stress_mapping


def create_tone_mapping(symbols_it: Set[Symbol]) -> Tuple[Set[Symbol], ToneMapping]:
  all_symbols, all_tones = split_tones(symbols_it)

  unique_symbols = set(all_symbols)
  unique_tones = set(all_tones)
  stress_mapping = build_mapping(unique_tones)

  return unique_symbols, stress_mapping


def create_duration_mapping(symbols_it: Set[Symbol]) -> Tuple[Set[Symbol], DurationMapping]:
  all_symbols, all_durations = split_durations(symbols_it)

  unique_symbols = set(all_symbols)
  unique_durations = set(all_durations)
  stress_mapping = build_mapping(unique_durations)

  return unique_symbols, stress_mapping


def build_mapping(characters: Set[str]) -> OrderedDictType[str, int]:
  mapping = OrderedDict((
    (character, character_nr)
    for character_nr, character in enumerate(sorted(characters), start=PADDING_SHIFT)
  ))
  return mapping
