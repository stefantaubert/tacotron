from enum import IntEnum
from typing import Tuple

from tacotron.arpa_symbols import STRESS_NONE as ARPA_STRESS_NONE
from tacotron.arpa_symbols import STRESS_PRIMARY as ARPA_STRESS_PRIMARY
from tacotron.arpa_symbols import STRESS_SECONDARY as ARPA_STRESS_SECONDARY
from tacotron.arpa_symbols import VOWELS as ARPA_VOWELS
from tacotron.arpa_symbols import \
  VOWELS_WITH_NUMBERED_STRESSES as ARPA_VOWELS_WITH_NUMBERED_STRESSES
from tacotron.ipa_symbols import APPENDIX, ENG_DIPHTHONGS, SCHWAS
from tacotron.ipa_symbols import STRESS_PRIMARY as IPA_STRESS_PRIMARY
from tacotron.ipa_symbols import STRESS_SECONDARY as IPA_STRESS_SECONDARY
from tacotron.ipa_symbols import VOWELS as IPA_VOWELS
from tacotron.typing import Symbol


class StressType(IntEnum):
  UNSTRESSED = 0
  PRIMARY = 1
  SECONDARY = 2
  # Consonants and punctuation
  NOT_APPLICABLE = 3


ARPA_STRESS_MAP = {
    ARPA_STRESS_NONE: StressType.UNSTRESSED,
    ARPA_STRESS_PRIMARY: StressType.PRIMARY,
    ARPA_STRESS_SECONDARY: StressType.SECONDARY,
}

IPA_STRESS_MAP = {
    IPA_STRESS_PRIMARY: StressType.PRIMARY,
    IPA_STRESS_SECONDARY: StressType.SECONDARY,
}

IPA_STRESSABLE = {
    symbol
    for symbol in SCHWAS | ENG_DIPHTHONGS | IPA_VOWELS
}

IPA_APPENDIX = "".join(APPENDIX)


def split_stress_arpa(symbol: Symbol) -> Tuple[Symbol, StressType]:
  if symbol in ARPA_VOWELS:
    return symbol, StressType.UNSTRESSED
  if symbol in ARPA_VOWELS_WITH_NUMBERED_STRESSES:
    vowel = symbol[0:-1]
    stress = symbol[-1]
    assert stress in ARPA_STRESS_MAP
    stress_type = ARPA_STRESS_MAP[stress]
    return vowel, stress_type
  return symbol, StressType.NOT_APPLICABLE


def get_ipa_symbol_without_appendix(symbol: Symbol) -> Symbol:
  raw_symbol = symbol.rstrip(IPA_APPENDIX)
  return raw_symbol


def split_stress_ipa(symbol: Symbol) -> Tuple[Symbol, StressType]:
  raw_symbol = get_ipa_symbol_without_appendix(symbol)

  if raw_symbol in IPA_STRESSABLE:
    return symbol, StressType.UNSTRESSED

  if len(raw_symbol) > 1:
    first_symbol = raw_symbol[0]
    potential_raw_vowel = raw_symbol[1:]
    if potential_raw_vowel in IPA_STRESSABLE:
      vowel = symbol[1:]
      if first_symbol in IPA_STRESS_MAP:
        stress_type = IPA_STRESS_MAP[first_symbol]
        return vowel, stress_type
      return symbol, StressType.NOT_APPLICABLE

  return symbol, StressType.NOT_APPLICABLE
