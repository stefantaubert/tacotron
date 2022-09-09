from typing import Tuple

TONE_1 = "¹"
TONE_2 = "²"
TONE_3 = "³"
TONE_4 = "⁴"
TONE_5 = "⁵"
TONE_6 = "⁶"
TONE_7 = "⁷"
TONE_8 = "⁸"
TONE_9 = "⁹"


TONE_EXTRA_HIGH = "˥"
TONE_HIGH = "˦"
TONE_MID = "˧"
TONE_LOW = "˨"
TONE_EXTRA_LOW = "˩"

TONE_EXTRA_HIGH_ALT = "\u030B"
TONE_HIGH_ALT = "\u0301"
TONE_MID_ALT = "\u0304"
TONE_LOW_ALT = "\u0300"
TONE_EXTRA_LOW_ALT = "\u030F"

TONES = {
  TONE_1,
  TONE_2,
  TONE_3,
  TONE_4,
  TONE_5,
  TONE_6,
  TONE_7,
  TONE_8,
  TONE_9,
  TONE_EXTRA_HIGH,
  TONE_EXTRA_HIGH,
  TONE_HIGH,
  TONE_MID,
  TONE_LOW,
  TONE_EXTRA_LOW,
  TONE_EXTRA_HIGH_ALT,
  TONE_EXTRA_HIGH_ALT,
  TONE_HIGH_ALT,
  TONE_MID_ALT,
  TONE_LOW_ALT,
  TONE_EXTRA_LOW_ALT,
}


def separate_syllable_ipa_into_phonemes_and_tones(syllable_ipa: str) -> Tuple[str, str]:
  syllable_phonemes = ""
  syllable_tones = ""
  for character in syllable_ipa:
    if character in TONES:
      syllable_tones += character
    else:
      # No characters after tones allowed
      assert syllable_tones == ""
      syllable_phonemes += character
  return syllable_phonemes, syllable_tones


def split_phoneme_and_tone(ipa: str) -> Tuple[str, str]:
  phoneme = ""
  tone = ""
  for character in ipa:
    if character in TONES:
      tone += character
    else:
      # No phonemes after tones allowed
      if tone != "":
        return ipa, ""
      phoneme += character
  return phoneme, tone
