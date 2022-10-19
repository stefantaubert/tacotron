

from tacotron.frontend.main import split_tone


def test_vowel_toned__returns_vowel_tone():
  phoneme, tone = split_tone("i˧˩˧")
  assert phoneme == "i"
  assert tone == "˧˩˧"


def test_diphthong_toned__returns_vowel_tone():
  phoneme, tone = split_tone("ai˩˧")
  assert phoneme == "ai"
  assert tone == "˩˧"


def test_consonant__returns_consonant_empty():
  phoneme, tone = split_tone("b")
  assert phoneme == "b"
  assert tone == "-"


def test_consonant_toned__returns_consonant_empty():
  phoneme, tone = split_tone("b˧")
  assert phoneme == "b"
  assert tone == "˧"
