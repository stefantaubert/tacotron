from tacotron.utils import cut_string


def test_empty_empty__returns_empty_empty():
  rest_str, cut_str = cut_string("", {})
  assert rest_str == ""
  assert cut_str == ""


def test_empty_X__returns_empty_empty():
  rest_str, cut_str = cut_string("", {"X"})
  assert rest_str == ""
  assert cut_str == ""


def test_X_empty__returns_X_empty():
  rest_str, cut_str = cut_string("X", {""})
  assert rest_str == "X"
  assert cut_str == ""


def test_X_X__returns_empty_X():
  rest_str, cut_str = cut_string("X", {"X"})
  assert rest_str == ""
  assert cut_str == "X"


def test_yXzX_X__returns_yz_XX():
  rest_str, cut_str = cut_string("yXzX", {"X"})
  assert rest_str == "yz"
  assert cut_str == "XX"


def test_yXX_X__returns_y_XX():
  rest_str, cut_str = cut_string("yXX", {"X"})
  assert rest_str == "y"
  assert cut_str == "XX"


def test_yYzX_XY__returns_yz_YX():
  rest_str, cut_str = cut_string("yYzX", {"X", "Y"})
  assert rest_str == "yz"
  assert cut_str == "YX"


def test_half_long_vowel__returns_vowel_suprasegmentalia():
  rest_str, cut_str = cut_string("iˑ", {"ˑ"})
  assert rest_str == "i"
  assert cut_str == "ˑ"


def test_extra_short_vowel_unicode__returns_vowel_suprasegmentalia():
  rest_str, cut_str = cut_string("a\u0306", {"\u0306"})
  assert rest_str == "a"
  assert cut_str == "\u0306"


def test_extra_short_vowel__returns_vowel_suprasegmentalia():
  rest_str, cut_str = cut_string("a˘", {"˘"})
  assert rest_str == "a"
  assert cut_str == "˘"


def test_extra_short_vowel_unicode_combined__returns_unchanged():
  #res = unicodedata.decomposition("ă").split(" ")
  #A = "\u0061"
  #BREVE = "\u0306"
  rest_str, cut_str = cut_string("ă", {"\u0306"})
  assert rest_str == "ă"
  assert cut_str == ""
