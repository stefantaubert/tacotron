from text_utils import Symbol

from tacotron.globals import SPACE_DISPLAYABLE


def get_symbol_printable(symbol: Symbol) -> str:
  result = symbol.replace(" ", SPACE_DISPLAYABLE)
  result = repr(result)[1:-1]
  return result