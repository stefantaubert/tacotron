from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

EntryId = int
Symbol = str
Symbols = Tuple[Symbol, ...]
Stress = str
Stresses = Tuple[Stress, ...]
Speaker = str
SpeakerId = str
SpeakerMapping = OrderedDictType[Speaker, int]
SymbolMapping = OrderedDictType[Symbol, int]
StressMapping = OrderedDictType[Stress, int]

SymbolToSymbolMapping = Dict[Symbol, Symbol]


@dataclass()
class Entry():
  stem: str
  basename: str
  speaker_name: Speaker
  speaker_gender: int
  symbols_language: str
  symbols: Tuple[str]
  wav_absolute_path: Path
  #wav_duration: float
  #wav_sampling_rate: int
  #mel_absolute_path: Path
  #mel_n_channels: int


Entries = List[Entry]
