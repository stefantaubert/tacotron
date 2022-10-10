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
Tone = str
Tones = Tuple[Tone, ...]
Duration = str
Durations = Tuple[Duration, ...]
Speaker = str
SpeakerId = int
MappingId = int
Mapping = OrderedDictType[str, MappingId]
SpeakerMapping = OrderedDictType[Speaker, MappingId]
SymbolMapping = OrderedDictType[Symbol, MappingId]
StressMapping = OrderedDictType[Stress, MappingId]
ToneMapping = OrderedDictType[Tone, MappingId]
DurationMapping = OrderedDictType[Duration, MappingId]

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
