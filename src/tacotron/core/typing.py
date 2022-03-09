from typing import Dict, Tuple
from typing import OrderedDict as OrderedDictType
from text_utils import Speaker, Symbol

Stress = str
Stresses = Tuple[Stress, ...]
SpeakerMapping = OrderedDictType[Speaker, int]
SymbolMapping = OrderedDictType[Symbol, int]
StressMapping = OrderedDictType[Stress, int]

SymbolToSymbolMapping = Dict[Symbol, Symbol]
