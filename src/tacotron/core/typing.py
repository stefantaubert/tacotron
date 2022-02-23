
from collections import OrderedDict
from dataclasses import asdict, dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional
from typing import OrderedDict as OrderedDictType
from text_utils import Speaker, SpeakersDict, Symbol, SymbolIdDict

Stress = str

SpeakerMapping = OrderedDictType[Speaker, int]
SymbolMapping = OrderedDictType[Symbol, int]
StressMapping = OrderedDictType[Stress, int]
