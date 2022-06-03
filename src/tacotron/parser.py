from pathlib import Path
from typing import Iterator

from speech_dataset_parser import Entry as SDPEntry
from speech_dataset_parser import parse_dataset

from tacotron.typing import Entries, Entry


def __get_entries_from_sdp_entries(audio_dir: Path, entries: Iterator[SDPEntry]) -> Entries:
  result = []
  for entry in entries:
    new_entry = Entry(
        stem=entry.audio_file_abs.relative_to(audio_dir).parent / entry.audio_file_abs.stem,
        basename=entry.audio_file_abs.stem,
        speaker_gender=entry.speaker_gender,
        speaker_name=entry.speaker_name,
        symbols=entry.symbols,
        symbols_language=entry.symbols_language,
        wav_absolute_path=entry.audio_file_abs,
    )
    result.append(new_entry)
  return result


def load_dataset(directory: Path, tier_name: str) -> Entries:
  ds = parse_dataset(directory, tier_name, 16)
  entries = __get_entries_from_sdp_entries(directory, ds)
  return entries
