

from typing import Iterator
from speech_dataset_parser_api import Entry as SDPEntry
from tacotron.core.typing import Entry, Entries


def get_entries_from_sdp_entries(entries: Iterator[SDPEntry]) -> Entries:
    result = []
    for entry in entries:
        new_entry = Entry(
            stem=entry.audio_file_rel.parent / entry.audio_file_rel.stem,
            basename=entry.audio_file_abs.stem,
            speaker_gender=entry.speaker_gender,
            speaker_name=entry.speaker_name,
            symbols=entry.symbols,
            symbols_language=entry.symbols_language,
            wav_absolute_path=entry.audio_file_abs,
        )
        result.append(new_entry)
    return result
