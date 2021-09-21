import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Optional, Set

import numpy as np
from audio_utils.mel import plot_melspec_np
from general_utils import GenericList
from pandas.core.frame import DataFrame
from tacotron.core.synthesizer import Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import plot_alignment_np_new
from text_utils.types import Speaker, SpeakerId, Symbols
from tts_preparation import InferableUtterance, InferableUtterances


@dataclass
class InferenceEntry():
  utterance: InferableUtterance
  speaker_id: SpeakerId
  speaker_name: Speaker
  iteration: int
  reached_max_decoder_steps: bool
  inference_duration_s: float
  timepoint: datetime.datetime
  train_name: str
  sampling_rate: int
  seed: int

  @property
  def unique_symbols(self) -> Symbols:
    return set(self.utterance.symbols)

  @property
  def symbols_count(self) -> int:
    return len(self.utterance.symbols)

  @property
  def unique_symbols_count(self) -> int:
    return len(self.unique_symbols)


class InferenceEntries(GenericList[InferenceEntry]):
  pass


def get_df(entries: InferenceEntries) -> DataFrame:
  if len(entries) == 0:
    return DataFrame()

  data = [{
    "Id": entry.utterance.utterance_id,
    "Timepoint": f"{entry.timepoint:%Y/%m/%d %H:%M:%S}",
    "Iteration": entry.iteration,
    "Language": repr(entry.utterance.language),
    "Symbols": ''.join(entry.utterance.symbols),
    "Symbols format": repr(entry.utterance.symbols_format),
    "Speaker": entry.speaker_name,
    "Speaker Id": entry.speaker_id,
    "Inference duration (s)": entry.inference_duration_s,
    "Reached max. steps": entry.reached_max_decoder_steps,
    "Train name": entry.train_name,
    "Sampling rate (Hz)": entry.sampling_rate,
    "Seed": entry.seed,
    "# Symbols": entry.symbols_count,
    "Unique symbols": ' '.join(sorted(entry.unique_symbols)),
    "# Unique symbols": entry.unique_symbols_count,
   } for entry in entries.items()]

  df = DataFrame(
    data=[x.values() for x in data],
    columns=data[0].keys(),
  )

  return df


@dataclass
class InferenceEntryOutput():
  sampling_rate: int = None
  mel_img: np.ndarray = None
  postnet_img: np.ndarray = None
  postnet_mel: np.ndarray = None
  alignments_img: np.ndarray = None
  # gate_out_img: np.ndarray = None


def get_subset(utterances: InferableUtterances, utterance_ids: Set[int]) -> InferableUtterances:
  result = InferableUtterances(utterance for utterance in utterances.items()
                               if utterance.utterance_id in utterance_ids)
  return result


def infer(checkpoint: CheckpointTacotron, custom_hparams: Optional[Dict[str, str]], speaker_name: Optional[Speaker], train_name: str, utterances: InferableUtterances, utterance_ids: Optional[Set[int]], full_run: bool, save_callback: Callable[[InferableUtterance, InferenceEntryOutput], None], max_decoder_steps: int, seed: int, logger: Logger) -> InferenceEntries:
  model_speakers = checkpoint.get_speakers()

  if full_run:
    pass
  elif utterance_ids is not None:
    utterances = InferableUtterances(get_subset(utterances, utterance_ids))
  else:
    assert seed is not None
    utterances = InferableUtterances([utterances.get_random_entry(seed)])
  synth = Synthesizer(
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger,
  )

  speaker_id = model_speakers.get_id(speaker_name)

  result = InferenceEntries()
  for utterance in utterances.items():
    timepoint = datetime.datetime.now()
    inf_sent_output = synth.infer(
      utterance=utterance,
      speaker=speaker_name,
      max_decoder_steps=max_decoder_steps,
      seed=seed,
    )

    infer_entry_output = InferenceEntry(
      utterance=utterance,
      speaker_id=speaker_id,
      speaker_name=speaker_name,
      iteration=checkpoint.iteration,
      timepoint=timepoint,
      train_name=train_name,
      sampling_rate=inf_sent_output.sampling_rate,
      inference_duration_s=inf_sent_output.inference_duration_s,
      reached_max_decoder_steps=inf_sent_output.reached_max_decoder_steps,
      seed=seed,
    )

    _, mel_img = plot_melspec_np(inf_sent_output.mel_outputs)
    _, postnet_img = plot_melspec_np(inf_sent_output.mel_outputs_postnet)
    _, alignments_img = plot_alignment_np_new(inf_sent_output.alignments)

    inference_data_output = InferenceEntryOutput(
      postnet_mel=inf_sent_output.mel_outputs_postnet,
      sampling_rate=inf_sent_output.sampling_rate,
      mel_img=mel_img,
      alignments_img=alignments_img,
      postnet_img=postnet_img,
    )

    save_callback(utterance, inference_data_output)
    result.append(infer_entry_output)

  return result
