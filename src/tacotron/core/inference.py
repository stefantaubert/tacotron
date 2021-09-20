import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Optional, Set
from general_utils import GenericList

import numpy as np
import pandas as pd
from audio_utils.mel import plot_melspec_np
from tacotron.core.synthesizer import Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import plot_alignment_np_new
from text_utils.types import Speaker, SpeakerId
from tts_preparation import InferableUtterance, InferableUtterances


@dataclass
class InferenceEntry():
  utterance_id: int = None
  text: str = None
  speaker_id: SpeakerId = None
  speaker_name: Speaker = None
  iteration: int = None
  reached_max_decoder_steps: bool = None
  inference_duration_s: float = None
  unique_symbols: str = None
  unique_symbols_count: int = None
  symbol_count: int = None
  timepoint: str = None
  train_name: str = None
  sampling_rate: int = None


class InferenceEntries(GenericList[InferenceEntry]):
  pass


def get_df(entries: InferenceEntries) -> pd.DataFrame:
  # TODO
  pass


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
    inf_sent_output = synth.infer(
      utterance=utterance,
      speaker=speaker_name,
      max_decoder_steps=max_decoder_steps,
      seed=seed,
    )

    symbol_count = len(utterance.symbols)
    unique_symbols = set(utterance.symbols)
    unique_symbols_str = " ".join(sorted(unique_symbols))
    unique_symbols_count = len(unique_symbols)
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"
    text = "".join(utterance.symbols)

    infer_entry_output = InferenceEntry(
      utterance_id=utterance.utterance_id,
      text=text,
      speaker_id=speaker_id,
      speaker_name=speaker_name,
      iteration=checkpoint.iteration,
      unique_symbols=unique_symbols_str,
      unique_symbols_count=unique_symbols_count,
      symbol_count=symbol_count,
      timepoint=timepoint,
      train_name=train_name,
      sampling_rate=inf_sent_output.sampling_rate,
      inference_duration_s=inf_sent_output.inference_duration_s,
      reached_max_decoder_steps=inf_sent_output.reached_max_decoder_steps,
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
