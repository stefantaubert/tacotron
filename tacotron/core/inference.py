import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from audio_utils.mel import plot_melspec_np
from tacotron.core.synthesizer import InferenceResult, Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import GenericList, plot_alignment_np
from tts_preparation import InferSentence, InferSentenceList


@dataclass
class InferenceEntry():
  sent_id: int = None
  text_original: str = None
  text: str = None
  speaker_id: int = None
  speaker_name: str = None
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


@dataclass
class InferenceEntryOutput():
  sampling_rate: int = None
  mel_img: np.ndarray = None
  postnet_img: np.ndarray = None
  postnet_mel: np.ndarray = None
  alignments_img: np.ndarray = None
  # gate_out_img: np.ndarray = None


def infer(checkpoint: CheckpointTacotron, custom_hparams: Optional[Dict[str, str]], sentence_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, sentences: InferSentenceList, save_callback: Callable[[InferSentence, InferenceEntryOutput], None], max_decoder_steps: int, logger: Logger) -> InferenceEntries:
  model_speakers = checkpoint.get_speakers()

  if full_run:
    sents = sentences
  else:
    sents = InferSentenceList(sentences.get_subset(sentence_ids))

  synth = Synthesizer(
      checkpoint=checkpoint,
      custom_hparams=custom_hparams,
      logger=logger,
  )

  result = InferenceEntries()
  inf_res = synth.infer_all(
    sentences=sents,
    speaker=speaker_name,
    ignore_unknown_symbols=False,
    max_decoder_steps=max_decoder_steps,
  )
  speaker_id = model_speakers.get_id(speaker_name)

  tmp: List[Tuple[InferSentence, InferenceResult]] = zip(sents, inf_res)
  for inf_sent_input, inf_sent_output in tmp:
    symbol_count = len(inf_sent_input.symbols)
    unique_symbols = set(inf_sent_input.symbols)
    unique_symbols_str = " ".join(list(sorted(unique_symbols)))
    unique_symbols_count = len(unique_symbols)
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"
    text = "".join(inf_sent_input.symbols)

    infer_entry_output = InferenceEntry(
      sent_id=inf_sent_input.sent_id,
      text_original=inf_sent_input.original_text,
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
    alignments_img = plot_alignment_np(inf_sent_output.alignments)

    inference_data_output = InferenceEntryOutput(
      postnet_mel=inf_sent_output.mel_outputs_postnet,
      sampling_rate=inf_sent_output.sampling_rate,
      mel_img=mel_img,
      alignments_img=alignments_img,
      postnet_img=postnet_img,
    )

    save_callback(inf_sent_input, inference_data_output)
    result.append(infer_entry_output)

  return result
