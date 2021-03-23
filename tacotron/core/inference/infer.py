import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from audio_utils import get_duration_s
from audio_utils.mel import TacotronSTFT, plot_melspec_np
from tacotron.core.inference.synthesizer import InferenceResult, Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import (GenericList, get_shard_size, plot_alignment_np,
                            symbol_count_to_shards)
from text_utils.utils import deserialize_list, serialize_list
from tts_preparation import InferSentence, InferSentenceList


@dataclass
class InferenceEntry():
  sent_id: int = None
  text_original: str = None
  text: str = None
  inferred_duration_s: float = None
  speaker_id: int = None
  speaker_name: str = None
  wg_iteration: int = None
  taco_iteration: int = None
  reached_max_decoder_steps: bool = None
  taco_inference_duration_s: float = None
  wg_inference_duration_s: float = None
  total_inference_duration_s: float = None
  shards: float = None
  unique_symbol_ids: str = None
  unique_symbol_ids_count: int = None
  symbol_count: int = None
  timepoint: str = None
  train_name: str = None
  sampling_rate: int = None


class InferenceEntries(GenericList[InferenceEntry]):
  pass


@dataclass
class InferenceEntryOutput():
  inferred_wav: np.ndarray = None
  inferred_wav_sr: int = None
  inferred_wav_img: np.ndarray = None
  alignments_img: np.ndarray = None
  postnet_img: np.ndarray = None
  # gate_out_img: np.ndarray = None


def infer2(tacotron_checkpoint: CheckpointTacotron, waveglow_checkpoint: CheckpointWaveglow, denoiser_strength: float, sigma: float, custom_taco_hparams: Optional[Dict[str, str]], custom_wg_hparams: Optional[Dict[str, str]], sentence_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, sentences: InferSentenceList, sentence_pause_s: float, save_callback: Callable[[InferSentence, InferenceEntryOutput], None], logger: Logger) -> Tuple[np.ndarray, InferenceEntries]:
  model_symbols = tacotron_checkpoint.get_symbols()
  model_speakers = tacotron_checkpoint.get_speakers()

  if full_run:
    sents = sentences
  else:
    sents = InferSentenceList(sentences.get_subset(sentence_ids))

  synth = Synthesizer(
    tacotron_checkpoint,
    waveglow_checkpoint,
    logger=logger,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )
  shard_size = get_shard_size(model_symbols)

  result = InferenceEntries()
  res_wav, inf_res = synth.infer(
    sentences=sents,
    speaker=speaker_name,
    sigma=sigma, denoiser_strength=denoiser_strength,
    sentence_pause_s=sentence_pause_s
  )
  speaker_id = model_speakers.get_id(speaker_name)

  tmp: List[Tuple[InferSentence, InferenceResult]] = zip(sents, inf_res)
  for inf_sent_input, inf_sent_output in tmp:
    symbol_count = len(inf_sent_input.symbols)
    shards = symbol_count_to_shards(
      symbol_count=symbol_count,
      shard_size=shard_size,
    )

    unique_symbols_ids = set(inf_sent_input.symbols)
    unique_symbols_ids_str = " ".join(list(sorted(unique_symbols_ids)))
    unique_symbols_ids_count = len(unique_symbols_ids)
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"
    text = "".join(inf_sent_input.symbols)
    infer_entry_output = InferenceEntry(
      sent_id=inf_sent_input.sent_id,
      text_original=inf_sent_input.original_text,
      text=text,
      speaker_id=speaker_id,
      speaker_name=speaker_name,
      wg_iteration=waveglow_checkpoint.iteration,
      taco_iteration=tacotron_checkpoint.iteration,
      shards=shards,
      unique_symbol_ids=unique_symbols_ids_str,
      unique_symbol_ids_count=unique_symbols_ids_count,
      symbol_count=symbol_count,
      timepoint=timepoint,
      train_name=train_name,
      sampling_rate=synth.get_sampling_rate(),
    )

    inference_data_output = InferenceEntryOutput()
    inference_data_output.inferred_wav = inf_sent_output.wav
    inference_data_output.inferred_wav_sr = inf_sent_output.sampling_rate

    infer_entry_output.reached_max_decoder_steps = inf_sent_output.reached_max_decoder_steps
    infer_entry_output.wg_inference_duration_s = inf_sent_output.wg_inference_duration_s
    infer_entry_output.taco_inference_duration_s = inf_sent_output.taco_inference_duration_s
    infer_entry_output.total_inference_duration_s = inf_sent_output.wg_inference_duration_s + \
        inf_sent_output.taco_inference_duration_s
    duration_s = get_duration_s(inf_sent_output.wav, inf_sent_output.sampling_rate)
    infer_entry_output.inferred_duration_s = duration_s

    _, inferred_mel_plot = plot_melspec_np(inf_sent_output.mel_outputs)
    inference_data_output.inferred_wav_img = inferred_mel_plot

    alignments_img = plot_alignment_np(inf_sent_output.alignments)
    inference_data_output.alignments_img = alignments_img
    _, post_mel_img = plot_melspec_np(inf_sent_output.mel_outputs_postnet)
    inference_data_output.postnet_img = post_mel_img
    save_callback(inf_sent_input, inference_data_output)
    result.append(infer_entry_output)
  return res_wav, result


def infer(tacotron_checkpoint: CheckpointTacotron, waveglow_checkpoint: CheckpointWaveglow, speaker: str, sentence_pause_s: float, sigma: float, denoiser_strength: float, sentences: InferSentenceList, custom_taco_hparams: Optional[Dict[str, str]], custom_wg_hparams: Optional[Dict[str, str]], logger: Logger) -> Tuple[np.ndarray, List[InferenceResult]]:
  synth = Synthesizer(
    tacotron_checkpoint,
    waveglow_checkpoint,
    logger=logger,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )

  return synth.infer(
    sentences=sentences,
    speaker=speaker,
    denoiser_strength=denoiser_strength,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
  )
