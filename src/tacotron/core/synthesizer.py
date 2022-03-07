import logging
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, Optional, cast

import numpy as np
from audio_utils.mel import mel_to_numpy
from general_utils import overwrite_custom_hparams
from torch import IntTensor, LongTensor  # pylint: disable=no-name-in-module
import torch
from tacotron.core.checkpoint_handling import CheckpointDict, get_hparams, get_speaker_mapping, get_stress_mapping, get_symbol_mapping
from tacotron.core.dataloader import split_stresses
from tacotron.core.model import Tacotron2
from tacotron.core.training import load_model
from tacotron.core.typing import SymbolMapping
from tacotron.globals import NOT_INFERABLE_SYMBOL_MARKER
from tacotron.utils import init_global_seeds, try_copy_to_gpu
from text_utils import Speaker, Symbol
from tts_preparation import InferableUtterance
from general_utils import console_out_len


@dataclass
class InferenceResult():
  utterance: InferableUtterance
  sampling_rate: int
  reached_max_decoder_steps: bool
  inference_duration_s: float
  mel_outputs: np.ndarray
  mel_outputs_postnet: np.ndarray
  gate_outputs: np.ndarray
  alignments: np.ndarray


def get_symbols_noninferable_marked(symbols: Iterable[Symbol], symbol_mapping: SymbolMapping) -> Generator[Symbol, None, None]:
  marker = NOT_INFERABLE_SYMBOL_MARKER
  result = (symbol if symbol in symbol_mapping else marker * console_out_len(symbol)
            for symbol in symbols)
  return result


class Synthesizer():
  def __init__(self, checkpoint: CheckpointDict, custom_hparams: Optional[Dict[str, str]], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    hparams = get_hparams(checkpoint)
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    self.symbol_mapping = get_symbol_mapping(checkpoint)
    n_symbols = len(self.symbol_mapping)

    self.stress_mapping = None
    n_stresses = None
    if hparams.use_stress_embedding:
      self.stress_mapping = get_stress_mapping(checkpoint)
      n_stresses = len(self.stress_mapping)

    self.speaker_mapping = None
    n_speakers = None
    if hparams.use_speaker_embedding:
      self.speaker_mapping = get_speaker_mapping(checkpoint)
      n_speakers = len(self.speaker_mapping)

    model = load_model(
      hparams=hparams,
      checkpoint=checkpoint,
      n_speakers=n_speakers,
      n_stresses=n_stresses,
      n_symbols=n_symbols,
    )

    model = cast(Tacotron2, try_copy_to_gpu(model))
    model = model.eval()

    self.hparams = hparams
    self.model = model

  def get_sampling_rate(self) -> int:
    return self.hparams.sampling_rate

  def infer(self, utterance: InferableUtterance, speaker: Optional[Speaker], max_decoder_steps: int, seed: int) -> InferenceResult:
    marker = NOT_INFERABLE_SYMBOL_MARKER
    if self.hparams.use_stress_embedding:

      symbols, stresses = split_stresses(utterance.symbols, self.hparams.symbols_are_ipa)

      mappable_entries = tuple(
        symbol in self.symbol_mapping and stress in self.stress_mapping
        for symbol, stress in zip(symbols, stresses)
      )

      mapped_symbols = (
        self.symbol_mapping[symbol]
        for symbol, is_mappable in zip(symbols, mappable_entries)
        if is_mappable
      )

      mapped_stresses = (
        self.stress_mapping[stress]
        for stress, is_mappable in zip(stresses, mappable_entries)
        if is_mappable
      )

      print_text = ' '.join(
        f"{symbol}{stress}" if is_mappable
        else marker * (console_out_len(symbol) + console_out_len(stress))
        for symbol, stress, is_mappable in zip(symbols, stresses, mappable_entries)
      )

      self._logger.info(print_text)
    else:
      mapped_symbols = (
        self.symbol_mapping[symbol]
        for symbol in utterance.symbols
        if symbol in self.symbol_mapping
      )

      print_text = ' '.join(
        f"{symbol}" if symbol in self.symbol_mapping
        else marker * console_out_len(symbol)
        for symbol in zip(utterance.symbols)
      )

      self._logger.info(print_text)

    init_global_seeds(seed)

    symbol_tensor = IntTensor([list(mapped_symbols)])
    symbol_tensor = try_copy_to_gpu(symbol_tensor)

    stress_tensor = None
    if self.hparams.use_stress_embedding:
      stress_tensor = LongTensor([list(mapped_stresses)])
      stress_tensor = try_copy_to_gpu(stress_tensor)

    speaker_tensor = None
    if self.hparams.use_speaker_embedding:
      assert speaker is not None
      assert speaker in self.speaker_mapping
      mapped_speaker = self.speaker_mapping[speaker]

      speaker_tensor = IntTensor(symbol_tensor.size(0), symbol_tensor.size(1))
      torch.nn.init.constant_(speaker_tensor, mapped_speaker)
      speaker_tensor = try_copy_to_gpu(speaker_tensor)

    start = time.perf_counter()

    with torch.no_grad():
      mel_outputs, mel_outputs_postnet, gate_outputs, alignments, reached_max_decoder_steps = self.model.inference(
        symbols=symbol_tensor,
        speakers=speaker_tensor,
        stresses=stress_tensor,
        max_decoder_steps=max_decoder_steps,
      )

    end = time.perf_counter()
    inference_duration_s = end - start

    infer_res = InferenceResult(
      utterance=utterance,
      sampling_rate=self.hparams.sampling_rate,
      reached_max_decoder_steps=reached_max_decoder_steps,
      inference_duration_s=inference_duration_s,
      mel_outputs=mel_to_numpy(mel_outputs),
      mel_outputs_postnet=mel_to_numpy(mel_outputs_postnet),
      gate_outputs=mel_to_numpy(gate_outputs),
      alignments=mel_to_numpy(alignments),
    )

    return infer_res
