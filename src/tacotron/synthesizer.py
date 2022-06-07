import logging
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, Optional, Set, Tuple, cast

import numpy as np
import torch
from torch import IntTensor, LongTensor  # pylint: disable=no-name-in-module

from tacotron.audio_utils import mel_to_numpy
from tacotron.checkpoint_handling import (CheckpointDict, get_hparams, get_speaker_mapping,
                                          get_stress_mapping, get_symbol_mapping)
from tacotron.dataloader import (get_speaker_mappings_count, get_stress_mappings_count,
                                 get_symbol_mappings_count, split_stresses)
from tacotron.globals import NOT_INFERABLE_SYMBOL_MARKER
from tacotron.model import Tacotron2
from tacotron.training import load_model
from tacotron.typing import Speaker, Symbol, SymbolMapping, Symbols
from tacotron.utils import console_out_len, init_global_seeds, overwrite_custom_hparams, try_copy_to


@dataclass
class InferenceResult():
  sampling_rate: int
  reached_max_decoder_steps: bool
  inference_duration_s: float
  mel_outputs_postnet: np.ndarray
  mel_outputs: np.ndarray
  gate_outputs: np.ndarray
  alignments: np.ndarray


@dataclass
class InferenceResultV2():
  sampling_rate: int
  reached_max_decoder_steps: bool
  inference_duration_s: float
  mel_outputs_postnet: np.ndarray
  mel_outputs: Optional[np.ndarray]
  gate_outputs: Optional[np.ndarray]
  alignments: Optional[np.ndarray]
  unknown_symbols: Set[Symbol]


def get_symbols_noninferable_marked(symbols: Iterable[Symbol], symbol_mapping: SymbolMapping) -> Generator[Symbol, None, None]:
  marker = NOT_INFERABLE_SYMBOL_MARKER
  result = (symbol if symbol in symbol_mapping else marker * console_out_len(symbol)
            for symbol in symbols)
  return result


class Synthesizer():
  def __init__(self, checkpoint: CheckpointDict, custom_hparams: Optional[Dict[str, str]], device: torch.device, logger: logging.Logger):
    super().__init__()
    self._logger = logger

    hparams = get_hparams(checkpoint)
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    self.symbol_mapping = get_symbol_mapping(checkpoint)
    n_symbols = get_symbol_mappings_count(self.symbol_mapping)

    self.stress_mapping = None
    n_stresses = None
    if hparams.use_stress_embedding:
      self.stress_mapping = get_stress_mapping(checkpoint)
      n_stresses = get_stress_mappings_count(self.stress_mapping)

    self.speaker_mapping = None
    n_speakers = None
    if hparams.use_speaker_embedding:
      self.speaker_mapping = get_speaker_mapping(checkpoint)
      n_speakers = get_speaker_mappings_count(self.speaker_mapping)

    model = load_model(
        hparams=hparams,
        checkpoint=checkpoint,
        n_speakers=n_speakers,
        n_stresses=n_stresses,
        n_symbols=n_symbols,
    )

    self.device = device

    model = cast(Tacotron2, try_copy_to(model, device))
    model = model.eval()

    self.hparams = hparams
    self.model = model

  def get_sampling_rate(self) -> int:
    return self.hparams.sampling_rate

  def infer(self, symbols: Tuple[str], speaker: Optional[Speaker], max_decoder_steps: int, seed: int) -> InferenceResultV2:
    marker = NOT_INFERABLE_SYMBOL_MARKER
    if self.hparams.use_stress_embedding:

      symbols_wo_stress, stresses = split_stresses(
          symbols, self.hparams.symbols_are_ipa)

      mappable_entries = tuple(
          symbol in self.symbol_mapping and stress in self.stress_mapping
          for symbol, stress in zip(symbols_wo_stress, stresses)
      )

      mapped_symbols = (
          self.symbol_mapping[symbol]
          for symbol, is_mappable in zip(symbols_wo_stress, mappable_entries)
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
          for symbol, stress, is_mappable in zip(symbols_wo_stress, stresses, mappable_entries)
      )

      self._logger.info(print_text)
    else:
      mapped_symbols = (
          self.symbol_mapping[symbol]
          for symbol in symbols
          if symbol in self.symbol_mapping
      )

      print_text = ' '.join(
          f"{symbol}" if symbol in self.symbol_mapping
          else marker * console_out_len(symbol)
          for symbol in symbols
      )

      self._logger.info(print_text)

    init_global_seeds(seed)

    symbol_tensor = IntTensor([list(mapped_symbols)])
    symbol_tensor = try_copy_to(symbol_tensor, self.device)

    stress_tensor = None
    if self.hparams.use_stress_embedding:
      stress_tensor = LongTensor([list(mapped_stresses)])
      stress_tensor = try_copy_to(stress_tensor, self.device)

    speaker_tensor = None
    if self.hparams.use_speaker_embedding:
      assert speaker is not None
      assert speaker in self.speaker_mapping
      mapped_speaker = self.speaker_mapping[speaker]

      speaker_tensor = IntTensor(
          symbol_tensor.size(0), symbol_tensor.size(1))
      torch.nn.init.constant_(speaker_tensor, mapped_speaker)
      speaker_tensor = try_copy_to(speaker_tensor, self.device)

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
        sampling_rate=self.hparams.sampling_rate,
        reached_max_decoder_steps=reached_max_decoder_steps,
        inference_duration_s=inference_duration_s,
        mel_outputs=mel_to_numpy(mel_outputs),
        mel_outputs_postnet=mel_to_numpy(mel_outputs_postnet),
        gate_outputs=mel_to_numpy(gate_outputs),
        alignments=mel_to_numpy(alignments),
    )

    return infer_res

  def infer_v2(self, symbols: Symbols, speaker: Speaker, max_decoder_steps: int, seed: int, include_stats: bool) -> InferenceResultV2:
    marker = NOT_INFERABLE_SYMBOL_MARKER

    if self.hparams.use_stress_embedding:

      symbols, stresses = split_stresses(
          symbols, self.hparams.symbols_are_ipa)

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
    else:
      mapped_symbols = (
          self.symbol_mapping[symbol]
          for symbol in symbols
          if symbol in self.symbol_mapping
      )

      print_text = ' '.join(
          f"{symbol}" if symbol in self.symbol_mapping
          else marker * console_out_len(symbol)
          for symbol in symbols
      )

    self._logger.info(print_text)

    non_mappable_symbols = set(
        symbol for symbol in symbols if symbol not in self.symbol_mapping)
    if len(non_mappable_symbols) > 0:
      self._logger.warn(
          f"Unknown symbols: {' '.join(sorted(non_mappable_symbols))}")

    init_global_seeds(seed)

    symbol_tensor = IntTensor([list(mapped_symbols)])
    symbol_tensor = try_copy_to(symbol_tensor, self.device)

    stress_tensor = None
    if self.hparams.use_stress_embedding:
      stress_tensor = LongTensor([list(mapped_stresses)])
      stress_tensor = try_copy_to(stress_tensor, self.device)

    speaker_tensor = None
    if self.hparams.use_speaker_embedding:
      assert speaker is not None
      assert speaker in self.speaker_mapping
      mapped_speaker = self.speaker_mapping[speaker]

      speaker_tensor = IntTensor(
          symbol_tensor.size(0), symbol_tensor.size(1))
      torch.nn.init.constant_(speaker_tensor, mapped_speaker)
      speaker_tensor = try_copy_to(speaker_tensor, self.device)

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

    infer_res = InferenceResultV2(
        sampling_rate=self.hparams.sampling_rate,
        reached_max_decoder_steps=reached_max_decoder_steps,
        inference_duration_s=inference_duration_s,
        mel_outputs_postnet=mel_to_numpy(mel_outputs_postnet),
        mel_outputs=None,
        gate_outputs=None,
        alignments=None,
        unknown_symbols=non_mappable_symbols,
    )

    if include_stats:
      infer_res.mel_outputs = mel_to_numpy(mel_outputs)
      infer_res.gate_outputs = mel_to_numpy(gate_outputs)
      infer_res.alignments = mel_to_numpy(alignments)

    return infer_res
