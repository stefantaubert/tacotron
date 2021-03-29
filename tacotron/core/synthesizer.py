import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from audio_utils.mel import mel_to_numpy
from tacotron.core.model_symbols import get_model_symbol_ids
from tacotron.core.training import CheckpointTacotron, load_model
from tacotron.globals import DEFAULT_PADDING_ACCENT, DEFAULT_PADDING_SYMBOL
from tacotron.utils import overwrite_custom_hparams, pass_lines
from tts_preparation import InferSentence, InferSentenceList


@dataclass
class InferenceResult():
  sentence: InferSentence
  sampling_rate: int
  reached_max_decoder_steps: bool
  inference_duration_s: float
  mel_outputs: np.ndarray
  mel_outputs_postnet: np.ndarray
  gate_outputs: np.ndarray
  alignments: np.ndarray


class Synthesizer():
  def __init__(self, checkpoint: CheckpointTacotron, custom_hparams: Optional[Dict[str, str]], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    self.accents = checkpoint.get_accents()
    self.symbols = checkpoint.get_symbols()
    self.speakers = checkpoint.get_speakers()
    hparams = checkpoint.get_hparams(logger)
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    model = load_model(
      hparams=hparams,
      state_dict=checkpoint.state_dict,
      logger=logger
    )

    model = model.eval()

    self.hparams = hparams
    self.model = model

  def get_sampling_rate(self) -> int:
    return self.hparams.sampling_rate

  def _get_model_symbols_tensor(self, symbol_ids: List[int], accent_ids: List[int]) -> torch.LongTensor:
    model_symbol_ids = get_model_symbol_ids(
      symbol_ids, accent_ids, self.hparams.n_symbols, self.hparams.accents_use_own_symbols)
    #self._logger.debug(f"Symbol ids:\n{symbol_ids}")
    #self._logger.debug(f"Model symbol ids:\n{model_symbol_ids}")
    symbols_tensor = np.array([model_symbol_ids])
    symbols_tensor = torch.from_numpy(symbols_tensor)
    symbols_tensor = torch.autograd.Variable(symbols_tensor)
    symbols_tensor = symbols_tensor.cuda()
    symbols_tensor = symbols_tensor.long()
    return symbols_tensor

  def infer(self, sentence: InferSentence, speaker: str, ignore_unknown_symbols: bool, max_decoder_steps: int) -> InferenceResult:
    accent_ids = self.accents.get_ids(sentence.accents)
    accents_tensor = np.array([accent_ids])
    accents_tensor = torch.from_numpy(accents_tensor)
    accents_tensor = torch.autograd.Variable(accents_tensor)
    accents_tensor = accents_tensor.cuda()
    accents_tensor = accents_tensor.long()

    symbols = sentence.symbols
    if self.symbols.has_unknown_symbols(symbols):
      if ignore_unknown_symbols:
        symbols = self.symbols.replace_unknown_symbols_with_pad(
         symbols, pad_symbol=DEFAULT_PADDING_SYMBOL)
        self._logger.info(f"After ignoring unknown symbols: {''.join(symbols)}")
      else:
        self._logger.exception("Unknown symbols are not allowed!")
        raise Exception()

    symbol_ids = self.symbols.get_ids(symbols)
    symbols_tensor = self._get_model_symbols_tensor(symbol_ids, accent_ids)

    speaker_id = self.speakers[speaker]
    speaker_tensor = torch.IntTensor([speaker_id])
    speaker_tensor = speaker_tensor.cuda()
    speaker_tensor = speaker_tensor.long()

    start = time.perf_counter()

    with torch.no_grad():
      mel_outputs, mel_outputs_postnet, gate_outputs, alignments, reached_max_decoder_steps = self.model.inference(
        inputs=symbols_tensor,
        accents=accents_tensor,
        speaker_id=speaker_tensor,
        max_decoder_steps=max_decoder_steps,
      )

    end = time.perf_counter()
    inference_duration_s = end - start

    infer_res = InferenceResult(
      sentence=sentence,
      sampling_rate=self.hparams.sampling_rate,
      reached_max_decoder_steps=reached_max_decoder_steps,
      inference_duration_s=inference_duration_s,
      mel_outputs=mel_to_numpy(mel_outputs),
      mel_outputs_postnet=mel_to_numpy(mel_outputs_postnet),
      gate_outputs=mel_to_numpy(gate_outputs),
      alignments=mel_to_numpy(alignments),
    )

    return infer_res

  def infer_all(self, sentences: InferSentenceList, speaker: str, ignore_unknown_symbols: bool, max_decoder_steps: int) -> List[InferenceResult]:
    self._logger.debug(f"Selected speaker: {speaker}")

    result: List[InferenceResult] = []

    accent_id_dict = self.accents

    all_in_one = False

    if all_in_one:
      sentence = sentences.to_sentence(
        space_symbol=" ",
        space_accent=DEFAULT_PADDING_ACCENT,
      )
      self._logger.info(f"\n{sentence.get_formatted(accent_id_dict)}")
      infer_res = self.infer(sentence, speaker, ignore_unknown_symbols, max_decoder_steps)
      result.append(infer_res)
    else:
      # Speed is: 1min inference for 3min wav result
      for sentence in sentences.items(True):
        pass_lines(self._logger.info, sentence.get_formatted(accent_id_dict))
        infer_res = self.infer(sentence, speaker, ignore_unknown_symbols, max_decoder_steps)
        result.append(infer_res)

    return result
