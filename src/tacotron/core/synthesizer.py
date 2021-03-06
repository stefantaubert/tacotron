import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from audio_utils.mel import mel_to_numpy
from general_utils import overwrite_custom_hparams
from tacotron.core.training import CheckpointTacotron, load_model
from tacotron.globals import NOT_INFERABLE_SYMBOL_MARKER
from tacotron.utils import init_global_seeds
from text_utils import Speaker
from tts_preparation import (InferableUtterance, InferableUtterances,
                             log_utterances)


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


class Synthesizer():
  def __init__(self, checkpoint: CheckpointTacotron, custom_hparams: Optional[Dict[str, str]], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    self.symbol_id_dict = checkpoint.get_symbols()
    self.speaker_id_dict = checkpoint.get_speakers()
    hparams = checkpoint.get_hparams(logger)
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    model = load_model(
      hparams=hparams,
      state_dict=checkpoint.model_state_dict,
      logger=logger
    )

    model = model.eval()

    self.hparams = hparams
    self.model = model

  def get_sampling_rate(self) -> int:
    return self.hparams.sampling_rate

  def infer(self, utterance: InferableUtterance, speaker: Speaker, max_decoder_steps: int, seed: int) -> InferenceResult:
    same_symbol_id_dict_was_used = self.symbol_id_dict.get_ids(
      utterance.symbols) == utterance.symbol_ids
    assert same_symbol_id_dict_was_used

    log_utterances(InferableUtterances([utterance]), marker=NOT_INFERABLE_SYMBOL_MARKER)
    init_global_seeds(seed)

    inferable_symbol_ids = [
      symbol_id for symbol_id in utterance.symbol_ids if symbol_id is not None]
    symbols_tensor = np.array([inferable_symbol_ids])
    symbols_tensor = torch.from_numpy(symbols_tensor)
    symbols_tensor = torch.autograd.Variable(symbols_tensor)
    symbols_tensor = symbols_tensor.cuda()
    symbols_tensor = symbols_tensor.long()

    speaker_id = self.speaker_id_dict[speaker]
    speaker_tensor = torch.IntTensor([speaker_id])
    speaker_tensor = speaker_tensor.cuda()
    speaker_tensor = speaker_tensor.long()

    start = time.perf_counter()

    with torch.no_grad():
      mel_outputs, mel_outputs_postnet, gate_outputs, alignments, reached_max_decoder_steps = self.model.inference(
        inputs=symbols_tensor,
        speaker_id=speaker_tensor,
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
