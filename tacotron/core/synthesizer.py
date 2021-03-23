import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from src.core.common.globals import PADDING_SYMBOL
from src.core.common.train import overwrite_custom_hparams
from tacotron.core.model_symbols import get_model_symbol_ids
from tacotron.core.training import CheckpointTacotron, load_model


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

  def infer(self, symbols: List[str], accents: List[str], speaker: str, allow_unknown: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.symbols.has_unknown_symbols(symbols):
      if allow_unknown:
        symbols = self.symbols.replace_unknown_symbols_with_pad(symbols, pad_symbol=PADDING_SYMBOL)
      else:
        self._logger.exception("Unknown symbols are not allowed!")
        raise Exception()

    accent_ids = self.accents.get_ids(accents)
    accents_tensor = np.array([accent_ids])
    accents_tensor = torch.from_numpy(accents_tensor)
    accents_tensor = torch.autograd.Variable(accents_tensor)
    accents_tensor = accents_tensor.cuda()
    accents_tensor = accents_tensor.long()

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
        speaker_id=speaker_tensor
      )

    end = time.perf_counter()
    inference_duration_s = end - start
    stats = reached_max_decoder_steps, inference_duration_s
    outputs = mel_outputs, mel_outputs_postnet, gate_outputs, alignments
    return outputs, stats
