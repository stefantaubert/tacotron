from itertools import chain
from logging import Logger
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import torch
from audio_utils.mel import TacotronSTFT
from tacotron.core.hparams import HParams
from tacotron.utils import to_gpu
from text_utils import SpeakerId, Symbol, Symbols
from text_utils.pronunciation.arpa_symbols import VOWELS_WITH_STRESSES
from torch import (FloatTensor, IntTensor,  # pylint: disable=no-name-in-module
                   LongTensor, Tensor)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tts_preparation import PreparedDataList

DEFAULT_NA_STRESS = "-"


def split_stress_arpa(symbol: Symbol) -> Tuple[Symbol, Optional[str]]:
  has_stress_mark = symbol in VOWELS_WITH_STRESSES
  if has_stress_mark:
    vowel = symbol[0:-1]
    stress = symbol[-1]
    return vowel, stress
  return symbol, DEFAULT_NA_STRESS


# def split_stresses_arpa(symbols: Symbols, na_stress: str) -> Generator[Tuple[Symbol, str], None, None]:
#   for symbol in symbols:
#     yield split_stress_arpa(symbol, na_stress)


def split_stresses_arpa(symbols: Symbols) -> Tuple[Symbols, Tuple[str, ...]]:
  res_symbols = []
  stresses = []
  for symbol in symbols:
    symbol_core, stress = split_stress_arpa(symbol)
    res_symbols.append(symbol_core)
    stresses.append(stress)
  return tuple(res_symbols), tuple(stresses)


PADDING_SHIFT = 1


def get_symbols_dict(valset: PreparedDataList, trainset: PreparedDataList) -> Dict[str, int]:
  all_valsymbols = (entry.symbols for entry in valset.items())
  all_trainsymbols = (entry.symbols for entry in trainset.items())
  all_symbols = chain(all_valsymbols, all_trainsymbols)

  unique_symbols = {symbol for symbols in all_symbols for symbol in symbols}

  symbol_ids = {
    symbol: symbol_nr + PADDING_SHIFT
    for symbol, symbol_nr in zip(sorted(unique_symbols), range(len(unique_symbols)))
  }
  return symbol_ids


def get_symbols_stresses_dicts(valset: PreparedDataList, trainset: PreparedDataList) -> Tuple[Dict[Symbol, int], Dict[str, int]]:
  all_valsymbols = (entry.symbols for entry in valset.items())
  all_trainsymbols = (entry.symbols for entry in trainset.items())
  all_symbols = chain(all_valsymbols, all_trainsymbols)
  all_symbols_stress_splitted = (split_stresses_arpa(symbols) for symbols in all_symbols)

  all_symbols, all_stresses = zip(*all_symbols_stress_splitted)
  unique_symbols = {symbol for symbols in all_symbols for symbol in symbols}
  unique_stresses = {stress for stresses in all_stresses for stress in stresses}

  symbol_ids = {
    symbol: symbol_nr + PADDING_SHIFT
    for symbol, symbol_nr in zip(sorted(unique_symbols), range(len(unique_symbols)))
  }

  stress_ids = {
    stress: stress_nr + PADDING_SHIFT
    for stress, stress_nr in zip(sorted(unique_stresses), range(len(unique_stresses)))
  }

  return symbol_ids, stress_ids


LoaderEntry = Tuple[IntTensor, Tensor, SpeakerId, Optional[IntTensor]]


class SymbolsMelLoader(Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """

  def __init__(self, data: PreparedDataList, hparams: HParams, symbols_dict: Dict[Symbol, int], stress_dict: Optional[Dict[str, int]], logger: Logger):
    # random.seed(hparams.seed)
    # random.shuffle(data)
    self.use_saved_mels = hparams.use_saved_mels
    if not hparams.use_saved_mels:
      self.mel_parser = TacotronSTFT(hparams, logger)

    logger.info("Reading files...")

    self.data: Dict[int, Tuple[IntTensor, Path, SpeakerId, Optional[IntTensor]]] = {}

    # for i, entry, symbols in enumerate(zip(data.items(True), )

    for i, entry in enumerate(data.items(True)):
      symbols = entry.symbols

      stress_tensor = None
      if hparams.use_stress_embedding:
        symbols, stresses = split_stresses_arpa(symbols)
        stress_ids = (stress_dict[stress] for stress in stresses)
        stress_tensor = IntTensor(list(stress_ids))

      symbol_ids = (symbols_dict[symbol] for symbol in symbols)
      symbols_tensor = IntTensor(list(symbol_ids))

      if hparams.use_saved_mels:
        self.data[i] = (symbols_tensor, entry.mel_absolute_path, entry.speaker_id, stress_tensor)
      else:
        self.data[i] = (symbols_tensor, entry.wav_absolute_path, entry.speaker_id, stress_tensor)

    if hparams.use_saved_mels and hparams.cache_mels:
      logger.info("Loading mels into memory...")
      self.cache: Dict[int, Tensor] = {}
      vals: tuple
      for i, vals in tqdm(self.data.items()):
        mel_tensor = torch.load(vals[1], map_location='cpu')
        self.cache[i] = mel_tensor
    self.use_cache: bool = hparams.cache_mels

  def __getitem__(self, index: int) -> LoaderEntry:
    # return self.cache[index]
    # debug_logger.debug(f"getitem called {index}")
    symbols_tensor, path, speaker_id, stress_tensor = self.data[index]
    if self.use_saved_mels:
      if self.use_cache:
        mel_tensor = self.cache[index].clone().detach()
      else:
        mel_tensor: Tensor = torch.load(path, map_location='cpu')
    else:
      mel_tensor = self.mel_parser.get_mel_tensor_from_file(path)

    symbols_tensor_cloned = symbols_tensor.clone().detach()
    stress_tensor_cloned = None
    if stress_tensor is not None:
      stress_tensor_cloned = stress_tensor.clone().detach()
    # debug_logger.debug(f"getitem finished {index}")
    return symbols_tensor_cloned, mel_tensor, speaker_id, stress_tensor_cloned

  def __len__(self):
    return len(self.data)


Batch = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor,
              torch.FloatTensor, torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]]


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """

  def __init__(self, n_frames_per_step: int, padding_symbol_id: int, use_stress: bool):
    self.n_frames_per_step = n_frames_per_step
    self.padding_symbol_id = padding_symbol_id
    self.use_stress = use_stress

  def __call__(self, batch: List[LoaderEntry]) -> Batch:
    """Collate's training batch from normalized text and mel-spectrogram
    PARAMS
    ------
    batch: [text_normalized, mel_normalized]
    """
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
      LongTensor([len(symbols_tensor) for symbols_tensor, _, _, _ in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]

    symbols_padded = LongTensor(len(batch), max_input_len)
    torch.nn.init.constant_(symbols_padded, self.padding_symbol_id)

    stresses_padded = None
    if self.use_stress:
      stresses_padded = LongTensor(len(batch), max_input_len)
      torch.nn.init.constant_(stresses_padded, 0)

      for i, batch_id in enumerate(ids_sorted_decreasing):
        _, _, _, stesses_tensor = batch[batch_id]
        stresses_padded[i, :stesses_tensor.size(0)] = stesses_tensor

    for i, batch_id in enumerate(ids_sorted_decreasing):
      symbols_tensor, _, _, _ = batch[batch_id]
      symbols_padded[i, :symbols_tensor.size(0)] = symbols_tensor

    # Right zero-pad mel-spec
    _, first_mel, _, _ = batch[0]
    num_mels = first_mel.size(0)
    max_target_len = max([mel_tensor.size(1) for _, mel_tensor, _, _ in batch])
    if max_target_len % self.n_frames_per_step != 0:
      max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
      assert max_target_len % self.n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()

    gate_padded = FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()

    output_lengths = LongTensor(len(batch))
    for i, batch_id in enumerate(ids_sorted_decreasing):
      _, mel_tensor, _, _ = batch[batch_id]
      mel_padded[i, :, :mel_tensor.size(1)] = mel_tensor
      gate_padded[i, mel_tensor.size(1) - 1:] = 1
      output_lengths[i] = mel_tensor.size(1)

    # count number of items - characters in text
    # len_x = []
    speaker_ids = []
    for i, batch_id in enumerate(ids_sorted_decreasing):
      # len_symb = batch[batch_id][0].get_shape()[0]
      # len_x.append(len_symb)
      _, _, speaker_id, _ = batch[batch_id]
      speaker_ids.append(speaker_id)

    # len_x = Tensor(len_x)
    speaker_ids = LongTensor(speaker_ids)

    return (
      symbols_padded,
      input_lengths,
      mel_padded,
      gate_padded,
      output_lengths,
      speaker_ids,
      stresses_padded
    )


def parse_batch(batch: Batch) -> Tuple[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]], Tuple[torch.FloatTensor, torch.FloatTensor]]:
  symbols_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids, stress_ids = batch
  symbols_padded = to_gpu(symbols_padded).long()
  input_lengths = to_gpu(input_lengths).long()
  max_len = torch.max(input_lengths.data).item()
  mel_padded = to_gpu(mel_padded).float()
  gate_padded = to_gpu(gate_padded).float()
  output_lengths = to_gpu(output_lengths).long()
  speaker_ids = to_gpu(speaker_ids).long()
  if stress_ids is not None:
    stress_ids = to_gpu(stress_ids).long()

  x = (symbols_padded, input_lengths,
       mel_padded, max_len, output_lengths, speaker_ids, stress_ids)
  y = (mel_padded, gate_padded)
  return x, y


def prepare_valloader(hparams: HParams, collate_fn: SymbolsMelCollate, valset: PreparedDataList, symbols_dict: Dict[Symbol, int], stress_dict: Optional[Dict[str, int]], logger: Logger) -> DataLoader:
  logger.info(
    f"Duration valset {valset.total_duration_s / 60:.2f}m / {valset.total_duration_s / 60 / 60:.2f}h")

  val = SymbolsMelLoader(valset, hparams, symbols_dict, stress_dict, logger)

  val_loader = DataLoader(
    dataset=val,
    num_workers=1,
    shuffle=False,
    sampler=None,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=False,
    collate_fn=collate_fn,
  )

  return val_loader


def prepare_trainloader(hparams: HParams, collate_fn: SymbolsMelCollate, trainset: PreparedDataList, symbols_dict: Dict[Symbol, int], stress_dict: Optional[Dict[str, int]], logger: Logger) -> DataLoader:
  # Get data, data loaders and collate function ready
  logger.info(
    f"Duration trainset {trainset.total_duration_s / 60:.2f}m / {trainset.total_duration_s / 60 / 60:.2f}h")

  trn = SymbolsMelLoader(trainset, hparams, symbols_dict, stress_dict, logger)

  train_loader = DataLoader(
    dataset=trn,
    num_workers=1,
    shuffle=True,
    sampler=None,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True,
    collate_fn=collate_fn,
  )

  return train_loader
