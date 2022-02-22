from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
from itertools import chain
from logging import Logger
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import torch
from audio_utils.mel import TacotronSTFT
from tacotron.core.hparams import HParams
from tacotron.core.model import ForwardXIn
from tacotron.utils import to_gpu
from text_utils import SpeakerId, Symbol, Symbols
from text_utils.pronunciation.arpa_symbols import VOWELS_WITH_STRESSES
from torch import (ByteTensor, FloatTensor, IntTensor,  # pylint: disable=no-name-in-module
                   LongTensor, ShortTensor, Tensor)
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


def get_symbols_dict(valset: PreparedDataList, trainset: PreparedDataList) -> OrderedDictType[Symbol, int]:
  all_valsymbols = (entry.symbols for entry in valset.items())
  all_trainsymbols = (entry.symbols for entry in trainset.items())
  all_symbols = chain(all_valsymbols, all_trainsymbols)

  unique_symbols = {symbol for symbols in all_symbols for symbol in symbols}

  symbol_ids = OrderedDict((
    (symbol, symbol_nr + PADDING_SHIFT)
    for symbol_nr, symbol in enumerate(sorted(unique_symbols))
  ))

  return symbol_ids


def get_symbols_stresses_dicts(valset: PreparedDataList, trainset: PreparedDataList) -> Tuple[OrderedDictType[Symbol, int], OrderedDictType[str, int]]:
  all_valsymbols = (entry.symbols for entry in valset.items())
  all_trainsymbols = (entry.symbols for entry in trainset.items())
  all_symbols = chain(all_valsymbols, all_trainsymbols)
  all_symbols_stress_splitted = (split_stresses_arpa(symbols) for symbols in all_symbols)

  all_symbols, all_stresses = zip(*all_symbols_stress_splitted)
  unique_symbols = {symbol for symbols in all_symbols for symbol in symbols}
  unique_stresses = {stress for stresses in all_stresses for stress in stresses}

  symbol_ids = OrderedDict((
    (symbol, symbol_nr + PADDING_SHIFT)
    for symbol_nr, symbol in enumerate(sorted(unique_symbols))
  ))

  stress_ids = OrderedDict((
    (stress, stress_nr + PADDING_SHIFT)
    for stress_nr, stress in enumerate(sorted(unique_stresses))
  ))

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


Batch = Tuple[LongTensor, LongTensor, LongTensor, FloatTensor,
              FloatTensor, LongTensor, LongTensor, Optional[LongTensor]]


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """

  def __init__(self, n_frames_per_step: int, padding_symbol_id: int, use_stress: bool):
    self.n_frames_per_step = n_frames_per_step
    self.use_stress = use_stress
    self.use_speakers = True

  def __call__(self, batch: List[LoaderEntry]) -> Batch:
    # batches need to be sorted descending for encoder part: nn.utils.rnn.pack_padded_sequence
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    symbol_tensors, mel_tensors, speaker_ids, stress_tensors = zip(*batch)

    symbol_lens = [tensor.size(0) for tensor in symbol_tensors]
    symbol_lens_tensor = IntTensor(symbol_lens)

    # prepare padding
    max_symbol_len = max(symbol_lens)

    # pad symbols
    symbols_padded_tensor = IntTensor(len(symbol_tensors), max_symbol_len)
    symbols_padded_tensor.zero_()
    for i, tensor in enumerate(symbol_tensors):
      symbols_padded_tensor[i, :tensor.size(0)] = tensor

    # pad stresses
    stresses_padded_tensor = None
    if self.use_stress:
      # needs to be long for one-hot later
      stresses_padded_tensor = LongTensor(len(stress_tensors), max_symbol_len)
      stresses_padded_tensor.zero_()
      for i, tensor in enumerate(stress_tensors):
        stresses_padded_tensor[i, :tensor.size(0)] = tensor

    # pad speakers
    speakers_padded_tensor = None
    if self.use_speakers:
      speakers_padded_tensor = IntTensor(len(speaker_ids), max_symbol_len)
      speakers_padded_tensor.zero_()
      for i, (symbols_len, speaker_id) in enumerate(zip(symbol_lens, speaker_ids)):
        speakers_padded_tensor[i, :symbols_len] = speaker_id + 1

    # calculate mel lengths
    mel_lens = [tensor.size(1) for tensor in mel_tensors]
    mel_lens_tensor = IntTensor(mel_lens)

    # prepare mel padding
    max_mel_len = max(mel_lens)
    if max_mel_len % self.n_frames_per_step != 0:
      max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
      assert max_mel_len % self.n_frames_per_step == 0

    # pad mels
    # 80
    num_mels = mel_tensors[0].size(0)
    mel_padded_tensor = FloatTensor(len(mel_tensors), num_mels, max_mel_len)
    mel_padded_tensor.zero_()
    for i, tensor in enumerate(mel_tensors):
      mel_padded_tensor[i, :, :tensor.size(1)] = tensor

    # pad gates
    gate_padded_tensor = FloatTensor(len(mel_tensors), max_mel_len)
    gate_padded_tensor.zero_()
    for i, tensor in enumerate(mel_tensors):
      # why - 1?
      gate_padded_tensor[i, tensor.size(1) - 1:] = 1

    return (
      symbols_padded_tensor,
      symbol_lens_tensor,
      mel_padded_tensor,
      gate_padded_tensor,
      mel_lens_tensor,
      speakers_padded_tensor,
      stresses_padded_tensor
    )


def parse_batch(batch: Batch) -> Tuple[ForwardXIn, Tuple[FloatTensor, FloatTensor]]:
  symbols_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids, stress_ids = batch
  symbols_padded = to_gpu(symbols_padded)
  input_lengths = to_gpu(input_lengths)
  mel_padded = to_gpu(mel_padded)
  gate_padded = to_gpu(gate_padded)
  output_lengths = to_gpu(output_lengths)
  if speaker_ids is not None:
    speaker_ids = to_gpu(speaker_ids)
  if stress_ids is not None:
    stress_ids = to_gpu(stress_ids)

  x = (symbols_padded, input_lengths,
       mel_padded, output_lengths, speaker_ids, stress_ids)
  y = (mel_padded, gate_padded)
  return x, y


def prepare_valloader(hparams: HParams, collate_fn: SymbolsMelCollate, valset: PreparedDataList, symbols_dict: Dict[Symbol, int], stress_dict: Optional[Dict[str, int]], logger: Logger) -> DataLoader:
  logger.info(
    f"Duration valset {valset.total_duration_s / 60:.2f}m / {valset.total_duration_s / 60 / 60:.2f}h")

  val = SymbolsMelLoader(valset, hparams, symbols_dict, stress_dict, logger)

  val_loader = DataLoader(
    dataset=val,
    num_workers=16,
    shuffle=False,
    sampler=None,
    batch_size=hparams.batch_size,
    pin_memory=True,
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
    num_workers=16,
    # shuffle for better training and to fix that the last batch is dropped
    shuffle=True,
    sampler=None,
    batch_size=hparams.batch_size,
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/7
    pin_memory=True,
    #  drop the last incomplete batch, if the dataset size is not divisible by the batch size
    drop_last=True,
    collate_fn=collate_fn,
  )

  return train_loader
