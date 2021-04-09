import random
from logging import Logger
from typing import Dict, List, Tuple

import torch
from audio_utils.mel import TacotronSTFT
from tacotron.core.hparams import HParams
from tacotron.core.model_symbols import get_model_symbol_ids
from tacotron.utils import to_gpu
from text_utils import deserialize_list
from torch import (FloatTensor, IntTensor,  # pylint: disable=no-name-in-module
                   LongTensor, Tensor)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tts_preparation import PreparedDataList


class SymbolsMelLoader(Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """

  def __init__(self, data: PreparedDataList, hparams: HParams, logger: Logger):
    # random.seed(hparams.seed)
    # random.shuffle(data)
    self.use_saved_mels: bool = hparams.use_saved_mels
    if not hparams.use_saved_mels:
      self.mel_parser = TacotronSTFT(hparams, logger)

    logger.info("Reading files...")
    self.data: Dict[int, Tuple[IntTensor, IntTensor, str, int]] = {}
    for i, values in enumerate(data.items(True)):
      symbol_ids = deserialize_list(values.serialized_symbol_ids)
      accent_ids = deserialize_list(values.serialized_accent_ids)

      model_symbol_ids = get_model_symbol_ids(
        symbol_ids, accent_ids, hparams.n_symbols, hparams.accents_use_own_symbols)

      symbols_tensor = IntTensor(model_symbol_ids)
      accents_tensor = IntTensor(accent_ids)

      if hparams.use_saved_mels:
        self.data[i] = (symbols_tensor, accents_tensor, values.mel_path, values.speaker_id)
      else:
        self.data[i] = (symbols_tensor, accents_tensor, values.wav_path, values.speaker_id)

    if hparams.use_saved_mels and hparams.cache_mels:
      logger.info("Loading mels into memory...")
      self.cache: Dict[int, Tensor] = {}
      vals: tuple
      for i, vals in tqdm(self.data.items()):
        mel_tensor = torch.load(vals[1], map_location='cpu')
        self.cache[i] = mel_tensor
    self.use_cache: bool = hparams.cache_mels

  def __getitem__(self, index: int) -> Tuple[IntTensor, IntTensor, Tensor, int]:
    # return self.cache[index]
    # debug_logger.debug(f"getitem called {index}")
    symbols_tensor, accents_tensor, path, speaker_id = self.data[index]
    if self.use_saved_mels:
      if self.use_cache:
        mel_tensor = self.cache[index].clone().detach()
      else:
        mel_tensor: Tensor = torch.load(path, map_location='cpu')
    else:
      mel_tensor = self.mel_parser.get_mel_tensor_from_file(path)

    symbols_tensor_cloned = symbols_tensor.clone().detach()
    accents_tensor_cloned = accents_tensor.clone().detach()
    # debug_logger.debug(f"getitem finished {index}")
    return symbols_tensor_cloned, accents_tensor_cloned, mel_tensor, speaker_id

  def __len__(self):
    return len(self.data)


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """

  def __init__(self, n_frames_per_step: int, padding_symbol_id: int, padding_accent_id: int):
    self.n_frames_per_step = n_frames_per_step
    self.padding_symbol_id = padding_symbol_id
    self.padding_accent_id = padding_accent_id

  def __call__(self, batch: List[Tuple[IntTensor, IntTensor, Tensor, int]]):
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

    accents_padded = LongTensor(len(batch), max_input_len)
    torch.nn.init.constant_(accents_padded, self.padding_accent_id)

    for i, batch_id in enumerate(ids_sorted_decreasing):
      symbols = batch[batch_id][0]
      symbols_padded[i, :symbols.size(0)] = symbols

      accents = batch[batch_id][1]
      accents_padded[i, :accents.size(0)] = accents

    # Right zero-pad mel-spec
    _, _, first_mel, _ = batch[0]
    num_mels = first_mel.size(0)
    max_target_len = max([mel_tensor.size(1) for _, _, mel_tensor, _ in batch])
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
      _, _, mel, _ = batch[batch_id]
      mel_padded[i, :, :mel.size(1)] = mel
      gate_padded[i, mel.size(1) - 1:] = 1
      output_lengths[i] = mel.size(1)

    # count number of items - characters in text
    # len_x = []
    speaker_ids = []
    for i, batch_id in enumerate(ids_sorted_decreasing):
      # len_symb = batch[batch_id][0].get_shape()[0]
      # len_x.append(len_symb)
      _, _, _, speaker_id = batch[batch_id]
      speaker_ids.append(speaker_id)

    # len_x = Tensor(len_x)
    speaker_ids = LongTensor(speaker_ids)

    return make_batch(
      symbols_padded,
      accents_padded,
      input_lengths,
      mel_padded,
      gate_padded,
      output_lengths,
      speaker_ids
    )


def make_batch(symbols_padded: torch.LongTensor, accents_padded: torch.LongTensor, input_lengths: torch.LongTensor, mel_padded: torch.FloatTensor, gate_padded: torch.FloatTensor, output_lengths: torch.LongTensor, speaker_ids: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
  return symbols_padded, accents_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids


def parse_batch(batch: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]) -> Tuple[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
  symbols_padded, accents_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids = batch
  symbols_padded = to_gpu(symbols_padded).long()
  accents_padded = to_gpu(accents_padded).long()
  input_lengths = to_gpu(input_lengths).long()
  max_len = torch.max(input_lengths.data).item()
  mel_padded = to_gpu(mel_padded).float()
  gate_padded = to_gpu(gate_padded).float()
  output_lengths = to_gpu(output_lengths).long()
  speaker_ids = to_gpu(speaker_ids).long()

  x = (symbols_padded, accents_padded, input_lengths,
       mel_padded, max_len, output_lengths, speaker_ids)
  y = (mel_padded, gate_padded)
  return x, y


def prepare_valloader(hparams: HParams, collate_fn: SymbolsMelCollate, valset: PreparedDataList, logger: Logger) -> DataLoader:
  logger.info(
    f"Duration valset {valset.get_total_duration_s() / 60:.2f}m / {valset.get_total_duration_s() / 60 / 60:.2f}h")

  val = SymbolsMelLoader(valset, hparams, logger)

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


def prepare_trainloader(hparams: HParams, collate_fn: SymbolsMelCollate, trainset: PreparedDataList, logger: Logger) -> DataLoader:
  # Get data, data loaders and collate function ready
  logger.info(
    f"Duration trainset {trainset.get_total_duration_s() / 60:.2f}m / {trainset.get_total_duration_s() / 60 / 60:.2f}h")

  trn = SymbolsMelLoader(trainset, hparams, logger)

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
