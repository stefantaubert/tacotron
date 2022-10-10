from collections import OrderedDict
from itertools import chain
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import torch
from torch import FloatTensor, IntTensor, LongTensor, Tensor  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tacotron.frontend.main import get_mapped_indices
from tacotron.hparams import HParams
from tacotron.model import ForwardXIn
from tacotron.taco_stft import TacotronSTFT
from tacotron.typing import (DurationMapping, Entries, Entry, Speaker, SpeakerId, SpeakerMapping,
                             Stress, Stresses, StressMapping, Symbol, SymbolMapping, Symbols,
                             ToneMapping, Tones)

LoaderEntry = Tuple[IntTensor, Tensor,
                    Optional[SpeakerId], Optional[IntTensor]]


class SymbolsMelLoader(Dataset):
  def __init__(self, data: Entries, hparams: HParams, symbol_mapping: SymbolMapping, stress_mapping: Optional[StressMapping], tone_mapping: Optional[ToneMapping], duration_mapping: Optional[DurationMapping], speaker_mapping: Optional[SpeakerMapping], device: torch.device, logger: Logger):
    super().__init__()

    # random.seed(hparams.seed)
    # random.shuffle(data)
    self.use_saved_mels = hparams.use_saved_mels
    self.use_cache: bool = hparams.cache_mels

    if not hparams.use_saved_mels:
      self.mel_parser = TacotronSTFT(hparams, device, logger)

    self.data: Dict[int, Tuple[IntTensor, Path,
                               Optional[SpeakerId], Optional[IntTensor], Optional[IntTensor], Optional[IntTensor]]] = {}

    # for i, entry, symbols in enumerate(zip(data.items(True), )

    entry: Entry
    for i, entry in enumerate(tqdm(data, desc="Reading files", unit=" file(s)")):
      symbol_ids, stress_ids, tone_ids, duration_ids, speaker_id = get_mapped_indices(
        entry.symbols, entry.speaker_name, symbol_mapping, stress_mapping, tone_mapping, duration_mapping, speaker_mapping, hparams)

      stress_tensor = None
      if hparams.use_stress_embedding:
        assert stress_ids is not None
        stress_tensor = IntTensor(stress_ids)

      tone_tensor = None
      if hparams.use_tone_embedding:
        assert tone_ids is not None
        tone_tensor = IntTensor(tone_ids)

      duration_tensor = None
      if hparams.use_duration_embedding:
        assert duration_ids is not None
        duration_tensor = IntTensor(duration_ids)

      symbols_tensor = IntTensor(symbol_ids)

      if hparams.use_saved_mels:
        raise NotImplementedError()
        # self.data[i] = (
        #   symbols_tensor, entry.mel_absolute_path, speaker_id, stress_tensor)
      self.data[i] = (
          symbols_tensor, entry.wav_absolute_path, speaker_id, stress_tensor, tone_tensor, duration_tensor)

    if hparams.use_saved_mels and hparams.cache_mels:
      logger.info("Loading mels into memory...")
      self.cache: Dict[int, Tensor] = {}
      vals: tuple
      for i, vals in tqdm(self.data.items()):
        mel_tensor = torch.load(vals[1], map_location='cpu')
        self.cache[i] = mel_tensor

  def __getitem__(self, index: int) -> LoaderEntry:
    # return self.cache[index]
    # debug_logger.debug(f"getitem called {index}")
    symbols_tensor, path, speaker_id, stress_tensor, tone_tensor, duration_tensor = self.data[index]
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

    tone_tensor_cloned = None
    if tone_tensor is not None:
      tone_tensor_cloned = tone_tensor.clone().detach()

    duration_tensor_cloned = None
    if duration_tensor is not None:
      duration_tensor_cloned = duration_tensor.clone().detach()

    # debug_logger.debug(f"getitem finished {index}")

    return symbols_tensor_cloned, mel_tensor, speaker_id, stress_tensor_cloned, tone_tensor_cloned, duration_tensor_cloned

  def __len__(self):
    return len(self.data)


Batch = Tuple[LongTensor, LongTensor, LongTensor, FloatTensor,
              FloatTensor, LongTensor, Optional[LongTensor], Optional[LongTensor], Optional[LongTensor], Optional[LongTensor]]


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """

  def __init__(self, hparams: HParams):
    self.n_frames_per_step = hparams.n_frames_per_step
    self.use_stress = hparams.use_stress_embedding
    self.use_tones = hparams.use_tone_embedding
    self.use_durations = hparams.use_duration_embedding
    self.use_speakers = hparams.use_speaker_embedding

  def __call__(self, batch: List[LoaderEntry]) -> Batch:
    # batches need to be sorted descending for encoder part: nn.utils.rnn.pack_padded_sequence
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    symbol_tensors, mel_tensors, speaker_ids, stress_tensors, tone_tensors, duration_tensors = zip(
      *batch)

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
      stresses_padded_tensor = LongTensor(
          len(stress_tensors), max_symbol_len)
      stresses_padded_tensor.zero_()
      for i, tensor in enumerate(stress_tensors):
        stresses_padded_tensor[i, :tensor.size(0)] = tensor

    # pad tones
    tones_padded_tensor = None
    if self.use_tones:
      # needs to be long for one-hot later
      tones_padded_tensor = LongTensor(
          len(tone_tensors), max_symbol_len)
      tones_padded_tensor.zero_()
      for i, tensor in enumerate(tone_tensors):
        tones_padded_tensor[i, :tensor.size(0)] = tensor

    # pad durations
    durations_padded_tensor = None
    if self.use_durations:
      # needs to be long for one-hot later
      durations_padded_tensor = LongTensor(
          len(duration_tensors), max_symbol_len)
      durations_padded_tensor.zero_()
      for i, tensor in enumerate(duration_tensors):
        durations_padded_tensor[i, :tensor.size(0)] = tensor

    # pad speakers
    speakers_padded_tensor = None
    if self.use_speakers:
      speakers_padded_tensor = IntTensor(
          len(speaker_ids), max_symbol_len)
      speakers_padded_tensor.zero_()
      for i, (symbols_len, speaker_id) in enumerate(zip(symbol_lens, speaker_ids)):
        speakers_padded_tensor[i, :symbols_len] = speaker_id

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
    mel_padded_tensor = FloatTensor(
        len(mel_tensors), num_mels, max_mel_len)
    mel_padded_tensor.zero_()
    for i, tensor in enumerate(mel_tensors):
      mel_padded_tensor[i, :, :tensor.size(1)] = tensor

    # pad gates
    gate_padded_tensor = FloatTensor(len(mel_tensors), max_mel_len)
    gate_padded_tensor.zero_()
    stop_token = 1
    for i, tensor in enumerate(mel_tensors):
      # TODO assert tensor.size(1) > 1
      # the last frame is set to the stop token
      gate_padded_tensor[i, tensor.size(1) - 1] = stop_token
      # pad the stop token
      gate_padded_tensor[i, tensor.size(1):] = stop_token

    return (
        symbols_padded_tensor,
        symbol_lens_tensor,
        mel_padded_tensor,
        gate_padded_tensor,
        mel_lens_tensor,
        speakers_padded_tensor,
        stresses_padded_tensor,
        tones_padded_tensor,
        durations_padded_tensor,
    )


def parse_batch(batch: Batch) -> Tuple[ForwardXIn, Tuple[FloatTensor, FloatTensor]]:
  symbols_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids, stress_ids, tone_ids, duration_ids = batch

  x = (symbols_padded, input_lengths,
       mel_padded, output_lengths, speaker_ids, stress_ids, tone_ids, duration_ids)
  y = (mel_padded, gate_padded)
  return x, y


def prepare_valloader(hparams: HParams, collate_fn: SymbolsMelCollate, valset: Entries, symbol_mapping: SymbolMapping, stress_mapping: Optional[StressMapping], tone_mapping: Optional[ToneMapping], duration_mapping: Optional[DurationMapping], speaker_mapping: Optional[SpeakerMapping], device: torch.device, logger: Logger) -> DataLoader:
  # logger.info(
  #   f"Duration valset {valset.total_duration_s / 60:.2f}m / {valset.total_duration_s / 60 / 60:.2f}h")

  val = SymbolsMelLoader(valset, hparams, symbol_mapping,
                         stress_mapping, tone_mapping, duration_mapping, speaker_mapping, device, logger)

  device_is_cuda = device.type == "cuda"

  val_loader = DataLoader(
      dataset=val,
      num_workers=16,
      shuffle=False,
      sampler=None,
      batch_size=hparams.batch_size,
      pin_memory=device_is_cuda,
      drop_last=False,
      collate_fn=collate_fn,
  )

  return val_loader


def prepare_trainloader(hparams: HParams, collate_fn: SymbolsMelCollate, trainset: Entries, symbol_mapping: SymbolMapping, stress_mapping: Optional[StressMapping], tone_mapping: Optional[ToneMapping], duration_mapping: Optional[DurationMapping], speaker_mapping: Optional[SpeakerMapping], device: torch.device, logger: Logger) -> DataLoader:
  # # Get data, data loaders and collate function ready
  # logger.info(
  #   f"Duration trainset {trainset.total_duration_s / 60:.2f}m / {trainset.total_duration_s / 60 / 60:.2f}h")

  trn = SymbolsMelLoader(trainset, hparams, symbol_mapping,
                         stress_mapping, tone_mapping, duration_mapping, speaker_mapping, device, logger)

  # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/7
  # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
  device_is_cuda = device.type == "cuda"

  train_loader = DataLoader(
      dataset=trn,
      num_workers=16,
      # shuffle for better training and to fix that the last batch is dropped
      shuffle=True,
      sampler=None,
      batch_size=hparams.batch_size,
      pin_memory=device_is_cuda,
      #  drop the last incomplete batch, if the dataset size is not divisible by the batch size
      drop_last=True,
      collate_fn=collate_fn,
  )

  return train_loader
