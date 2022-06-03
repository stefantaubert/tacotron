from typing import List, Optional, Tuple, TypeVar

import matplotlib.ticker as ticker
import numpy as np
import torch
from fastdtw.fastdtw import fastdtw
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
from scipy.spatial.distance import euclidean

from tacotron.utils import figure_to_numpy_rgb

_T = TypeVar('_T')
PYTORCH_EXT = ".pt"


def align_mels_with_dtw(mel_spec_1: np.ndarray, mel_spec_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int]]:
  mel_spec_1, mel_spec_2 = mel_spec_1.T, mel_spec_2.T
  dist, path = fastdtw(mel_spec_1, mel_spec_2, dist=euclidean)
  path_for_mel_spec_1 = list(map(lambda l: l[0], path))
  path_for_mel_spec_2 = list(map(lambda l: l[1], path))
  aligned_mel_spec_1 = mel_spec_1[path_for_mel_spec_1].T
  aligned_mel_spec_2 = mel_spec_2[path_for_mel_spec_2].T
  return aligned_mel_spec_1, aligned_mel_spec_2, dist, path_for_mel_spec_1, path_for_mel_spec_2


def get_msd(dist: float, total_frame_number: int) -> float:
  msd = dist / total_frame_number
  return msd


def plot_melspec_np(mel: np.ndarray, mel_dim_x: int = 16, mel_dim_y: int = 5, factor: int = 1, title: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
  height, width = mel.shape
  width_factor = width / 1000
  fig, axes = plt.subplots(
      nrows=1,
      ncols=1,
      figsize=(mel_dim_x * factor * width_factor, mel_dim_y * factor),
  )

  img = axes.imshow(
      X=mel,
      aspect='auto',
      origin='lower',
      interpolation='none'
  )

  axes.set_yticks(np.arange(0, height, step=5))
  axes.set_xticks(np.arange(0, width, step=50))
  axes.xaxis.set_major_locator(ticker.NullLocator())
  axes.yaxis.set_major_locator(ticker.NullLocator())
  plt.tight_layout()  # font logging occurs here
  figa_core = figure_to_numpy_rgb(fig)

  fig.colorbar(img, ax=axes)
  axes.xaxis.set_major_locator(ticker.AutoLocator())
  axes.yaxis.set_major_locator(ticker.AutoLocator())

  if title is not None:
    axes.set_title(title)
  axes.set_xlabel("Frames")
  axes.set_ylabel("Freq. channel")
  plt.tight_layout()  # font logging occurs here
  figa_labeled = figure_to_numpy_rgb(fig)
  plt.close()

  return figa_core, figa_labeled


def wav_to_float32_tensor(path: str) -> Tuple[torch.Tensor, int]:
  wav, sampling_rate = wav_to_float32(path)
  wav_tensor = torch.FloatTensor(wav)

  return wav_tensor, sampling_rate


def wav_to_float32(path: str) -> Tuple[np.float, int]:
  sampling_rate, wav = read(path)
  wav = convert_wav(wav, np.float32)
  return wav, sampling_rate


def convert_wav(wav, to_dtype):
  '''
  if the wav is overamplified the result will also be overamplified.
  '''
  if wav.dtype != to_dtype:
    wav = wav / (-1 * get_min_value(wav.dtype)) * get_max_value(to_dtype)
    if to_dtype in (np.int16, np.int32):
      # the default seems to be np.fix instead of np.round on wav.astype()
      wav = np.round(wav, 0)
    wav = wav.astype(to_dtype)

  return wav


def get_max_value(dtype):
  # see wavfile.write() max positive eg. on 16-bit PCM is 32767
  if dtype == np.int16:
    return INT16_MAX

  if dtype == np.int32:
    return INT32_MAX

  if dtype in (np.float32, np.float64):
    return FLOAT32_64_MAX_WAV

  assert False


def get_min_value(dtype):
  if dtype == np.int16:
    return INT16_MIN

  if dtype == np.int32:
    return INT32_MIN

  if dtype == np.float32 or dtype == np.float64:
    return FLOAT32_64_MIN_WAV

  assert False


FLOAT32_64_MIN_WAV = -1.0
FLOAT32_64_MAX_WAV = 1.0
INT16_MIN = np.iinfo(np.int16).min  # -32768 = -(2**15)
INT16_MAX = np.iinfo(np.int16).max  # 32767 = 2**15 - 1
INT32_MIN = np.iinfo(np.int32).min  # -2147483648 = -(2**31)
INT32_MAX = np.iinfo(np.int32).max  # 2147483647 = 2**31 - 1


def mel_to_numpy(mel: torch.Tensor) -> np.ndarray:
  mel = mel.squeeze(0)
  mel = mel.cpu()
  mel_np: np.ndarray = mel.numpy()
  return mel_np
