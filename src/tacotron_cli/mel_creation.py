from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tacotron.audio_utils import mel_to_numpy
from tacotron.taco_stft import TacotronSTFT, TSTFTHParams
from tacotron.utils import set_torch_thread_to_max
from tacotron_cli.argparse_helper import (get_optional, parse_existing_directory,
                                          parse_non_empty_or_whitespace, parse_non_negative_float,
                                          parse_non_negative_integer, parse_path)
from tacotron_cli.helper import add_device_argument
from tacotron_cli.logging_configuration import LOGGER_NAME
from tacotron_cli.textgrid_inference import get_all_files_in_all_subfolders


def init_mel_creation_parser(parser: ArgumentParser) -> None:
  parser.description = "Create numpy mel-spectrograms (.npy) from .wav files."
  parser.add_argument('folder', metavar="FOLDER",
                      type=parse_existing_directory, help="path to folder containing .wav files")
  parser.add_argument("--filter-length", metavar="LENGTH",
                      type=parse_non_negative_integer, default=1024)
  parser.add_argument("--hop-length", metavar="LENGTH",
                      type=parse_non_negative_integer, default=256)
  parser.add_argument("--win-length", metavar="LENGTH",
                      type=parse_non_negative_integer, default=1024)
  parser.add_argument("--window", metavar="WINDOW",
                      type=parse_non_empty_or_whitespace, default="hann")
  parser.add_argument("--n-mel-channels", metavar="N",
                      type=parse_non_negative_integer, default=80)
  parser.add_argument("--sampling-rate", metavar="SAMPLING-RATE",
                      type=parse_non_negative_integer, default=22050)
  parser.add_argument("--mel-fmin", metavar="FMAX", type=parse_non_negative_float, default=0.0)
  parser.add_argument("--mel-fmax", metavar="FMAX", type=parse_non_negative_float, default=8000.0)
  parser.add_argument('-out', '--custom-output-directory',
                      metavar="OUTPUT-FOLDER", type=get_optional(parse_path), help="path to folder in which the resulting files should be written if not to FOLDER", default=None)
  add_device_argument(parser)
  return create_mels_ns


def create_mels_ns(ns: Namespace) -> None:
  logger = getLogger(LOGGER_NAME)

  output_directory: Path = ns.custom_output_directory
  if output_directory is None:
    output_directory = ns.folder
  else:
    if output_directory.is_file():
      logger.error("Output directory is a file!")
      return False

  all_files = get_all_files_in_all_subfolders(ns.folder)
  all_wav_files = list(file for file in all_files if file.suffix.lower() == ".wav")

  set_torch_thread_to_max()

  hparams = TSTFTHParams(
    filter_length=ns.filter_length,
    hop_length=ns.hop_length,
    mel_fmax=ns.mel_fmax,
    mel_fmin=ns.mel_fmin,
    n_mel_channels=ns.n_mel_channels,
    sampling_rate=ns.sampling_rate,
    win_length=ns.win_length,
    window=ns.window,
  )

  taco_stft = TacotronSTFT(hparams, ns.device)

  all_wav_files = tqdm(all_wav_files, unit=" wav(s)", ncols=100, desc="Creating")
  for wav_path in all_wav_files:
    logger.debug(f"Loading wav: \"{wav_path}\"")
    mel = taco_stft.get_mel_tensor_from_file(wav_path)
    mel_np = mel_to_numpy(mel)

    output_npy_path = output_directory / \
        wav_path.relative_to(ns.folder).parent / f"{wav_path.stem}.npy"
    logger.debug(f"Saving: \"{output_npy_path.absolute()}\"")
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy_path, mel_np)

  logger.info(f"Saved output to: \"{output_directory.absolute()}\"")

  return True
