import datetime
import os
import random
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import Generator
from typing import OrderedDict as OrderedDictType

import numpy as np
from textgrid import IntervalTier, TextGrid
from tqdm import tqdm

from tacotron.checkpoint_handling import get_learning_rate, get_speaker_mapping
from tacotron.synthesizer import Synthesizer
from tacotron.typing import Speaker, Symbols
from tacotron.utils import set_torch_thread_to_max, split_hparams_string
from tacotron_cli.argparse_helper import (get_optional, parse_codec, parse_existing_directory,
                                          parse_existing_file, parse_non_empty,
                                          parse_non_negative_integer, parse_path)
from tacotron_cli.helper import (add_device_argument, add_hparams_argument,
                                 add_max_decoder_steps_argument)
from tacotron_cli.io import try_load_checkpoint

Utterances = OrderedDictType[int, Symbols]
Paragraphs = OrderedDictType[int, Utterances]


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


def init_grid_synthesis_parser(parser: ArgumentParser) -> None:
  parser.description = "Synthesize each .TextGrid file into a mel-spectrogram."
  parser.add_argument('checkpoint', metavar="CHECKPOINT", type=parse_existing_file,
                      help="path to checkpoint that should be used for synthesis")
  parser.add_argument('directory', metavar="DIRECTORY", type=parse_existing_directory,
                      help="path to folder containing .TextGrid files that should be synthesized")
  parser.add_argument('tier', metavar="TIER", type=parse_non_empty,
                      help="tier containing the symbols in separate intervals")
  parser.add_argument('--speaker', type=get_optional(parse_non_empty), default=None,
                      help="use that speaker for syntheses; defaults to the first speaker if left unset")
  add_max_decoder_steps_argument(parser)
  # parser.add_argument('--batch-size', type=parse_positive_integer, default=64, help="")
  parser.add_argument('--seed', type=get_optional(parse_non_negative_integer),
                      default=None, help="custom seed used for synthesis; if left unset a random seed will be chosen")
  add_device_argument(parser)
  add_hparams_argument(parser)
  parser.add_argument('--encoding', type=parse_codec, default="UTF-8",
                      help="encoding of .TextGrid files")
  parser.add_argument('-out', '--output-directory', type=parse_path, default=None,
                      help="custom output directory")
  return synthesize_ns


def synthesize_ns(ns: Namespace) -> bool:
  logger = getLogger(__name__)
  set_torch_thread_to_max()

  checkpoint_dict = try_load_checkpoint(ns.checkpoint, ns.device, logger)
  if checkpoint_dict is None:
    return False

  speaker: Speaker = ns.speaker
  if ns.speaker is not None:
    speaker_mapping = get_speaker_mapping(checkpoint_dict)
    if ns.speaker in speaker_mapping:
      speaker = ns.speaker
    else:
      speaker_parts = ns.speaker.split(";")
      tried_speaker = speaker_parts[0]
      if tried_speaker not in speaker_mapping:
        logger.error(f"Speaker \"{ns.speaker}\" doesn't exist!")
        return False
      speaker = tried_speaker
  else:
    speaker_mapping = get_speaker_mapping(checkpoint_dict)
    speaker = next(iter(speaker_mapping.keys()))
  logger.info(f"Speaker: \"{speaker}\"")

  output_directory: Path = ns.output_directory
  if output_directory is None:
    output_directory = ns.directory

  if output_directory.is_file():
    logger.error("Output directory is a file!")
    return False

  seed: int = ns.seed
  if seed is None:
    seed = random.randint(1, 9999)
  logger.info(f"Seed: {seed}")

  logger.info(
      f"Last checkpoint learning rate was: {get_learning_rate(checkpoint_dict)}")

  custom_hparams = split_hparams_string(ns.custom_hparams)

  synth = Synthesizer(
    checkpoint=checkpoint_dict,
    custom_hparams=custom_hparams,
    device=ns.device,
    logger=logger,
  )

  all_files = get_all_files_in_all_subfolders(ns.directory)
  all_textgrid_files = [file for file in all_files if file.suffix.lower() == ".textgrid"]

  logger.info("Inferring...")
  unmappable_symbols = set()

  with tqdm(all_textgrid_files, unit=" grid(s)", ncols=100, desc="Inferring") as progress_bar:
    for grid_path in progress_bar:
      grid = TextGrid()
      try:
        grid.read(grid_path, 16, ns.encoding)
      except Exception as ex:
        logger.error(f"File \"{grid_path.absolute()}\" couldn't be read! Skipped.")
        logger.debug(ex)
        continue
      tier: IntervalTier = grid.getFirst(ns.tier)
      if tier is None:
        logger.error(f"File \"{grid_path.absolute()}\" has no tier \"{ns.tier}\"! Skipped.")
        continue
      utterance = tuple(interval.mark for interval in tier.intervals)

      logger.info(f"Inferring \"{grid_path.absolute()}\"")
      logger.info(f"Timepoint: {datetime.datetime.now()}")
      inf_sent_output = synth.infer(
        symbols=utterance,
        speaker=speaker,
        include_stats=False,
        max_decoder_steps=ns.max_decoder_steps,
        seed=seed,
      )

      output_path = output_directory / \
          grid_path.relative_to(ns.directory).parent / f"{grid_path.stem}.npy"
      output_path.parent.mkdir(exist_ok=True, parents=True)
      np.save(output_path, inf_sent_output.mel_outputs_postnet)
      logger.info(f"Saved output to: \"{output_path.absolute()}\"")

      unmappable_symbols |= inf_sent_output.unmapable_symbols

      logger.info(f"Spectrogram duration: {inf_sent_output.duration_s:.2f}s")
      if inf_sent_output.reached_max_decoder_steps:
        logger.warning("Reached max decoder steps!")
      logger.debug(
          f"Inference duration: {inf_sent_output.inference_duration_s}")
      logger.debug(
          f"Sampling rate: {inf_sent_output.sampling_rate}")
      if len(inf_sent_output.unmapable_symbols) > 0:
        logger.warning(
            f"Unknown symbols: {' '.join(sorted(inf_sent_output.unmapable_symbols))}")
      else:
        logger.debug("All symbols were known.")

  if len(unmappable_symbols) > 0:
    logger.warning(
        f"Unknown symbols: {' '.join(sorted(unmappable_symbols))} (#{len(unmappable_symbols)})")

  logger.info(f"Written output to: \"{output_directory.absolute()}\"")

  return True
