import datetime
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set

import imageio
import numpy as np
from audio_utils.mel import plot_melspec_np
from general_utils import parse_json, save_json
from image_utils import stack_images_horizontally, stack_images_vertically
from ordered_set import OrderedSet
from tacotron.app.defaults import (DEFAULT_MAX_DECODER_STEPS,
                                   DEFAULT_SAVE_MEL_INFO_COPY_PATH,
                                   DEFAULT_SEED)
from tacotron.app.io import (get_checkpoints_dir, get_inference_root_dir,
                             get_mel_info_dict, get_mel_out_dict,
                             get_train_dir, load_checkpoint,
                             load_prep_settings)
from tacotron.core import InferenceEntries, InferenceEntryOutput
from tacotron.core import infer as infer_core
from tacotron.core.checkpoint_handling import get_speaker_mapping
from tacotron.core.inference import get_df
from tacotron.core.synthesizer import Synthesizer
from tacotron.globals import DEFAULT_CSV_SEPERATOR
from tacotron.utils import (add_console_out_to_logger, add_file_out_to_logger,
                            get_custom_or_last_checkpoint, get_default_logger,
                            init_logger, plot_alignment_np_new)
from text_utils import Speaker, StringFormat2, Symbols
from tqdm import tqdm
from tts_preparation import (InferableUtterance, InferableUtterances,
                             get_merged_dir, get_text_dir, load_utterances)

Utterances = OrderedDictType[int, Symbols]
Paragraphs = OrderedDictType[int, Utterances]


def parse_paragraphs_from_text(text: str) -> Paragraphs:
  symbol_sep = "|"
  logger = getLogger(__name__)
  lines = text.splitlines()
  result = OrderedDict()
  paragraph_nr = 1
  current_utterances = OrderedDict()
  for line_nr, line in enumerate(lines, start=1):
    if line == "":
      if len(current_utterances) > 0:
        assert paragraph_nr not in result
        result[paragraph_nr] = current_utterances
        paragraph_nr += 1
        current_utterances = OrderedDict()
    else:
      if not StringFormat2.SPACED.can_convert_string_to_symbols(line, symbol_sep):
        logger.error(f"Line {line_nr}: Line couldn't be parsed! Skipped.")
        continue
      line_symbols = StringFormat2.SPACED.convert_string_to_symbols(line, symbol_sep)
      assert line_nr not in current_utterances
      current_utterances[line_nr] = line_symbols

  if len(current_utterances) > 0:
    result[paragraph_nr] = current_utterances
  return result


def infer_text(base_dir: Path, checkpoint: Path, text: Path, encoding: str, custom_speaker: Optional[Speaker], custom_lines: List[int], max_decoder_steps: int, batch_size: int, include_stats: bool, custom_seed: Optional[int], paragraph_directories: bool, output_directory: Optional[Path], overwrite: bool) -> bool:
  logger = getLogger(__name__)

  if not checkpoint.is_file():
    logger.error("Checkpoint was not found!")
    return False

  if not text.is_file():
    logger.error("Text was not found!")
    return False

  custom_lines = OrderedSet(custom_lines)
  if not all(x >= 0 for x in custom_lines):
    logger.error("Custom line values need to be greater than or equal to zero!")
    return False

  if not max_decoder_steps > 0:
    logger.error("Maximum decoder steps need to be greater than zero!")
    return False

  if not batch_size > 0:
    logger.error("Batch size need to be greater than zero!")
    return False

  if custom_seed is not None and not custom_seed >= 0:
    logger.error("Custom seed needs to be greater than or equal to zero!")
    return False

  try:
    logger.debug(f"Loading checkpoint...")
    checkpoint_dict = load_checkpoint(checkpoint)
  except Exception as ex:
    logger.error("Checkpoint couldn't be loaded!")
    return False

  if custom_speaker is not None:
    speaker_mapping = get_speaker_mapping(checkpoint_dict)
    if custom_speaker not in speaker_mapping:
      logger.error("Custom speaker was not found!")
      return False

  try:
    logger.debug(f"Loading text.")
    text_content = text.read_text(encoding)
  except Exception as ex:
    logger.error("Text couldn't be read!")
    return False

  if output_directory is None:
    output_directory = text.parent / text.stem

  if output_directory.is_file():
    logger.error("Output directory is a file!")
    return False

  paragraphs = parse_paragraphs_from_text(text_content)

  line_nrs_to_infer = OrderedSet(line_nr for par in paragraphs.values() for line_nr in par.keys())
  if len(custom_lines) > 0:
    for custom_line in custom_lines:
      if custom_line not in line_nrs_to_infer:
        logger.error(f"Line {custom_line} is not inferable!")
        return False

    line_nrs_to_infer = custom_lines

  if custom_seed is not None:
    seed = custom_seed
  else:
    seed = random.randint(1, 9999)
    logger.info(f"Using random seed: {seed}.")

  if custom_speaker is not None:
    speaker = custom_speaker
  else:
    speaker_mapping = get_speaker_mapping(checkpoint_dict)
    speaker = next(iter(speaker_mapping.keys()))
    logger.debug(f"Speaker: {speaker}")

  synth = Synthesizer(
    checkpoint=checkpoint_dict,
    custom_hparams=None,
    logger=logger,
  )

  max_paragraph_nr = max(paragraphs.keys())
  max_line_nr = max(utt_nr for paragraph in paragraphs.values() for utt_nr in paragraph.keys())
  zfill_paragraph = len(str(max_paragraph_nr))
  zfill_line_nr = len(str(max_line_nr))
  unknown_symbols = set()
  with tqdm(total=len(line_nrs_to_infer), unit=" lines", ncols=100, desc="Inference") as progress_bar:
    for paragraph_nr, utterances in paragraphs.items():
      if paragraph_directories:
        paragraph_folder = output_directory / f"{paragraph_nr}".zfill(zfill_paragraph)
      else:
        paragraph_folder = output_directory
      for line_nr, utterance in utterances.items():
        if line_nr not in line_nrs_to_infer:
          logger.debug(f"Skipped line {line_nr}.")
          continue

        utt_path_stem = f"{line_nr}".zfill(zfill_line_nr)
        utterance_mel_path = paragraph_folder / f"{utt_path_stem}.npy"

        if utterance_mel_path.exists() and not overwrite:
          logger.info(f"Line {line_nr}: Skipped inference because line is already synthesized!")
          continue

        if include_stats:
          log_out = paragraph_folder / f"{utt_path_stem}.log"
          align_img_path = paragraph_folder / f"{utt_path_stem}-1-alignments.png"
          mel_prepost_img_path = paragraph_folder / f"{utt_path_stem}-2-prepost.png"
          mel_postnet_img_path = paragraph_folder / f"{utt_path_stem}-3-postnet.png"
          comp_img_path = paragraph_folder / f"{utt_path_stem}.png"

          if not overwrite:
            if log_out.exists():
              logger.info(f"Line {line_nr}: Log already exists! Skipped inference.")
              continue

            if mel_postnet_img_path.exists():
              logger.info(f"Line {line_nr}: Mel image already exists! Skipped inference.")
              continue

            if mel_prepost_img_path.exists():
              logger.info(
                f"Line {line_nr}: Mel pre-postnet image already exists! Skipped inference.")
              continue

            if align_img_path.exists():
              logger.info(f"Line {line_nr}: Alignments image already exists! Skipped inference.")
              continue

            if comp_img_path.exists():
              logger.info(f"Line {line_nr}: Comparison image already exists! Skipped inference.")
              continue

        logger.debug(f"Infering {line_nr}...")

        inf_sent_output = synth.infer_v2(
          symbols=utterance,
          speaker=speaker,
          include_stats=include_stats,
          max_decoder_steps=max_decoder_steps,
          seed=seed,
        )

        logger.debug(f"Saving {utterance_mel_path}...")
        paragraph_folder.mkdir(parents=True, exist_ok=True)
        np.save(utterance_mel_path, inf_sent_output.mel_outputs_postnet)

        unknown_symbols |= inf_sent_output.unknown_symbols

        if include_stats:
          log_lines = []
          log_lines.append(f"Timepoint: {datetime.datetime.now()}")
          log_lines.append(
            f"Reached max decoder steps: {inf_sent_output.reached_max_decoder_steps}")
          log_lines.append(f"Inference duration: {inf_sent_output.inference_duration_s}")
          log_lines.append(f"Sampling rate: {inf_sent_output.sampling_rate}")
          log_lines.append(f"Unknown symbols: {' '.join(inf_sent_output.unknown_symbols)}")

          logger.debug(f"Saving {log_out}...")
          log_out.write_text("\n".join(log_lines), encoding="UTF-8")

          logger.debug(f"Saving {mel_postnet_img_path}...")
          _, postnet_img = plot_melspec_np(inf_sent_output.mel_outputs_postnet)
          imageio.imsave(mel_postnet_img_path, postnet_img)

          logger.debug(f"Saving {mel_prepost_img_path}...")
          _, mel_img = plot_melspec_np(inf_sent_output.mel_outputs)
          imageio.imsave(mel_prepost_img_path, mel_img)

          logger.debug(f"Saving {align_img_path}...")
          _, alignments_img = plot_alignment_np_new(inf_sent_output.alignments)
          imageio.imsave(align_img_path, alignments_img)

          logger.debug(f"Saving {comp_img_path}...")
          stack_images_vertically(
            list_im=[
              align_img_path,
              mel_prepost_img_path,
              mel_postnet_img_path,
            ],
            out_path=comp_img_path,
          )
        progress_bar.update()

  if len(unknown_symbols) > 0:
    logger.warning(
      f"Unknown symbols: {' '.join(sorted(unknown_symbols))} (#{len(unknown_symbols)})")
  logger.info(f"Done. Written output to: {output_directory.absolute()}")
  return True
