import datetime
import os
from functools import partial
from logging import getLogger
from shutil import copyfile
from typing import Any, Dict, List, Optional, Set, Tuple

import imageio
import numpy as np
from image_utils import stack_images_vertically
from scipy.io.wavfile import write
from tacotron.app.defaults import (DEFAULT_MAX_DECODER_STEPS,
                                   DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME,
                                   DEFAULT_MEL_INFO_COPY_PATH,
                                   DEFAULT_REPETITIONS, DEFAULT_SEED)
from tacotron.app.io import (_get_validation_root_dir, get_checkpoints_dir,
                             get_mel_info_dict, get_mel_out_dict,
                             get_train_dir, load_prep_settings)
from tacotron.core import (CheckpointTacotron, ValidationEntries,
                           ValidationEntryOutput)
from tacotron.core import validate as validate_core
from tacotron.utils import (create_parent_folder,
                            get_all_checkpoint_iterations, get_checkpoint,
                            get_last_checkpoint, get_subdir, pass_lines_list,
                            prepare_logger, save_json)
from tqdm import tqdm
from tts_preparation import (PreparedData, get_merged_dir, get_prep_dir,
                             load_testset, load_trainset, load_valset)


def get_repr_entries(entry_ids: Optional[Set[int]]):
  if entry_ids is None:
    return "none"
  if len(entry_ids) == 0:
    return "empty"
  return ",".join(list(sorted(map(str, entry_ids))))


def get_repr_speaker(speaker: Optional[str]):
  if speaker is None:
    return "none"
  return speaker


def get_run_name(ds: str, iterations: Set[int], full_run: bool, entry_ids: Optional[Set[int]], speaker: Optional[str]) -> str:
  if len(iterations) > 3:
    its = ",".join(str(x) for x in sorted(iterations)[:3]) + ",..."
  else:
    its = ",".join(str(x) for x in sorted(iterations))

  subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__ds={ds}__entries={get_repr_entries(entry_ids)}__speaker={get_repr_speaker(speaker)}__its={its}__full={full_run}"
  return subdir_name


def get_val_dir(train_dir: str, run_name: str) -> str:
  return get_subdir(_get_validation_root_dir(train_dir), run_name, create=True)


# def get_val_dir_new(train_dir: str):
#   subdir_name = f"{datetime.datetime.now():%Y-%m-%d__%H-%M-%S}"
#   return get_subdir(_get_validation_root_dir(train_dir), subdir_name, create=True)


def get_val_entry_dir(val_dir: str, result_name: str) -> None:
  return get_subdir(val_dir, result_name, create=True)


def save_stats(val_dir: str, validation_entries: ValidationEntries) -> None:
  path = os.path.join(val_dir, "total.csv")
  validation_entries.save(path, header=True)


def save_mel_postnet_npy_paths(val_dir: str, name: str, mel_postnet_npy_paths: List[Dict[str, Any]]) -> str:
  info_json = get_mel_out_dict(
    name=name,
    mel_info_dict=mel_postnet_npy_paths,
  )

  path = os.path.join(val_dir, "mel_postnet_npy.json")
  save_json(path, info_json)
  # text = '\n'.join(mel_postnet_npy_paths)
  # save_txt(path, text)
  return path


def get_result_name(entry: PreparedData, iteration: int, repetition: int):
  return f"it={iteration}_id={entry.entry_id}_rep={repetition}"


def save_results(entry: PreparedData, output: ValidationEntryOutput, val_dir: str, iteration: int, mel_postnet_npy_paths: List[Dict[str, Any]]):
  result_name = get_result_name(entry, iteration, output.repetition)
  dest_dir = get_val_entry_dir(val_dir, result_name)
  write(os.path.join(dest_dir, "original.wav"), output.orig_sr, output.wav_orig)
  imageio.imsave(os.path.join(dest_dir, "original.png"), output.mel_orig_img)
  imageio.imsave(os.path.join(dest_dir, "original_aligned.png"), output.mel_orig_aligned_img)
  imageio.imsave(os.path.join(dest_dir, "inferred.png"), output.mel_postnet_img)
  imageio.imsave(os.path.join(dest_dir, "inferred_aligned.png"), output.mel_postnet_aligned_img)
  imageio.imsave(os.path.join(dest_dir, "mel.png"), output.mel_img)
  imageio.imsave(os.path.join(dest_dir, "alignments.png"), output.alignments_img)
  imageio.imsave(os.path.join(dest_dir, "alignments_aligned.png"), output.alignments_aligned_img)
  imageio.imsave(os.path.join(dest_dir, "diff.png"), output.mel_postnet_diff_img)
  imageio.imsave(os.path.join(dest_dir, "diff_aligned.png"), output.mel_postnet_aligned_diff_img)
  np.save(os.path.join(dest_dir, "original.mel.npy"), output.mel_orig)
  np.save(os.path.join(dest_dir, "original_aligned.mel.npy"), output.mel_orig_aligned)

  mel_postnet_npy_path = os.path.join(dest_dir, "inferred.mel.npy")
  np.save(mel_postnet_npy_path, output.mel_postnet)
  np.save(os.path.join(dest_dir, "inferred_aligned.mel.npy"), output.mel_postnet_aligned)

  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, "original.png"),
      os.path.join(dest_dir, "inferred.png"),
      os.path.join(dest_dir, "diff.png"),
      os.path.join(dest_dir, "alignments.png"),
      os.path.join(dest_dir, "mel.png"),
    ],
    out_path=os.path.join(dest_dir, "comparison.png")
  )

  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, "original.png"),
      os.path.join(dest_dir, "inferred.png"),
      os.path.join(dest_dir, "original_aligned.png"),
      os.path.join(dest_dir, "inferred_aligned.png"),
      os.path.join(dest_dir, "diff_aligned.png"),
      os.path.join(dest_dir, "alignments_aligned.png"),
    ],
    out_path=os.path.join(dest_dir, "comparison_aligned.png")
  )

  mel_info = get_mel_info_dict(
    identifier=result_name,
    path=mel_postnet_npy_path,
    sr=output.mel_postnet_sr,
  )

  mel_postnet_npy_paths.append(mel_info)


def validate(base_dir: str, train_name: str, entry_ids: Optional[Set[int]] = None, entry_ids_w_seed: Optional[List[Tuple[int, int]]] = None, speaker: Optional[str] = None, ds: str = "val", custom_checkpoints: Optional[Set[int]] = None, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False, max_decoder_steps: int = DEFAULT_MAX_DECODER_STEPS, mcd_no_of_coeffs_per_frame: int = DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME, copy_mel_info_to: Optional[str] = DEFAULT_MEL_INFO_COPY_PATH, fast: bool = False, repetitions: int = DEFAULT_REPETITIONS, seed: int = DEFAULT_SEED) -> None:
  """Param: custom checkpoints: empty => all; None => random; ids"""
  assert repetitions > 0

  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  ttsp_dir, merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)

  if ds == "val":
    data = load_valset(prep_dir)
  elif ds == "test":
    data = load_testset(prep_dir)
  else:
    assert False

  iterations: Set[int] = set()
  checkpoint_dir = get_checkpoints_dir(train_dir)

  if custom_checkpoints is None:
    _, last_it = get_last_checkpoint(checkpoint_dir)
    iterations.add(last_it)
  else:
    if len(custom_checkpoints) == 0:
      iterations = set(get_all_checkpoint_iterations(checkpoint_dir))
    else:
      iterations = custom_checkpoints

  run_name = get_run_name(
    ds=ds,
    entry_ids=entry_ids,
    full_run=full_run,
    iterations=iterations,
    speaker=speaker,
  )

  val_dir = get_val_dir(
    train_dir=train_dir,
    run_name=run_name,
  )

  val_log_path = os.path.join(val_dir, "log.txt")
  logger = prepare_logger(val_log_path)
  logger.info("Validating...")
  logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

  result = ValidationEntries()
  save_callback = None
  trainset = load_trainset(prep_dir)

  for iteration in tqdm(sorted(iterations)):
    mel_postnet_npy_paths: List[str] = []
    logger.info(f"Current checkpoint: {iteration}")
    checkpoint_path = get_checkpoint(checkpoint_dir, iteration)
    taco_checkpoint = CheckpointTacotron.load(checkpoint_path, logger)
    if not fast:
      save_callback = partial(save_results, val_dir=val_dir, iteration=iteration,
                              mel_postnet_npy_paths=mel_postnet_npy_paths)

    validation_entries = validate_core(
      checkpoint=taco_checkpoint,
      data=data,
      trainset=trainset,
      custom_hparams=custom_hparams,
      entry_ids=entry_ids,
      entry_ids_w_seed=entry_ids_w_seed,
      full_run=full_run,
      speaker_name=speaker,
      train_name=train_name,
      logger=logger,
      max_decoder_steps=max_decoder_steps,
      fast=fast,
      save_callback=save_callback,
      mcd_no_of_coeffs_per_frame=mcd_no_of_coeffs_per_frame,
      repetitions=repetitions,
      seed=seed,
    )

    result.extend(validation_entries)

  if len(result) == 0:
    return

  save_stats(val_dir, result)

  if not fast:
    logger.info("Wrote all inferred mel paths including sampling rate into these file(s):")
    npy_path = save_mel_postnet_npy_paths(
      val_dir=val_dir,
      name=run_name,
      mel_postnet_npy_paths=mel_postnet_npy_paths
    )
    logger.info(npy_path)

    if copy_mel_info_to is not None:
      create_parent_folder(copy_mel_info_to)
      copyfile(npy_path, copy_mel_info_to)
      logger.info(copy_mel_info_to)

  logger.info(f"Saved output to: {val_dir}")
