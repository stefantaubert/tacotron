import datetime
import os
from functools import partial
from typing import Dict, List, Optional, Set

import imageio
from audio_utils.audio import float_to_wav
from src.app.io import (_get_validation_root_dir, get_checkpoints_dir,
                        get_val_log, load_prep_settings)
from src.app.pre.merge_ds import get_merged_dir
from src.app.pre.prepare import get_prep_dir, load_testset, load_valset
from src.app.tacotron.defaults import (DEFAULT_DENOISER_STRENGTH,
                                       DEFAULT_SIGMA, DEFAULT_WAVEGLOW)
from src.app.tacotron.io import get_train_dir
from src.app.utils import prepare_logger
from src.app.waveglow.training import get_train_dir as get_wg_train_dir
from src.core.common.train import (get_all_checkpoint_iterations,
                                   get_checkpoint,
                                   get_custom_or_last_checkpoint,
                                   get_last_checkpoint)
from src.core.common.utils import get_subdir, stack_images_vertically
from src.core.inference.validation import (ValidationEntries,
                                           ValidationEntryOutput)
from src.core.inference.validation import validate as validate_new
from src.core.pre.prep.data import PreparedData
from tacotron.core.training import CheckpointTacotron
from src.core.waveglow.train import CheckpointWaveglow
from tqdm import tqdm


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


def get_val_dir(train_dir: str, ds: str, iterations: Set[int], full_run: bool, entry_ids: Optional[Set[int]], speaker: Optional[str]):
  if len(iterations) > 3:
    its = ",".join(str(x) for x in sorted(iterations)[:3]) + ",..."
  else:
    its = ",".join(str(x) for x in sorted(iterations))

  subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__ds={ds}__entries={get_repr_entries(entry_ids)}__speaker={get_repr_speaker(speaker)}__its={its}__full={full_run}"
  return get_subdir(_get_validation_root_dir(train_dir), subdir_name, create=True)


# def get_val_dir_new(train_dir: str):
#   subdir_name = f"{datetime.datetime.now():%Y-%m-%d__%H-%M-%S}"
#   return get_subdir(_get_validation_root_dir(train_dir), subdir_name, create=True)


def get_val_entry_dir(val_dir: str, entry: PreparedData, iteration: int) -> None:
  return get_subdir(val_dir, f"it={iteration}_id={entry.entry_id}", create=True)


def save_stats(val_dir: str, validation_entries: ValidationEntries) -> None:
  path = os.path.join(val_dir, "total.csv")
  validation_entries.save(path, header=True)


def save_results(entry: PreparedData, output: ValidationEntryOutput, val_dir: str, iteration: int):
  dest_dir = get_val_entry_dir(val_dir, entry, iteration)
  imageio.imsave(os.path.join(dest_dir, "original.png"), output.orig_wav_img)
  imageio.imsave(os.path.join(dest_dir, "inferred.png"), output.inferred_wav_img)
  imageio.imsave(os.path.join(dest_dir, "postnet.png"), output.postnet_img)
  imageio.imsave(os.path.join(dest_dir, "alignments.png"), output.alignments_img)
  imageio.imsave(os.path.join(dest_dir, "diff.png"), output.struc_sim_img)
  float_to_wav(
    path=os.path.join(dest_dir, "original.wav"),
    wav=output.orig_wav,
    sample_rate=output.orig_wav_sr,
  )
  float_to_wav(
    path=os.path.join(dest_dir, "inferred.wav"),
    wav=output.inferred_wav,
    sample_rate=output.inferred_wav_sr,
  )
  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, "original.png"),
      os.path.join(dest_dir, "inferred.png"),
      os.path.join(dest_dir, "diff.png"),
      os.path.join(dest_dir, "postnet.png"),
      os.path.join(dest_dir, "alignments.png"),
    ],
    out_path=os.path.join(dest_dir, "comparison.png")
  )


def app_validate(base_dir: str, train_name: str, waveglow: str = DEFAULT_WAVEGLOW, entry_ids: Optional[Set[int]] = None, speaker: Optional[str] = None, ds: str = "val", custom_checkpoints: Optional[Set[int]] = None, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, custom_tacotron_hparams: Optional[Dict[str, str]] = None, full_run: bool = False, custom_waveglow_hparams: Optional[Dict[str, str]] = None):
  """Param: custom checkpoints: empty => all; None => random; ids"""

  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
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

  val_dir = get_val_dir(
    train_dir=train_dir,
    ds=ds,
    entry_ids=entry_ids,
    full_run=full_run,
    iterations=iterations,
    speaker=speaker,
  )

  val_log_path = os.path.join(val_dir, "log.txt")
  logger = prepare_logger(val_log_path)
  logger.info("Validating...")
  logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

  result = ValidationEntries()

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))
  wg_checkpoint = CheckpointWaveglow.load(wg_checkpoint_path, logger)

  for iteration in tqdm(sorted(iterations)):
    logger.info(f"Current checkpoint: {iteration}")
    checkpoint_path = get_checkpoint(checkpoint_dir, iteration)
    taco_checkpoint = CheckpointTacotron.load(checkpoint_path, logger)
    save_callback = partial(save_results, val_dir=val_dir, iteration=iteration)

    validation_entries = validate_new(
      tacotron_checkpoint=taco_checkpoint,
      waveglow_checkpoint=wg_checkpoint,
      sigma=sigma,
      denoiser_strength=denoiser_strength,
      data=data,
      logger=logger,
      custom_taco_hparams=custom_tacotron_hparams,
      custom_wg_hparams=custom_waveglow_hparams,
      entry_ids=entry_ids,
      full_run=full_run,
      speaker_name=speaker,
      train_name=train_name,
      save_callback=save_callback,
    )

    result.extend(validation_entries)
  if len(result) > 0:
    save_stats(val_dir, result)
    logger.info(f"Saved output to: {val_dir}")
