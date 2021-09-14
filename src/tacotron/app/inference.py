import datetime
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Set

import imageio
import numpy as np
from image_utils import stack_images_horizontally, stack_images_vertically
from tacotron.app.defaults import (DEFAULT_MAX_DECODER_STEPS,
                                   DEFAULT_SAVE_MEL_INFO_COPY_PATH,
                                   DEFAULT_SEED)
from tacotron.app.io import (get_checkpoints_dir, get_inference_root_dir,
                             get_mel_info_dict, get_mel_out_dict,
                             get_train_dir, load_prep_settings)
from tacotron.core import (CheckpointTacotron, InferenceEntries,
                           InferenceEntryOutput)
from tacotron.core import infer as infer_core
from tacotron.utils import (add_console_out_to_logger, add_file_out_to_logger,
                            create_parent_folder,
                            get_custom_or_last_checkpoint, get_default_logger,
                            get_subdir, init_logger, parse_json, save_json)
from tts_preparation import (InferSentence, InferSentenceList,
                             get_infer_sentences)


def get_run_name(input_name: str, iteration: int, speaker_name: str, full_run: bool) -> str:
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S}__text={input_name}__speaker={speaker_name}__it={iteration}__full={full_run}"
  return subdir_name


def get_infer_dir(train_dir: Path, run_name: str) -> None:
  return get_subdir(get_inference_root_dir(train_dir), run_name, create=True)


def load_infer_symbols_map(symbols_map: str) -> List[str]:
  return parse_json(symbols_map)


MEL_PNG = "mel.png"
MEL_POSTNET_PNG = "mel_postnet.png"
ALIGNMENTS_PNG = "alignments.png"


def save_mel_v_plot(infer_dir: Path, sentences: InferSentenceList) -> None:
  paths = [get_infer_sent_dir(infer_dir, get_result_name(x)) / MEL_PNG for x in sentences]
  path = infer_dir / "mel_v.png"
  stack_images_vertically(paths, path)


def save_alignments_v_plot(infer_dir: Path, sentences: InferSentenceList) -> None:
  paths = [get_infer_sent_dir(infer_dir, get_result_name(x)) / ALIGNMENTS_PNG for x in sentences]
  path = infer_dir / "alignments_v.png"
  stack_images_vertically(paths, path)


def save_mel_postnet_v_plot(infer_dir: Path, sentences: InferSentenceList) -> None:
  paths = [get_infer_sent_dir(infer_dir, get_result_name(x)) / MEL_POSTNET_PNG for x in sentences]
  path = infer_dir / "mel_postnet_v.png"
  stack_images_vertically(paths, path)


def save_mel_postnet_h_plot(infer_dir: Path, sentences: InferSentenceList) -> None:
  paths = [get_infer_sent_dir(infer_dir, get_result_name(x)) / MEL_POSTNET_PNG for x in sentences]
  path = infer_dir / "mel_postnet_h.png"
  stack_images_horizontally(paths, path)


def get_infer_sent_dir(infer_dir: Path, result_name: str) -> Path:
  return get_subdir(infer_dir, result_name, create=True)


def save_stats(infer_dir: Path, stats: InferenceEntries) -> None:
  path = infer_dir / "total.csv"
  stats.save(path, header=True)


def get_result_name(entry: InferSentence) -> str:
  return str(entry.sent_id)


def save_results(entry: InferSentence, output: InferenceEntryOutput, infer_dir: Path, mel_postnet_npy_paths: List[Dict[str, Any]]) -> None:
  result_name = get_result_name(entry)
  dest_dir = get_infer_sent_dir(infer_dir, result_name)
  imageio.imsave(dest_dir / MEL_PNG, output.mel_img)
  imageio.imsave(dest_dir / MEL_POSTNET_PNG, output.postnet_img)
  imageio.imsave(dest_dir / ALIGNMENTS_PNG, output.alignments_img)

  mel_postnet_npy_path = dest_dir / "inferred.mel.npy"
  np.save(mel_postnet_npy_path, output.postnet_mel)

  stack_images_vertically(
    list_im=[
      dest_dir / MEL_PNG,
      dest_dir / MEL_POSTNET_PNG,
      dest_dir / ALIGNMENTS_PNG,
    ],
    out_path=dest_dir / "comparison.png"
  )

  mel_info = get_mel_info_dict(
    identifier=result_name,
    path=mel_postnet_npy_path,
    sr=output.sampling_rate,
  )

  mel_postnet_npy_paths.append(mel_info)


def get_infer_log_new(infer_dir: Path) -> None:
  return infer_dir / "log.txt"


def infer(base_dir: Path, train_name: str, text_name: str, speaker: str, sentence_ids: Optional[Set[int]] = None, custom_checkpoint: Optional[int] = None, full_run: bool = True, custom_hparams: Optional[Dict[str, str]] = None, max_decoder_steps: int = DEFAULT_MAX_DECODER_STEPS, seed: int = DEFAULT_SEED, copy_mel_info_to: Optional[str] = DEFAULT_SAVE_MEL_INFO_COPY_PATH) -> None:
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert train_dir.is_dir()

  logger = get_default_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)

  logger.info("Inferring...")

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  taco_checkpoint = CheckpointTacotron.load(checkpoint_path, logger)

  ttsp_dir, merge_name, _ = load_prep_settings(train_dir)
  # merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)

  infer_sents = get_infer_sentences(ttsp_dir, merge_name, text_name)

  run_name = get_run_name(
    input_name=text_name,
    full_run=full_run,
    iteration=iteration,
    speaker_name=speaker,
  )

  infer_dir = get_infer_dir(
    train_dir=train_dir,
    run_name=run_name,
  )

  add_file_out_to_logger(logger, get_infer_log_new(infer_dir))

  mel_postnet_npy_paths: List[Dict[str, Any]] = []
  save_callback = partial(save_results, infer_dir=infer_dir,
                          mel_postnet_npy_paths=mel_postnet_npy_paths)

  inference_results = infer_core(
    checkpoint=taco_checkpoint,
    sentences=infer_sents,
    custom_hparams=custom_hparams,
    full_run=full_run,
    save_callback=save_callback,
    sentence_ids=sentence_ids,
    speaker_name=speaker,
    train_name=train_name,
    max_decoder_steps=max_decoder_steps,
    seed=seed,
    logger=logger,
  )

  logger.info("Creating mel_postnet_v.png")
  save_mel_postnet_v_plot(infer_dir, inference_results)

  logger.info("Creating mel_postnet_h.png")
  save_mel_postnet_h_plot(infer_dir, inference_results)

  logger.info("Creating mel_v.png")
  save_mel_v_plot(infer_dir, inference_results)

  logger.info("Creating alignments_v.png")
  save_alignments_v_plot(infer_dir, inference_results)

  logger.info("Creating total.csv")
  save_stats(infer_dir, inference_results)

  npy_path = save_mel_postnet_npy_paths(
    infer_dir=infer_dir,
    name=run_name,
    mel_postnet_npy_paths=mel_postnet_npy_paths
  )

  logger.info("Wrote all inferred mel paths including sampling rate into these file(s):")
  logger.info(npy_path)

  if copy_mel_info_to is not None:
    create_parent_folder(copy_mel_info_to)
    copyfile(npy_path, copy_mel_info_to)
    logger.info(copy_mel_info_to)

  logger.info(f"Saved output to: {infer_dir}")


def save_mel_postnet_npy_paths(infer_dir: Path, name: str, mel_postnet_npy_paths: List[Dict[str, Any]]) -> str:
  info_json = get_mel_out_dict(
    name=name,
    root_dir=infer_dir,
    mel_info_dict=mel_postnet_npy_paths,
  )

  path = infer_dir / "mel_out.json"
  save_json(path, info_json)
  #text = '\n'.join(mel_postnet_npy_paths)
  #save_txt(path, text)
  return path
