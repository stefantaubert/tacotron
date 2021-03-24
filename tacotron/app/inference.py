import datetime
import os
from functools import partial
from typing import Dict, List, Optional, Set

import imageio
import numpy as np
from image_utils import stack_images_horizontally, stack_images_vertically
from tacotron.app.io import (get_checkpoints_dir, get_inference_root_dir,
                             get_train_dir, load_prep_settings)
from tacotron.core.inference import (InferenceEntries, InferenceEntryOutput,
                                     infer)
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import (add_console_out_to_logger, add_file_out_to_logger,
                            get_custom_or_last_checkpoint, get_default_logger,
                            get_subdir, init_logger, parse_json)
from tts_preparation import (InferSentence, InferSentenceList,
                             get_infer_sentences)


def get_infer_dir(train_dir: str, input_name: str, iteration: int, speaker_name: str, full_run: bool):
  subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__text={input_name}__speaker={speaker_name}__it={iteration}__full={full_run}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


def load_infer_symbols_map(symbols_map: str) -> List[str]:
  return parse_json(symbols_map)


MEL_PNG = "mel.png"
MEL_POSTNET_PNG = "mel_postnet.png"
ALIGNMENTS_PNG = "alignments.png"


def save_mel_v_plot(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), MEL_PNG) for x in sentences]
  path = os.path.join(infer_dir, "mel_v.png")
  stack_images_vertically(paths, path)


def save_alignments_v_plot(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), ALIGNMENTS_PNG) for x in sentences]
  path = os.path.join(infer_dir, "alignments_v.png")
  stack_images_vertically(paths, path)


def save_mel_postnet_v_plot(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), MEL_POSTNET_PNG) for x in sentences]
  path = os.path.join(infer_dir, "mel_postnet_v.png")
  stack_images_vertically(paths, path)


def save_mel_postnet_h_plot(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), MEL_POSTNET_PNG) for x in sentences]
  path = os.path.join(infer_dir, "mel_postnet_h.png")
  stack_images_horizontally(paths, path)


def get_infer_sent_dir(infer_dir: str, entry: InferSentence) -> None:
  return get_subdir(infer_dir, str(entry.sent_id), create=True)


def save_stats(infer_dir: str, stats: InferenceEntries) -> None:
  path = os.path.join(infer_dir, "total.csv")
  stats.save(path, header=True)


def save_results(entry: InferSentence, output: InferenceEntryOutput, infer_dir: str):
  dest_dir = get_infer_sent_dir(infer_dir, entry)
  imageio.imsave(os.path.join(dest_dir, MEL_PNG), output.mel_img)
  imageio.imsave(os.path.join(dest_dir, MEL_POSTNET_PNG), output.postnet_img)
  imageio.imsave(os.path.join(dest_dir, ALIGNMENTS_PNG), output.alignments_img)
  np.save(os.path.join(dest_dir, "inferred.mel.npy"), output.postnet_mel)

  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, MEL_PNG),
      os.path.join(dest_dir, MEL_POSTNET_PNG),
      os.path.join(dest_dir, ALIGNMENTS_PNG),
    ],
    out_path=os.path.join(dest_dir, "comparison.png")
  )


def get_infer_log_new(infer_dir: str):
  return os.path.join(infer_dir, "log.txt")


def app_infer(base_dir: str, train_name: str, text_name: str, speaker: str, sentence_ids: Optional[Set[int]] = None, custom_checkpoint: Optional[int] = None, full_run: bool = True, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

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

  infer_dir = get_infer_dir(
    train_dir=train_dir,
    input_name=text_name,
    full_run=full_run,
    iteration=iteration,
    speaker_name=speaker,
  )

  add_file_out_to_logger(logger, get_infer_log_new(infer_dir))

  save_callback = partial(save_results, infer_dir=infer_dir)

  inference_results = infer(
    checkpoint=taco_checkpoint,
    sentences=infer_sents,
    custom_hparams=custom_hparams,
    full_run=full_run,
    save_callback=save_callback,
    sentence_ids=sentence_ids,
    speaker_name=speaker,
    train_name=train_name,
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

  logger.info(f"Saved output to: {infer_dir}")
