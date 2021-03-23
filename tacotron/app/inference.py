import datetime
import os
from functools import partial
from typing import Dict, List, Optional, Set

import imageio
from audio_utils import float_to_wav
from image_utils import stack_images_horizontally, stack_images_vertically
from tacotron.app.defaults import (DEFAULT_DENOISER_STRENGTH,
                                   DEFAULT_SENTENCE_PAUSE_S, DEFAULT_SIGMA,
                                   DEFAULT_WAVEGLOW)
from tacotron.app.inference import get_infer_sentences
from tacotron.app.io import (get_checkpoints_dir, get_inference_root_dir,
                             get_train_dir, load_prep_settings)
from tacotron.core.inference.infer import (InferenceEntries,
                                           InferenceEntryOutput, infer2)
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import (get_custom_or_last_checkpoint, get_last_checkpoint,
                            get_subdir, parse_json)
from tts_preparation import InferSentence, InferSentenceList, get_merged_dir


def get_infer_dir(train_dir: str, input_name: str, iteration: int, speaker_name: str, full_run: bool):
  subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__text={input_name}__speaker={speaker_name}__it={iteration}__full={full_run}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


def load_infer_symbols_map(symbols_map: str) -> List[str]:
  return parse_json(symbols_map)


def save_infer_v_pre_post(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), "postnet.png") for x in sentences]
  path = os.path.join(infer_dir, "postnet_v.png")
  stack_images_vertically(paths, path)


def save_infer_v_alignments(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), "alignments.png") for x in sentences]
  path = os.path.join(infer_dir, "alignments_v.png")
  stack_images_vertically(paths, path)


def save_infer_v_plot(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), "output.png") for x in sentences]
  path = os.path.join(infer_dir, "complete_v.png")
  stack_images_vertically(paths, path)


def save_infer_h_plot(infer_dir: str, sentences: InferSentenceList):
  paths = [os.path.join(get_infer_sent_dir(infer_dir, x), "output.png") for x in sentences]
  path = os.path.join(infer_dir, "complete_h.png")
  stack_images_horizontally(paths, path)


def get_infer_sent_dir(infer_dir: str, entry: InferSentence) -> None:
  return get_subdir(infer_dir, str(entry.sent_id), create=True)


def save_stats(infer_dir: str, stats: InferenceEntries) -> None:
  path = os.path.join(infer_dir, "total.csv")
  stats.save(path, header=True)


def save_results(entry: InferSentence, output: InferenceEntryOutput, infer_dir: str):
  dest_dir = get_infer_sent_dir(infer_dir, entry)
  imageio.imsave(os.path.join(dest_dir, "output.png"), output.inferred_wav_img)
  imageio.imsave(os.path.join(dest_dir, "postnet.png"), output.postnet_img)
  imageio.imsave(os.path.join(dest_dir, "alignments.png"), output.alignments_img)
  float_to_wav(
    path=os.path.join(dest_dir, "output.wav"),
    wav=output.inferred_wav,
    sample_rate=output.inferred_wav_sr,
  )
  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, "output.png"),
      os.path.join(dest_dir, "postnet.png"),
      os.path.join(dest_dir, "alignments.png"),
    ],
    out_path=os.path.join(dest_dir, "comparison.png")
  )


def get_infer_log_new(infer_dir: str):
  return os.path.join(infer_dir, "log.txt")


def infer_main2(base_dir: str, train_name: str, text_name: str, speaker: str, sentence_ids: Optional[Set[int]] = None, waveglow: str = DEFAULT_WAVEGLOW, custom_checkpoint: Optional[int] = None, sentence_pause_s: float = DEFAULT_SENTENCE_PAUSE_S, sigma: float = DEFAULT_SIGMA, full_run: bool = True, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, custom_tacotron_hparams: Optional[Dict[str, str]] = None, custom_waveglow_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_default_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)

  logger.info("Inferring...")

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  taco_checkpoint = CheckpointTacotron.load(checkpoint_path, logger)

  merge_name, _ = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)

  infer_sents = get_infer_sentences(base_dir, merge_dir, text_name)

  infer_dir = get_infer_dir(
    train_dir=train_dir,
    input_name=text_name,
    full_run=full_run,
    iteration=iteration,
    speaker_name=speaker,
  )

  add_file_out_to_logger(logger, get_infer_log_new(infer_dir))

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))
  wg_checkpoint = CheckpointWaveglow.load(wg_checkpoint_path, logger)
  save_callback = partial(save_results, infer_dir=infer_dir)

  wav, inference_results = infer2(
    tacotron_checkpoint=taco_checkpoint,
    waveglow_checkpoint=wg_checkpoint,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sentences=infer_sents,
    custom_taco_hparams=custom_tacotron_hparams,
    custom_wg_hparams=custom_waveglow_hparams,
    logger=logger,
    full_run=full_run,
    save_callback=save_callback,
    sentence_ids=sentence_ids,
    speaker_name=speaker,
    train_name=train_name,
  )

  float_to_wav(
    path=os.path.join(infer_dir, "complete.wav"),
    wav=wav,
    sample_rate=inference_results[0].sampling_rate
  )

  logger.info("Creating complete_v.png")
  save_infer_v_plot(infer_dir, inference_results)
  logger.info("Creating complete_h.png")
  save_infer_h_plot(infer_dir, inference_results)
  logger.info("Creating postnet_v.png")
  save_infer_v_pre_post(infer_dir, inference_results)
  logger.info("Creating alignments_v.png")
  save_infer_v_alignments(infer_dir, inference_results)
  logger.info("Creating total.csv")
  save_stats(infer_dir, inference_results)
  logger.info(f"Saved output to: {infer_dir}")
