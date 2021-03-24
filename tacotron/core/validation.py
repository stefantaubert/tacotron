import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Optional, Set

import imageio
import numpy as np
from audio_utils.mel import TacotronSTFT, plot_melspec_np
from image_utils import (calculate_structual_similarity_np,
                         make_same_width_by_filling_white)
from mcd import (get_audio_and_sampling_rate_from_path,
                 get_mcd_dtw_from_mel_spectograms)
from tacotron.core.synthesizer import Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.globals import MCD_NO_OF_COEFFS_PER_FRAME
from tacotron.utils import GenericList, cosine_dist_mels, plot_alignment_np
from text_utils import deserialize_list
from tts_preparation import InferSentence, PreparedData, PreparedDataList


@dataclass
class ValidationEntry():
  entry_id: int = None
  ds_entry_id: int = None
  text_original: str = None
  text: str = None
  wav_path: str = None
  original_duration_s: float = None
  speaker_id: int = None
  speaker_name: str = None
  iteration: int = None
  reached_max_decoder_steps: bool = None
  inference_duration_s: float = None
  unique_symbols: str = None
  unique_symbols_count: int = None
  symbol_count: int = None
  # grad_norm: float = None
  # loss: float = None
  mcd_dtw: float = None
  mcd_dtw_frames: int = None
  # mcd_dtw_v2: float = None
  # mcd_dtw_v2_frames: int = None
  structural_similarity: float = None
  cosine_similarity: float = None
  timepoint: str = None
  train_name: str = None
  sampling_rate: int = None


class ValidationEntries(GenericList[ValidationEntry]):
  pass


@dataclass
class ValidationEntryOutput():
  mel_orig: np.ndarray = None
  mel_orig_sr: int = None
  mel_orig_img: np.ndarray = None
  mel_postnet: np.ndarray = None
  mel_postnet_sr: int = None
  mel_postnet_img: np.ndarray = None
  mel_img: np.ndarray = None
  alignments_img: np.ndarray = None
  mel_diff_img: np.ndarray = None
  # gate_out_img: np.ndarray = None


class ValidationEntryOutputs(GenericList[ValidationEntryOutput]):
  pass


def validate(checkpoint: CheckpointTacotron, data: PreparedDataList, custom_hparams: Optional[Dict[str, str]], entry_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, save_callback: Callable[[PreparedData, ValidationEntryOutput], None], logger: Logger) -> ValidationEntries:
  model_symbols = checkpoint.get_symbols()
  model_accents = checkpoint.get_accents()
  model_speakers = checkpoint.get_speakers()
  validation_entries = ValidationEntries()

  if full_run:
    entries = data
  else:
    speaker_id: Optional[int] = None
    if speaker_name is not None:
      speaker_id = model_speakers.get_id(speaker_name)
    entries = PreparedDataList(data.get_for_validation(entry_ids, speaker_id))

  if len(entries) == 0:
    logger.info("Nothing to synthesize!")
    return validation_entries

  synth = Synthesizer(
      checkpoint=checkpoint,
      custom_hparams=custom_hparams,
      logger=logger,
  )

  # criterion = Tacotron2Loss()

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)

  for entry in entries.items(True):
    infer_sent = InferSentence(
      sent_id=1,
      symbols=model_symbols.get_symbols(entry.serialized_symbol_ids),
      accents=model_accents.get_accents(entry.serialized_accent_ids),
      original_text=entry.text_original,
    )

    speaker_name = model_speakers.get_speaker(entry.speaker_id)
    inference_result = synth.infer(
      sentence=infer_sent,
      speaker=speaker_name,
      ignore_unknown_symbols=False,
    )

    symbol_count = len(deserialize_list(entry.serialized_symbol_ids))
    unique_symbols = set(model_symbols.get_symbols(entry.serialized_symbol_ids))
    unique_symbols_str = " ".join(list(sorted(unique_symbols)))
    unique_symbols_count = len(unique_symbols)
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"

    val_entry = ValidationEntry(
      entry_id=entry.entry_id,
      ds_entry_id=entry.ds_entry_id,
      text_original=entry.text_original,
      text=entry.text,
      wav_path=entry.wav_path,
      original_duration_s=entry.duration,
      speaker_id=entry.speaker_id,
      speaker_name=speaker_name,
      iteration=checkpoint.iteration,
      unique_symbols=unique_symbols_str,
      unique_symbols_count=unique_symbols_count,
      symbol_count=symbol_count,
      timepoint=timepoint,
      train_name=train_name,
      sampling_rate=synth.get_sampling_rate(),
      reached_max_decoder_steps=inference_result.reached_max_decoder_steps,
      inference_duration_s=inference_result.inference_duration_s,
    )

    _, orig_sr = get_audio_and_sampling_rate_from_path(entry.wav_path)
    mel_orig = taco_stft.get_mel_tensor_from_file(entry.wav_path).cpu().numpy()

    validation_entry_output = ValidationEntryOutput(
      mel_orig=mel_orig,
      mel_orig_sr=orig_sr,
      mel_postnet=inference_result.mel_outputs_postnet,
      mel_postnet_sr=inference_result.sampling_rate,
    )

    mcd, frames = get_mcd_dtw_from_mel_spectograms(
      mel_spectogram_1=mel_orig,
      mel_spectogram_2=inference_result.mel_outputs_postnet, no_of_coeffs_per_frame=MCD_NO_OF_COEFFS_PER_FRAME
    )

    val_entry.mcd_dtw = mcd
    val_entry.mcd_dtw_frames = frames

    cosine_similarity = cosine_dist_mels(mel_orig, inference_result.mel_outputs_postnet)
    val_entry.cosine_similarity = cosine_similarity

    mel_orig_img_raw, mel_orig_img = plot_melspec_np(mel_orig)
    mel_outputs_postnet_img_raw, mel_outputs_postnet_img = plot_melspec_np(
      inference_result.mel_outputs_postnet)

    validation_entry_output.mel_orig_img = mel_orig_img
    validation_entry_output.mel_postnet_img = mel_outputs_postnet_img

    imageio.imsave("/tmp/mel_orig_img_raw.png", mel_orig_img_raw)
    imageio.imsave("/tmp/mel_outputs_postnet_img_raw.png", mel_outputs_postnet_img_raw)

    mel_orig_img_raw, mel_outputs_postnet_img_raw = make_same_width_by_filling_white(
      img_a=mel_orig_img_raw,
      img_b=mel_outputs_postnet_img_raw,
    )

    structural_similarity, mel_diff_img_raw = calculate_structual_similarity_np(
        img_a=mel_orig_img_raw,
        img_b=mel_outputs_postnet_img_raw,
    )
    val_entry.structural_similarity = structural_similarity
    imageio.imsave("/tmp/mel_diff_img_raw.png", mel_diff_img_raw)

    mel_orig_img, mel_outputs_postnet_img = make_same_width_by_filling_white(
      img_a=mel_orig_img,
      img_b=mel_outputs_postnet_img,
    )

    _, mel_diff_img = calculate_structual_similarity_np(
        img_a=mel_orig_img,
        img_b=mel_outputs_postnet_img,
    )

    validation_entry_output.mel_diff_img = mel_diff_img

    alignments_img = plot_alignment_np(inference_result.alignments)
    validation_entry_output.alignments_img = alignments_img
    _, post_mel_img = plot_melspec_np(inference_result.mel_outputs_postnet)

    # validation_entry_output.gate_out_img = None
    validation_entry_output.mel_img = post_mel_img
    # val_entry.grad_norm = None
    # val_entry.loss = None

    # logger.info(val_entry)
    logger.info(f"MCD DTW: {val_entry.mcd_dtw}")
    # logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
    logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
    logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")
    save_callback(entry, validation_entry_output)
    validation_entries.append(val_entry)
  return validation_entries
