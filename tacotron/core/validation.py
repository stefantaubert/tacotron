import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Optional, Set

import imageio
import numpy as np
from audio_utils.mel import TacotronSTFT, plot_melspec_np
from image_utils import (calculate_structual_similarity_np,
                         make_same_width_by_filling_white)
from mcd import get_mcd_between_mel_spectograms
from scipy.io.wavfile import read
from tacotron.core.synthesizer import Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.globals import MCD_NO_OF_COEFFS_PER_FRAME
from tacotron.utils import (GenericList, cosine_dist_mels, mse_mels,
                            plot_alignment_np)
from text_utils import deserialize_list
from tts_preparation import InferSentence, PreparedData, PreparedDataList


@dataclass
class ValidationEntry():
  timepoint: str
  entry_id: int
  ds_entry_id: int
  train_name: str
  iteration: int
  speaker_name: str
  speaker_id: int
  wav_path: str
  sampling_rate: int
  wav_duration_s: float
  text_original: str
  text: str
  unique_symbols: str
  unique_symbols_count: int
  symbol_count: int
  # grad_norm: float
  # loss: float
  inference_duration_s: float
  reached_max_decoder_steps: bool
  target_frames: int
  predicted_frames: int
  diff_frames: int
  # mcd_dtw_v2: float
  # mcd_dtw_v2_frames: int
  padded_mse: float
  padded_cosine_similarity: Optional[float]
  padded_structural_similarity: Optional[float]
  mfcc_dtw_mcd: float
  mfcc_dtw_penalty: float
  mfcc_dtw_frames: int


class ValidationEntries(GenericList[ValidationEntry]):
  pass


@dataclass
class ValidationEntryOutput():
  wav_orig: np.ndarray
  mel_orig: np.ndarray
  orig_sr: int
  mel_orig_img: np.ndarray
  mel_postnet: np.ndarray
  mel_postnet_sr: int
  mel_postnet_img: np.ndarray
  mel_postnet_diff_img: np.ndarray
  mel_img: np.ndarray
  alignments_img: np.ndarray
  # gate_out_img: np.ndarray


class ValidationEntryOutputs(GenericList[ValidationEntryOutput]):
  pass


def validate(checkpoint: CheckpointTacotron, data: PreparedDataList, custom_hparams: Optional[Dict[str, str]], entry_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, save_callback: Optional[Callable[[PreparedData, ValidationEntryOutput], None]], max_decoder_steps: int, fast: bool, logger: Logger) -> ValidationEntries:
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
      max_decoder_steps=max_decoder_steps,
    )

    symbol_count = len(deserialize_list(entry.serialized_symbol_ids))
    unique_symbols = set(model_symbols.get_symbols(entry.serialized_symbol_ids))
    unique_symbols_str = " ".join(list(sorted(unique_symbols)))
    unique_symbols_count = len(unique_symbols)
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"

    mel_orig: np.ndarray = taco_stft.get_mel_tensor_from_file(entry.wav_path).cpu().numpy()

    target_frames = mel_orig.shape[1]
    predicted_frames = inference_result.mel_outputs_postnet.shape[1]
    diff_frames = predicted_frames - target_frames

    dtw_mcd, dtw_penalty, dtw_frames = get_mcd_between_mel_spectograms(
      mel_1=mel_orig,
      mel_2=inference_result.mel_outputs_postnet,
      n_mfcc=MCD_NO_OF_COEFFS_PER_FRAME,
      take_log=False,
      use_dtw=True,
    )

    cosine_similarity = cosine_dist_mels(mel_orig, inference_result.mel_outputs_postnet)
    mse = mse_mels(mel_orig, inference_result.mel_outputs_postnet)
    structural_similarity = None

    if not fast:
      mel_orig_img_raw, mel_orig_img = plot_melspec_np(mel_orig)
      mel_outputs_postnet_img_raw, mel_outputs_postnet_img = plot_melspec_np(
        inference_result.mel_outputs_postnet)

      mel_orig_img_raw, mel_outputs_postnet_img_raw = make_same_width_by_filling_white(
        img_a=mel_orig_img_raw,
        img_b=mel_outputs_postnet_img_raw,
      )

      structural_similarity, mel_diff_img_raw = calculate_structual_similarity_np(
          img_a=mel_orig_img_raw,
          img_b=mel_outputs_postnet_img_raw,
      )

      # imageio.imsave("/tmp/mel_orig_img_raw.png", mel_orig_img_raw)
      # imageio.imsave("/tmp/mel_outputs_postnet_img_raw.png", mel_outputs_postnet_img_raw)
      # imageio.imsave("/tmp/mel_diff_img_raw.png", mel_diff_img_raw)

    val_entry = ValidationEntry(
      entry_id=entry.entry_id,
      ds_entry_id=entry.ds_entry_id,
      text_original=entry.text_original,
      text=entry.text,
      wav_path=entry.wav_path,
      wav_duration_s=entry.duration,
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
      predicted_frames=predicted_frames,
      target_frames=target_frames,
      diff_frames=diff_frames,
      padded_cosine_similarity=cosine_similarity,
      mfcc_dtw_mcd=dtw_mcd,
      mfcc_dtw_penalty=dtw_penalty,
      mfcc_dtw_frames=dtw_frames,
      padded_structural_similarity=structural_similarity,
      padded_mse=mse,
    )

    validation_entries.append(val_entry)

    # logger.info(val_entry)
    # logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
    # logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
    # logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")

    if not fast:
      orig_sr, orig_wav = read(entry.wav_path)

      mel_orig_img_eq_width, mel_outputs_postnet_img_eq_width = make_same_width_by_filling_white(
        img_a=mel_orig_img,
        img_b=mel_outputs_postnet_img,
      )

      _, mel_diff_img = calculate_structual_similarity_np(
        img_a=mel_orig_img_eq_width,
        img_b=mel_outputs_postnet_img_eq_width,
      )

      alignments_img = plot_alignment_np(inference_result.alignments)
      _, post_mel_img = plot_melspec_np(inference_result.mel_outputs_postnet)

      validation_entry_output = ValidationEntryOutput(
        wav_orig=orig_wav,
        mel_orig=mel_orig,
        orig_sr=orig_sr,
        mel_postnet=inference_result.mel_outputs_postnet,
        mel_postnet_sr=inference_result.sampling_rate,
        mel_orig_img=mel_orig_img,
        mel_postnet_img=mel_outputs_postnet_img,
        mel_postnet_diff_img=mel_diff_img,
        alignments_img=alignments_img,
        mel_img=post_mel_img,
      )

      assert save_callback is not None

      save_callback(entry, validation_entry_output)

    logger.info(f"MFCC MCD DTW: {val_entry.mfcc_dtw_mcd}")

  return validation_entries
