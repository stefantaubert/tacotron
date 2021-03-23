import datetime
import logging
import math
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional, Set, Tuple

import imageio
import numpy as np
import torch
from audio_utils import concatenate_audios, normalize_wav
from audio_utils.audio import (convert_wav, float_to_wav, get_duration_s,
                               wav_to_float32)
from audio_utils.mel import TacotronSTFT, mel_to_numpy, plot_melspec_np
from image_utils import calculate_structual_similarity
from image_utils.main import make_same_width_by_filling_white
from tts_preparation import PreparedDataList
from mcd import (get_audio_and_sampling_rate_from_path,
                 get_mcd_dtw_from_mel_spectograms, get_mel_spectogram,
                 get_spectogram)
from tacotron.core.synthesizer import Synthesizer as TacoSynthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import (GenericList, cosine_dist_mels, pass_lines,
                            plot_alignment_np, stack_images_vertically)
from tts_preparation import InferSentence, InferSentenceList


@dataclass
class MCDSettings():
  n_fft: int = 1024
  hop_length: int = 256
  n_mels: int = 80
  no_of_coeffs_per_frame: int = 16


@dataclass
class ValidationEntry():
  entry_id: int = None
  ds_entry_id: int = None
  text_original: str = None
  text: str = None
  wav_path: str = None
  original_duration_s: float = None
  inferred_duration_s: float = None
  diff_duration_s: float = None
  speaker_id: int = None
  speaker_name: str = None
  wg_iteration: int = None
  taco_iteration: int = None
  reached_max_decoder_steps: bool = None
  taco_inference_duration_s: float = None
  wg_inference_duration_s: float = None
  total_inference_duration_s: float = None
  shards: float = None
  unique_symbol_ids: str = None
  unique_symbol_ids_count: int = None
  symbol_count: int = None
  # grad_norm: float = None
  # loss: float = None
  mcd_dtw: float = None
  mcd_dtw_frames: int = None
  mcd_dtw_v2: float = None
  mcd_dtw_v2_frames: int = None
  structural_similarity: float = None
  cosine_similarity: float = None
  timepoint: str = None
  train_name: str = None
  sampling_rate: int = None


class ValidationEntries(GenericList[ValidationEntry]):
  pass


@dataclass
class ValidationEntryOutput():
  orig_wav: np.ndarray = None
  orig_wav_sr: int = None
  orig_wav_img: np.ndarray = None
  inferred_wav: np.ndarray = None
  inferred_wav_sr: int = None
  inferred_wav_img: np.ndarray = None
  struc_sim_img: np.ndarray = None
  alignments_img: np.ndarray = None
  postnet_img: np.ndarray = None
  # gate_out_img: np.ndarray = None


class ValidationEntryOutputs(GenericList[ValidationEntryOutput]):
  pass


def validate(tacotron_checkpoint: CheckpointTacotron, waveglow_checkpoint: CheckpointWaveglow, data: PreparedDataList, denoiser_strength: float, sigma: float, custom_taco_hparams: Optional[Dict[str, str]], custom_wg_hparams: Optional[Dict[str, str]], entry_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, save_callback: Callable[[PreparedData, ValidationEntryOutput], None], logger: Logger) -> ValidationEntries:
  model_symbols = tacotron_checkpoint.get_symbols()
  model_accents = tacotron_checkpoint.get_accents()
  model_speakers = tacotron_checkpoint.get_speakers()
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
    tacotron_checkpoint,
    waveglow_checkpoint,
    logger=logger,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )
  shard_size = get_shard_size(model_symbols)
  # criterion = Tacotron2Loss()

  taco_stft = TacotronSTFT(synth._wg_synt.hparams, logger=logger)

  mcd_settings = MCDSettings()

  for entry in entries.items(True):
    speaker_name = model_speakers.get_speaker(entry.speaker_id)
    infer_sent = InferSentence(
      sent_id=1,
      symbols=model_symbols.get_symbols(entry.serialized_symbol_ids),
      accents=model_accents.get_accents(entry.serialized_accent_ids),
      original_text=entry.text_original,
    )

    symbol_count = len(deserialize_list(entry.serialized_symbol_ids))
    shards = symbol_count_to_shards(
      symbol_count=symbol_count,
      shard_size=shard_size,
    )

    unique_symbols_ids = set(model_symbols.get_symbols(entry.serialized_symbol_ids))
    unique_symbols_ids_str = " ".join(list(sorted(unique_symbols_ids)))
    unique_symbols_ids_count = len(unique_symbols_ids)
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
      wg_iteration=waveglow_checkpoint.iteration,
      taco_iteration=tacotron_checkpoint.iteration,
      shards=shards,
      unique_symbol_ids=unique_symbols_ids_str,
      unique_symbol_ids_count=unique_symbols_ids_count,
      symbol_count=symbol_count,
      timepoint=timepoint,
      train_name=train_name,
      sampling_rate=synth.get_sampling_rate(),
    )

    sentences = InferSentenceList([infer_sent])

    _, res = synth.infer(
      sentences=sentences,
      speaker=speaker_name,
      denoiser_strength=denoiser_strength,
      sentence_pause_s=0,
      sigma=sigma,
    )

    assert len(res) == 1
    inference_result = res[0]

    validation_entry_output = ValidationEntryOutput()

    validation_entry_output.inferred_wav = inference_result.wav
    validation_entry_output.inferred_wav_sr = inference_result.sampling_rate

    val_entry.reached_max_decoder_steps = inference_result.reached_max_decoder_steps
    val_entry.wg_inference_duration_s = inference_result.wg_inference_duration_s
    val_entry.taco_inference_duration_s = inference_result.taco_inference_duration_s
    val_entry.total_inference_duration_s = inference_result.wg_inference_duration_s + \
        inference_result.taco_inference_duration_s
    duration_s = get_duration_s(inference_result.wav, inference_result.sampling_rate)
    val_entry.inferred_duration_s = duration_s
    val_entry.diff_duration_s = abs(val_entry.inferred_duration_s - val_entry.original_duration_s)

    orig_wav, orig_sr = get_audio_and_sampling_rate_from_path(entry.wav_path)
    spectogram_orig = get_spectogram(
      orig_wav, n_fft=mcd_settings.n_fft, hop_length=mcd_settings.hop_length)
    spectogram_pred = get_spectogram(
      inference_result.wav, n_fft=mcd_settings.n_fft, hop_length=mcd_settings.hop_length)
    mel_spectogram_orig = get_mel_spectogram(
      spectogram_orig, sr=orig_sr, n_mels=mcd_settings.n_mels)
    mel_spectogram_pred = get_mel_spectogram(
      spectogram_pred, sr=synth.get_sampling_rate(), n_mels=mcd_settings.n_mels)
    mcd, frames = get_mcd_dtw_from_mel_spectograms(
      mel_spectogram_orig, mel_spectogram_pred, mcd_settings.no_of_coeffs_per_frame)

    val_entry.mcd_dtw = mcd
    val_entry.mcd_dtw_frames = frames

    val_entry.mcd_dtw_v2 = 0
    val_entry.mcd_dtw_v2_frames = 0

    if True:
      orig_mgc = get_mgc_wav_file(entry.wav_path)
      infered_wav_int = convert_wav(inference_result.wav, np.int16)
      infered_wav_int = infered_wav_int.astype(np.float64)
      inferred_mgc = get_mgc_wav(infered_wav_int)

      mcd_v2, frames_v2 = calcdist_mgc(
        x_mgc=orig_mgc,
        y_mgc=inferred_mgc,
      )

      val_entry.mcd_dtw_v2 = mcd_v2
      val_entry.mcd_dtw_v2_frames = frames_v2

    orig_mel = taco_stft.get_mel_tensor_from_file(entry.wav_path)
    orig_mel_np = orig_mel.cpu().numpy()

    # audio_tensor = torch.FloatTensor(inference_result.wav)
    # inferred_mel = taco_stft.get_mel_tensor(audio_tensor)
    # inferred_mel_np = inferred_mel.numpy()
    inferred_mel_np = inference_result.mel_outputs

    cos_sim = cosine_dist_mels(orig_mel_np, inferred_mel_np)
    val_entry.cosine_similarity = cos_sim

    orig_mel_plot_core, orig_mel_plot = plot_melspec_np(orig_mel_np)
    inferred_mel_plot_core, inferred_mel_plot = plot_melspec_np(inferred_mel_np)

    validation_entry_output.orig_wav_img = orig_mel_plot
    validation_entry_output.inferred_wav_img = inferred_mel_plot
    orig_wav, orig_sr = wav_to_float32(entry.wav_path)

    validation_entry_output.orig_wav = orig_wav
    validation_entry_output.orig_wav_sr = orig_sr

    imageio.imsave("/tmp/core_orig_mel.png", orig_mel_plot_core)
    imageio.imsave("/tmp/core_inferred_mel.png", inferred_mel_plot_core)
    orig_mel_plot_core, inferred_mel_plot_core = make_same_width_by_filling_white(
      img_a=orig_mel_plot_core,
      img_b=inferred_mel_plot_core,
    )

    structural_similarity, core_diff_img = compare_mels_core(
        img_a=orig_mel_plot_core,
        img_b=inferred_mel_plot_core,
    )
    imageio.imsave("/tmp/core_struc_sim_img.png", core_diff_img)
    val_entry.structural_similarity = structural_similarity

    orig_mel_plot, inferred_mel_plot = make_same_width_by_filling_white(
      img_a=orig_mel_plot,
      img_b=inferred_mel_plot,
    )
    _, diff_img = compare_mels_core(
        img_a=orig_mel_plot,
        img_b=inferred_mel_plot,
    )
    validation_entry_output.struc_sim_img = diff_img

    alignments_img = plot_alignment_np(inference_result.alignments)
    validation_entry_output.alignments_img = alignments_img
    _, post_mel_img = plot_melspec_np(inference_result.mel_outputs_postnet)

    # validation_entry_output.gate_out_img = None
    validation_entry_output.postnet_img = post_mel_img
    # val_entry.grad_norm = None
    # val_entry.loss = None

    # logger.info(val_entry)
    logger.info(f"MCD DTW: {val_entry.mcd_dtw}")
    logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
    logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
    logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")
    save_callback(entry, validation_entry_output)
    validation_entries.append(val_entry)
  return validation_entries
