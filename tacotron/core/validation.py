import datetime
from collections import OrderedDict
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set

import imageio
import numpy as np
from audio_utils.mel import (TacotronSTFT, align_mels_with_dtw, get_msd,
                             plot_melspec_np)
from image_utils import (calculate_structual_similarity_np,
                         make_same_width_by_filling_white)
from mcd import get_mcd_between_mel_spectograms
from scipy.io.wavfile import read
from sklearn.metrics import mean_squared_error
from tacotron.core.synthesizer import Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import (GenericList, cosine_dist_mels, make_same_dim,
                            plot_alignment_np, plot_alignment_np_new)
from text_utils import deserialize_list
from text_utils.symbol_id_dict import SymbolIdDict
from text_utils.text_selection import get_rarity_ngrams
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
  aligned_mse: float
  aligned_cosine_similarity: Optional[float]
  aligned_structural_similarity: Optional[float]
  mfcc_no_coeffs: int
  mfcc_dtw_mcd: float
  mfcc_dtw_penalty: float
  mfcc_dtw_frames: int
  msd: float
  train_one_gram_rarity: float
  global_one_gram_rarity: float
  train_two_gram_rarity: float
  global_two_gram_rarity: float
  train_three_gram_rarity: float
  global_three_gram_rarity: float
  train_combined_rarity: float
  global_combined_rarity: float


class ValidationEntries(GenericList[ValidationEntry]):
  pass


@dataclass
class ValidationEntryOutput():
  wav_orig: np.ndarray
  mel_orig: np.ndarray
  mel_orig_aligned: np.ndarray
  orig_sr: int
  mel_orig_img: np.ndarray
  mel_orig_aligned_img: np.ndarray
  mel_postnet: np.ndarray
  mel_postnet_aligned: np.ndarray
  mel_postnet_sr: int
  mel_postnet_img: np.ndarray
  mel_postnet_aligned_img: np.ndarray
  mel_postnet_diff_img: np.ndarray
  mel_postnet_aligned_diff_img: np.ndarray
  mel_img: np.ndarray
  alignments_img: np.ndarray
  alignments_aligned_img: np.ndarray
  # gate_out_img: np.ndarray


class ValidationEntryOutputs(GenericList[ValidationEntryOutput]):
  pass


def get_ngram_rarity(data: PreparedDataList, corpus: PreparedDataList, symbols: SymbolIdDict, ngram: int) -> OrderedDictType[int, float]:
  data_symbols_dict = OrderedDict({x.entry_id: symbols.get_symbols(
    x.serialized_symbol_ids) for x in data.items()})
  corpus_symbols_dict = OrderedDict({x.entry_id: symbols.get_symbols(
    x.serialized_symbol_ids) for x in corpus.items()})

  rarity = get_rarity_ngrams(
    data=data_symbols_dict,
    corpus=corpus_symbols_dict,
    n_gram=ngram,
    ignore_symbols=None,
  )

  return rarity


def validate(checkpoint: CheckpointTacotron, data: PreparedDataList, trainset: PreparedDataList, custom_hparams: Optional[Dict[str, str]], entry_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, save_callback: Optional[Callable[[PreparedData, ValidationEntryOutput], None]], max_decoder_steps: int, fast: bool, mcd_no_of_coeffs_per_frame: int, logger: Logger) -> ValidationEntries:
  model_symbols = checkpoint.get_symbols()
  model_accents = checkpoint.get_accents()
  model_speakers = checkpoint.get_speakers()

  if full_run:
    validation_data = data
  else:
    speaker_id: Optional[int] = None
    if speaker_name is not None:
      speaker_id = model_speakers.get_id(speaker_name)
    validation_data = PreparedDataList(data.get_for_validation(entry_ids, speaker_id))

  if len(validation_data) == 0:
    logger.info("Nothing to synthesize!")
    return validation_data

  train_onegram_rarities = get_ngram_rarity(validation_data, trainset, model_symbols, 1)
  train_twogram_rarities = get_ngram_rarity(validation_data, trainset, model_symbols, 2)
  train_threegram_rarities = get_ngram_rarity(validation_data, trainset, model_symbols, 3)

  synth = Synthesizer(
      checkpoint=checkpoint,
      custom_hparams=custom_hparams,
      logger=logger,
  )

  # criterion = Tacotron2Loss()

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)
  validation_entries = ValidationEntries()

  for entry in validation_data.items(True):
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
      n_mfcc=mcd_no_of_coeffs_per_frame,
      take_log=False,
      use_dtw=True,
    )

    padded_mel_orig, padded_mel_postnet = make_same_dim(
      mel_orig, inference_result.mel_outputs_postnet)

    aligned_mel_orig, aligned_mel_postnet, mel_dtw_dist, _, path_mel_postnet = align_mels_with_dtw(
      mel_orig, inference_result.mel_outputs_postnet)

    mel_aligned_length = len(aligned_mel_postnet)
    msd = get_msd(mel_dtw_dist, mel_aligned_length)

    padded_cosine_similarity = cosine_dist_mels(padded_mel_orig, padded_mel_postnet)
    aligned_cosine_similarity = cosine_dist_mels(aligned_mel_orig, aligned_mel_postnet)

    padded_mse = mean_squared_error(padded_mel_orig, padded_mel_postnet)
    aligned_mse = mean_squared_error(aligned_mel_orig, aligned_mel_postnet)

    padded_structural_similarity = None
    aligned_structural_similarity = None

    if not fast:
      padded_mel_orig_img_raw_1, padded_mel_orig_img = plot_melspec_np(
        padded_mel_orig, title="padded_mel_orig")
      padded_mel_outputs_postnet_img_raw_1, padded_mel_outputs_postnet_img = plot_melspec_np(
        padded_mel_postnet, title="padded_mel_postnet")

      imageio.imsave("/tmp/padded_mel_orig_img_raw_1.png", padded_mel_orig_img_raw_1)
      imageio.imsave("/tmp/padded_mel_outputs_postnet_img_raw_1.png",
                     padded_mel_outputs_postnet_img_raw_1)

      padded_structural_similarity, padded_mel_postnet_diff_img_raw = calculate_structual_similarity_np(
          img_a=padded_mel_orig_img_raw_1,
          img_b=padded_mel_outputs_postnet_img_raw_1,
      )

      imageio.imsave("/tmp/padded_mel_diff_img_raw_1.png", padded_mel_postnet_diff_img_raw)

      # mel_orig_img_raw, mel_orig_img = plot_melspec_np(mel_orig)
      # mel_outputs_postnet_img_raw, mel_outputs_postnet_img = plot_melspec_np(
      #   inference_result.mel_outputs_postnet)

      # padded_mel_orig_img_raw, padded_mel_outputs_postnet_img_raw = make_same_width_by_filling_white(
      #   img_a=mel_orig_img_raw,
      #   img_b=mel_outputs_postnet_img_raw,
      # )

      # imageio.imsave("/tmp/padded_mel_orig_img_raw.png", padded_mel_orig_img_raw)
      # imageio.imsave("/tmp/padded_mel_outputs_postnet_img_raw.png",
      #                padded_mel_outputs_postnet_img_raw)

      # padded_structural_similarity, padded_mel_diff_img_raw = calculate_structual_similarity_np(
      #     img_a=padded_mel_orig_img_raw,
      #     img_b=padded_mel_outputs_postnet_img_raw,
      # )

      # imageio.imsave("/tmp/padded_mel_diff_img_raw.png", padded_mel_diff_img_raw)

      aligned_mel_orig_img_raw, aligned_mel_orig_img = plot_melspec_np(
        aligned_mel_orig, title="aligned_mel_orig")
      aligned_mel_postnet_img_raw, aligned_mel_postnet_img = plot_melspec_np(
        aligned_mel_postnet, title="aligned_mel_postnet")

      imageio.imsave("/tmp/aligned_mel_orig_img_raw.png", aligned_mel_orig_img_raw)
      imageio.imsave("/tmp/aligned_mel_postnet_img_raw.png",
                     aligned_mel_postnet_img_raw)

      aligned_structural_similarity, aligned_mel_diff_img_raw = calculate_structual_similarity_np(
          img_a=aligned_mel_orig_img_raw,
          img_b=aligned_mel_postnet_img_raw,
      )

      imageio.imsave("/tmp/aligned_mel_diff_img_raw.png", aligned_mel_diff_img_raw)

      # imageio.imsave("/tmp/mel_orig_img_raw.png", mel_orig_img_raw)
      # imageio.imsave("/tmp/mel_outputs_postnet_img_raw.png", mel_outputs_postnet_img_raw)
      # imageio.imsave("/tmp/mel_diff_img_raw.png", mel_diff_img_raw)

    train_combined_rarity = train_onegram_rarities[entry.entry_id] + \
        train_twogram_rarities[entry.entry_id] + train_threegram_rarities[entry.entry_id]

    val_entry = ValidationEntry(
      entry_id=entry.entry_id,
      ds_entry_id=entry.ds_entry_id,
      text_original=entry.text_original,
      text=entry.text,
      wav_path=entry.wav_path,
      wav_duration_s=entry.duration_s,
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
      padded_cosine_similarity=padded_cosine_similarity,
      mfcc_no_coeffs=mcd_no_of_coeffs_per_frame,
      mfcc_dtw_mcd=dtw_mcd,
      mfcc_dtw_penalty=dtw_penalty,
      mfcc_dtw_frames=dtw_frames,
      padded_structural_similarity=padded_structural_similarity,
      padded_mse=padded_mse,
      msd=msd,
      aligned_cosine_similarity=aligned_cosine_similarity,
      aligned_mse=aligned_mse,
      aligned_structural_similarity=aligned_structural_similarity,
      global_one_gram_rarity=entry.one_gram_rarity,
      global_two_gram_rarity=entry.two_gram_rarity,
      global_three_gram_rarity=entry.three_gram_rarity,
      global_combined_rarity=entry.combined_rarity,
      train_one_gram_rarity=train_onegram_rarities[entry.entry_id],
      train_two_gram_rarity=train_twogram_rarities[entry.entry_id],
      train_three_gram_rarity=train_threegram_rarities[entry.entry_id],
      train_combined_rarity=train_combined_rarity,
    )

    validation_entries.append(val_entry)

    # logger.info(val_entry)
    # logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
    # logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
    # logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")

    if not fast:
      orig_sr, orig_wav = read(entry.wav_path)

      _, padded_mel_postnet_diff_img = calculate_structual_similarity_np(
        img_a=padded_mel_orig_img,
        img_b=padded_mel_outputs_postnet_img,
      )

      _, aligned_mel_postnet_diff_img = calculate_structual_similarity_np(
        img_a=aligned_mel_orig_img,
        img_b=aligned_mel_postnet_img,
      )

      _, mel_orig_img = plot_melspec_np(mel_orig, title="mel_orig")
      # alignments_img = plot_alignment_np(inference_result.alignments)
      _, post_mel_img = plot_melspec_np(inference_result.mel_outputs_postnet, title="mel_postnet")
      _, mel_img = plot_melspec_np(
        inference_result.mel_outputs, title="mel")
      _, alignments_img = plot_alignment_np_new(inference_result.alignments, title="alignments")

      aligned_alignments = inference_result.alignments[path_mel_postnet]
      _, aligned_alignments_img = plot_alignment_np_new(
        aligned_alignments, title="aligned_alignments")
      # imageio.imsave("/tmp/alignments.png", alignments_img)

      validation_entry_output = ValidationEntryOutput(
        wav_orig=orig_wav,
        mel_orig=mel_orig,
        orig_sr=orig_sr,
        mel_postnet=inference_result.mel_outputs_postnet,
        mel_postnet_sr=inference_result.sampling_rate,
        mel_orig_img=mel_orig_img,
        mel_postnet_img=post_mel_img,
        mel_postnet_diff_img=padded_mel_postnet_diff_img,
        alignments_img=alignments_img,
        mel_img=mel_img,
        mel_postnet_aligned_diff_img=aligned_mel_postnet_diff_img,
        mel_orig_aligned=aligned_mel_orig,
        mel_orig_aligned_img=aligned_mel_orig_img,
        mel_postnet_aligned=aligned_mel_postnet,
        mel_postnet_aligned_img=aligned_mel_postnet_img,
        alignments_aligned_img=aligned_alignments_img,
      )

      assert save_callback is not None

      save_callback(entry, validation_entry_output)

    logger.info(f"MFCC MCD DTW: {val_entry.mfcc_dtw_mcd}")

  return validation_entries
