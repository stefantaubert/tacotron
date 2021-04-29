import datetime
import random
from collections import OrderedDict
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

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
from tacotron.utils import (GenericList, cosine_dist_mels, init_global_seeds,
                            make_same_dim, plot_alignment_np,
                            plot_alignment_np_new)
from text_utils import deserialize_list
from text_utils.symbol_id_dict import SymbolIdDict
from text_utils.text_selection import get_rarity_ngrams
from tts_preparation import InferSentence, PreparedData, PreparedDataList


@dataclass
class ValidationEntry():
  timepoint: str
  repetition: int
  repetitions: int
  seed: int
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
  frame_deviation_percent: float
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
  repetition: int
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


def validate(checkpoint: CheckpointTacotron, data: PreparedDataList, trainset: PreparedDataList, custom_hparams: Optional[Dict[str, str]], entry_ids: Optional[Set[int]], entry_ids_w_seed: Optional[List[Tuple[int, int]]], speaker_name: Optional[str], train_name: str, full_run: bool, save_callback: Optional[Callable[[PreparedData, ValidationEntryOutput], None]], max_decoder_steps: int, fast: bool, mcd_no_of_coeffs_per_frame: int, repetitions: int, seed: int, logger: Logger) -> ValidationEntries:
  model_symbols = checkpoint.get_symbols()
  model_accents = checkpoint.get_accents()
  model_speakers = checkpoint.get_speakers()

  seeds: List[int]
  validation_data: PreparedDataList

  if full_run:
    validation_data = data
    seeds = [seed for _ in data]
  elif entry_ids_w_seed is not None:
    entry_ids = [entry_id for entry_id, _ in entry_ids_w_seed]
    have_no_double_entries = len(set(entry_ids)) == len(entry_ids)
    assert have_no_double_entries
    validation_data = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
    seeds = [s for _, s in entry_ids_w_seed]
  elif entry_ids is not None:
    validation_data = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
    seeds = [seed for _ in validation_data]
  elif speaker_name is not None:
    speaker_id = model_speakers.get_id(speaker_name)
    relevant_entries = [x for x in data.items() if x.speaker_id == speaker_id]
    assert len(relevant_entries) > 0
    random.seed(seed)
    entry = random.choice(relevant_entries)
    validation_data = PreparedDataList([entry])
    seeds = [seed]
  else:
    random.seed(seed)
    entry = random.choice(data)
    validation_data = PreparedDataList([entry])
    seeds = [seed]

  assert len(seeds) == len(validation_data)

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

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)
  validation_entries = ValidationEntries()

  for repetition in range(repetitions):
    #rep_seed = seed + repetition
    rep_human_readable = repetition + 1
    logger.info(f"Starting repetition: {rep_human_readable}/{repetitions}")
    for entry, entry_seed in zip(validation_data.items(True), seeds):
      rep_seed = entry_seed + repetition

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
        seed=rep_seed,
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
      frame_deviation_percent = (predicted_frames / target_frames) - 1

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

        #imageio.imsave("/tmp/padded_mel_orig_img_raw_1.png", padded_mel_orig_img_raw_1)
        #imageio.imsave("/tmp/padded_mel_outputs_postnet_img_raw_1.png", padded_mel_outputs_postnet_img_raw_1)

        padded_structural_similarity, padded_mel_postnet_diff_img_raw = calculate_structual_similarity_np(
            img_a=padded_mel_orig_img_raw_1,
            img_b=padded_mel_outputs_postnet_img_raw_1,
        )

        #imageio.imsave("/tmp/padded_mel_diff_img_raw_1.png", padded_mel_postnet_diff_img_raw)

        aligned_mel_orig_img_raw, aligned_mel_orig_img = plot_melspec_np(
          aligned_mel_orig, title="aligned_mel_orig")
        aligned_mel_postnet_img_raw, aligned_mel_postnet_img = plot_melspec_np(
          aligned_mel_postnet, title="aligned_mel_postnet")

        #imageio.imsave("/tmp/aligned_mel_orig_img_raw.png", aligned_mel_orig_img_raw)
        #imageio.imsave("/tmp/aligned_mel_postnet_img_raw.png", aligned_mel_postnet_img_raw)

        aligned_structural_similarity, aligned_mel_diff_img_raw = calculate_structual_similarity_np(
            img_a=aligned_mel_orig_img_raw,
            img_b=aligned_mel_postnet_img_raw,
        )

        #imageio.imsave("/tmp/aligned_mel_diff_img_raw.png", aligned_mel_diff_img_raw)

      train_combined_rarity = train_onegram_rarities[entry.entry_id] + \
          train_twogram_rarities[entry.entry_id] + train_threegram_rarities[entry.entry_id]

      val_entry = ValidationEntry(
        entry_id=entry.entry_id,
        repetition=rep_human_readable,
        repetitions=repetitions,
        seed=rep_seed,
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
        frame_deviation_percent=frame_deviation_percent,
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
          repetition=rep_human_readable,
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
