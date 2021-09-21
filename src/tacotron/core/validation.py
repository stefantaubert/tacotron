import datetime
import random
from collections import OrderedDict
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set

import jiwer
import jiwer.transforms as tr
import numpy as np
import pandas as pd
from audio_utils.mel import (TacotronSTFT, align_mels_with_dtw, get_msd,
                             plot_melspec_np)
from general_utils import GenericList
from image_utils import calculate_structual_similarity_np
from mcd import get_mcd_between_mel_spectograms
from ordered_set import OrderedSet
from pandas import DataFrame
from scipy.io.wavfile import read
from sklearn.metrics import mean_squared_error
from tacotron.core.synthesizer import Synthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import (cosine_dist_mels, make_same_dim,
                            plot_alignment_np_new)
from text_selection import get_rarity_ngrams
from tts_preparation import InferableUtterance, PreparedData, PreparedDataList


@dataclass
class ValidationEntry():
  timepoint: datetime.datetime
  utterance: PreparedData
  repetition: int
  repetitions: int
  seed: int
  train_name: str
  iteration: int
  sampling_rate: int
  # grad_norm: float
  # loss: float
  inference_duration_s: float
  reached_max_decoder_steps: bool
  target_frames: int
  predicted_frames: int
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
  wer: float
  mer: float
  wil: float
  wip: float
  train_one_gram_rarity: float
  train_two_gram_rarity: float
  train_three_gram_rarity: float


class ValidationEntries(GenericList[ValidationEntry]):
  pass


def get_df(entries: ValidationEntries) -> DataFrame:
  data = [
    {
      "Id": entry.utterance.utterance_id,
      "Timepoint": f"{entry.timepoint:%Y/%m/%d %H:%M:%S}",
      "Iteration": entry.iteration,
      "Seed": entry.seed,
      "Repetition": entry.repetition,
      "Repetitions": entry.repetitions,
      "Language": repr(entry.utterance.symbols_language),
      "Symbols": ''.join(entry.utterance.symbols),
      "Symbols format": repr(entry.utterance.symbols_format),
      "Speaker": entry.utterance.speaker_name,
      "Speaker Id": entry.utterance.speaker_id,
      "Inference duration (s)": entry.inference_duration_s,
      "Reached max. steps": entry.reached_max_decoder_steps,
      "Sampling rate (Hz)": entry.sampling_rate,
      "# MFCC Coefficients": entry.mfcc_no_coeffs,
      "MFCC DTW MCD": entry.mfcc_dtw_mcd,
      "MFCC DTW PEN": entry.mfcc_dtw_penalty,
      "# MFCC DTW frames": entry.mfcc_dtw_frames,
      "# Target frames": entry.target_frames,
      "# Predicted frames": entry.predicted_frames,
      "# Difference frames": entry.predicted_frames - entry.target_frames,
      "Frames deviation (%)": (entry.predicted_frames / entry.target_frames) - 1,
      "MSE (Padded)": entry.padded_mse,
      "Cosine Similarity (Padded)": entry.padded_cosine_similarity,
      "Structual Similarity (Padded)": entry.padded_structural_similarity,
      "MSE (Aligned)": entry.aligned_mse,
      "Cosine Similarity (Aligned)": entry.aligned_cosine_similarity,
      "Structual Similarity (Aligned)": entry.aligned_structural_similarity,
      "MSD": entry.msd,
      "WER": entry.wer,
      "MER": entry.mer,
      "WIL": entry.wil,
      "WIP": entry.wip,
      "1-gram rarity (train set)": entry.train_one_gram_rarity,
      "2-gram rarity (train set)": entry.train_two_gram_rarity,
      "3-gram rarity (train set)": entry.train_three_gram_rarity,
      "Combined rarity (train set)": entry.train_one_gram_rarity + entry.train_two_gram_rarity + entry.train_three_gram_rarity,
      "1-gram rarity (total set)": entry.utterance.one_gram_rarity,
      "2-gram rarity (total set)": entry.utterance.two_gram_rarity,
      "3-gram rarity (total set)": entry.utterance.three_gram_rarity,
      "Combined rarity (total set)": entry.utterance.one_gram_rarity + entry.utterance.two_gram_rarity + entry.utterance.three_gram_rarity,
      "# Symbols": len(entry.utterance.symbols),
      "Unique symbols": ' '.join(sorted(set(entry.utterance.symbols))),
      "# Unique symbols": len(set(entry.utterance.symbols)),
      "Train name": entry.train_name,
    }
    for entry in entries.items()
  ]

  if len(data) == 0:
    return DataFrame()

  df = DataFrame(
    data=[x.values() for x in data],
    columns=data[0].keys(),
  )

  return df


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


def get_ngram_rarity(data: PreparedDataList, corpus: PreparedDataList, ngram: int) -> OrderedDictType[int, float]:
  data_symbols_dict = OrderedDict({x.entry_id: x.symbols for x in data.items()})
  corpus_symbols_dict = OrderedDict({x.entry_id: x.symbols for x in corpus.items()})

  rarity = get_rarity_ngrams(
    data=data_symbols_dict,
    corpus=corpus_symbols_dict,
    n_gram=ngram,
    ignore_symbols=None,
  )

  return rarity


def get_best_seeds(select_best_from: pd.DataFrame, entry_ids: OrderedSet[int], iteration: int, logger: Logger) -> List[int]:
  df = select_best_from.loc[select_best_from['iteration'] == iteration]
  result = []
  for entry_id in entry_ids:
    logger.info(f"Entry {entry_id}")
    entry_df = df.loc[df['entry_id'] == entry_id]
    # best_row = entry_df.iloc[[entry_df["mfcc_dtw_mcd"].argmin()]]
    # worst_row = entry_df.iloc[[entry_df["mfcc_dtw_mcd"].argmax()]]

    metric_values = []

    for _, row in entry_df.iterrows():
      best_value = row['mfcc_dtw_mcd'] + row['mfcc_dtw_penalty']
      metric_values.append(best_value)

    logger.info(f"Mean MCD: {entry_df['mfcc_dtw_mcd'].mean()}")
    logger.info(f"Mean PEN: {entry_df['mfcc_dtw_penalty'].mean()}")
    logger.info(f"Mean metric: {np.mean(metric_values)}")

    best_idx = np.argmin(metric_values)
    best_row = entry_df.iloc[best_idx]
    test_x = best_row['mfcc_dtw_mcd'] + best_row['mfcc_dtw_penalty']
    assert test_x == metric_values[best_idx]

    worst_idx = np.argmax(metric_values)
    worst_row = entry_df.iloc[worst_idx]
    test_x = worst_row['mfcc_dtw_mcd'] + worst_row['mfcc_dtw_penalty']
    assert test_x == metric_values[worst_idx]

    seed = int(best_row["seed"])
    logger.info(
      f"The best seed was {seed} with an MCD of {float(best_row['mfcc_dtw_mcd']):.2f} and a PEN of {float(best_row['mfcc_dtw_penalty']):.5f} -> {metric_values[best_idx]:.5f}")
    logger.info(
      f"The worst seed was {int(worst_row['seed'])} with an MCD of {float(worst_row['mfcc_dtw_mcd']):.2f} and a PEN of {float(worst_row['mfcc_dtw_penalty']):.5f} -> {metric_values[worst_idx]:.5f}")
    # print(f"The worst mcd was {}")
    logger.info("------")
    result.append(seed)
  return result


def wav_to_text(wav: np.ndarray) -> str:
  return ""


def validate(checkpoint: CheckpointTacotron, data: PreparedDataList, trainset: PreparedDataList, custom_hparams: Optional[Dict[str, str]], entry_ids: Optional[Set[int]], speaker_name: Optional[str], train_name: str, full_run: bool, save_callback: Optional[Callable[[PreparedData, ValidationEntryOutput], None]], max_decoder_steps: int, fast: bool, mcd_no_of_coeffs_per_frame: int, repetitions: int, seed: Optional[int], select_best_from: Optional[pd.DataFrame], logger: Logger) -> ValidationEntries:
  seeds: List[int]
  validation_data: PreparedDataList

  if full_run:
    validation_data = data
    assert seed is not None
    seeds = [seed for _ in data]
  elif select_best_from is not None:
    assert entry_ids is not None
    logger.info("Finding best seeds...")
    validation_data = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
    if len(validation_data) != len(entry_ids):
      logger.error("Not all entry_id's were found!")
      assert False
    entry_ids_order_from_valdata = OrderedSet([x.entry_id for x in validation_data.items()])
    seeds = get_best_seeds(select_best_from, entry_ids_order_from_valdata,
                           checkpoint.iteration, logger)
  #   entry_ids = [entry_id for entry_id, _ in entry_ids_w_seed]
  #   have_no_double_entries = len(set(entry_ids)) == len(entry_ids)
  #   assert have_no_double_entries
  #   validation_data = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
  #   seeds = [s for _, s in entry_ids_w_seed]
  #   if len(validation_data) != len(entry_ids):
  #     logger.error("Not all entry_id's were found!")
  #     assert False
  elif entry_ids is not None:
    validation_data = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
    if len(validation_data) != len(entry_ids):
      logger.error("Not all entry_id's were found!")
      assert False
    assert seed is not None
    seeds = [seed for _ in validation_data]
  elif speaker_name is not None:
    relevant_entries = [x for x in data.items() if x.speaker_name == speaker_name]
    assert len(relevant_entries) > 0
    assert seed is not None
    random.seed(seed)
    entry = random.choice(relevant_entries)
    validation_data = PreparedDataList([entry])
    seeds = [seed]
  else:
    assert seed is not None
    random.seed(seed)
    entry = random.choice(data)
    validation_data = PreparedDataList([entry])
    seeds = [seed]

  assert len(seeds) == len(validation_data)

  if len(validation_data) == 0:
    logger.info("Nothing to synthesize!")
    return validation_data

  train_onegram_rarities = get_ngram_rarity(validation_data, trainset, 1)
  train_twogram_rarities = get_ngram_rarity(validation_data, trainset, 2)
  train_threegram_rarities = get_ngram_rarity(validation_data, trainset, 3)

  synth = Synthesizer(
      checkpoint=checkpoint,
      custom_hparams=custom_hparams,
      logger=logger,
  )

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)
  validation_entries = ValidationEntries()

  jiwer_ground_truth_transform = tr.Compose([
    tr.ToLowerCase(),
    tr.SentencesToListOfWords(),
  ])

  jiwer_inferred_asr_transform = tr.Compose([
    tr.ToLowerCase(),
    tr.SentencesToListOfWords(),
  ])

  #asr_pipeline = asr.load('deepspeech2', lang='en')
  # asr_pipeline.model.summary()

  for repetition in range(repetitions):
    # rep_seed = seed + repetition
    rep_human_readable = repetition + 1
    logger.info(f"Starting repetition: {rep_human_readable}/{repetitions}")
    for entry, entry_seed in zip(validation_data.items(True), seeds):
      rep_seed = entry_seed + repetition
      logger.info(
        f"Current --> entry_id: {entry.entry_id}; seed: {rep_seed}; iteration: {checkpoint.iteration}; rep: {rep_human_readable}/{repetitions}")

      infer_sent = InferableUtterance(
        utterance_id=1,
        symbols=entry.symbols,
        language=entry.symbols_language,
        symbol_ids=entry.symbol_ids,
        symbols_format=entry.symbols_format,
      )

      timepoint = datetime.datetime.now()
      inference_result = synth.infer(
        utterance=infer_sent,
        speaker=entry.speaker_name,
        max_decoder_steps=max_decoder_steps,
        seed=rep_seed,
      )

      mel_orig: np.ndarray = taco_stft.get_mel_tensor_from_file(
        entry.wav_absolute_path).cpu().numpy()

      target_frames = mel_orig.shape[1]
      predicted_frames = inference_result.mel_outputs_postnet.shape[1]

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

      ground_truth = ''.join(entry.symbols)
      #hypothesis = asr_pipeline.predict([inference_result.mel_outputs_postnet])[0]
      hypothesis = ""

      measures = jiwer.compute_measures(
        truth=ground_truth,
        truth_transform=jiwer_ground_truth_transform,
        hypothesis=hypothesis,
        hypothesis_transform=jiwer_inferred_asr_transform,
      )

      wer = measures['wer']
      mer = measures['mer']
      wil = measures['wil']
      wip = measures['wip']

      if not fast:
        padded_mel_orig_img_raw_1, padded_mel_orig_img = plot_melspec_np(
          padded_mel_orig, title="padded_mel_orig")
        padded_mel_outputs_postnet_img_raw_1, padded_mel_outputs_postnet_img = plot_melspec_np(
          padded_mel_postnet, title="padded_mel_postnet")

        # imageio.imsave("/tmp/padded_mel_orig_img_raw_1.png", padded_mel_orig_img_raw_1)
        # imageio.imsave("/tmp/padded_mel_outputs_postnet_img_raw_1.png", padded_mel_outputs_postnet_img_raw_1)

        padded_structural_similarity, padded_mel_postnet_diff_img_raw = calculate_structual_similarity_np(
            img_a=padded_mel_orig_img_raw_1,
            img_b=padded_mel_outputs_postnet_img_raw_1,
        )

        # imageio.imsave("/tmp/padded_mel_diff_img_raw_1.png", padded_mel_postnet_diff_img_raw)

        aligned_mel_orig_img_raw, aligned_mel_orig_img = plot_melspec_np(
          aligned_mel_orig, title="aligned_mel_orig")
        aligned_mel_postnet_img_raw, aligned_mel_postnet_img = plot_melspec_np(
          aligned_mel_postnet, title="aligned_mel_postnet")

        # imageio.imsave("/tmp/aligned_mel_orig_img_raw.png", aligned_mel_orig_img_raw)
        # imageio.imsave("/tmp/aligned_mel_postnet_img_raw.png", aligned_mel_postnet_img_raw)

        aligned_structural_similarity, aligned_mel_diff_img_raw = calculate_structual_similarity_np(
            img_a=aligned_mel_orig_img_raw,
            img_b=aligned_mel_postnet_img_raw,
        )

        # imageio.imsave("/tmp/aligned_mel_diff_img_raw.png", aligned_mel_diff_img_raw)

      val_entry = ValidationEntry(
        utterance=entry,
        repetition=rep_human_readable,
        repetitions=repetitions,
        seed=rep_seed,
        iteration=checkpoint.iteration,
        timepoint=timepoint,
        train_name=train_name,
        sampling_rate=synth.get_sampling_rate(),
        reached_max_decoder_steps=inference_result.reached_max_decoder_steps,
        inference_duration_s=inference_result.inference_duration_s,
        predicted_frames=predicted_frames,
        target_frames=target_frames,
        padded_cosine_similarity=padded_cosine_similarity,
        mfcc_no_coeffs=mcd_no_of_coeffs_per_frame,
        mfcc_dtw_mcd=dtw_mcd,
        mfcc_dtw_penalty=dtw_penalty,
        mfcc_dtw_frames=dtw_frames,
        padded_structural_similarity=padded_structural_similarity,
        padded_mse=padded_mse,
        msd=msd,
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        aligned_cosine_similarity=aligned_cosine_similarity,
        aligned_mse=aligned_mse,
        aligned_structural_similarity=aligned_structural_similarity,
        train_one_gram_rarity=train_onegram_rarities[entry.entry_id],
        train_two_gram_rarity=train_twogram_rarities[entry.entry_id],
        train_three_gram_rarity=train_threegram_rarities[entry.entry_id],
      )

      validation_entries.append(val_entry)

      if not fast:
        orig_sr, orig_wav = read(entry.wav_absolute_path)

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
